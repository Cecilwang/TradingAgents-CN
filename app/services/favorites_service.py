"""
自选股服务
"""

import re
from typing import List, Optional, Dict, Any
from datetime import datetime
from bson import ObjectId

from app.core.database import get_mongo_db
from app.utils.report_helpers import extract_report_action, extract_report_target_price
from app.services.quotes_service import get_quotes_service


class FavoritesService:
    """自选股服务类"""
    
    def __init__(self):
        self.db = None
    
    async def _get_db(self):
        """获取数据库连接"""
        if self.db is None:
            self.db = get_mongo_db()
        return self.db

    def _is_valid_object_id(self, user_id: str) -> bool:
        """
        检查是否是有效的ObjectId格式
        注意：这里只检查格式，不代表数据库中实际存储的是ObjectId类型
        为了兼容性，我们统一使用 user_favorites 集合存储自选股
        """
        # 强制返回 False，统一使用 user_favorites 集合
        return False

    def _format_favorite(self, favorite: Dict[str, Any]) -> Dict[str, Any]:
        """格式化收藏条目（仅基础信息，不包含实时行情）。
        行情将在 get_user_favorites 中批量富集。
        """
        added_at = favorite.get("added_at")
        if isinstance(added_at, datetime):
            added_at = added_at.isoformat()
        return {
            "stock_code": favorite.get("stock_code"),
            "stock_name": favorite.get("stock_name"),
            "market": favorite.get("market", "A股"),
            "added_at": added_at,
            "tags": favorite.get("tags", []),
            "notes": favorite.get("notes", ""),
            "alert_price_high": favorite.get("alert_price_high"),
            "alert_price_low": favorite.get("alert_price_low"),
            # 行情占位，稍后填充
            "current_price": None,
            "change_percent": None,
            "volume": None,
            "latest_report_action": None,
            "latest_report_target_price": None,
        }

    def _infer_market_from_code(self, stock_code: Optional[str]) -> str:
        """根据股票代码推断市场。"""
        code = str(stock_code or "").strip().upper()
        if not code:
            return "CN"
        if code.endswith(".HK") or re.match(r"^\d{4,5}$", code):
            return "HK"
        if code.endswith(".US"):
            return "US"
        if re.match(r"^\d{6}(\.(SH|SZ|BJ))?$", code):
            return "CN"
        return "US"

    def _normalize_market(self, market: Optional[str], stock_code: Optional[str]) -> str:
        """标准化市场标识。"""
        value = str(market or "").strip()
        if value in {"A股", "沪深", "CN"}:
            return "CN"
        if value in {"港股", "HK"}:
            return "HK"
        if value in {"美股", "US"}:
            return "US"
        return self._infer_market_from_code(stock_code)

    def _normalize_code_for_market(self, stock_code: Optional[str], market: str) -> str:
        """按市场标准化股票代码。"""
        code = str(stock_code or "").strip().upper()
        if not code:
            return ""

        if market == "CN":
            matched = re.match(r"^(\d{6})\.(SH|SZ|BJ)$", code)
            if matched:
                return matched.group(1)
            digits = "".join(ch for ch in code if ch.isdigit())
            if digits and len(digits) <= 6:
                return digits.zfill(6)
            return code

        if market == "HK":
            if code.endswith(".HK"):
                code = code[:-3]
            digits = "".join(ch for ch in code if ch.isdigit())
            if digits:
                return digits.zfill(5)
            return code

        if code.endswith(".US"):
            return code[:-3]
        return code

    def _apply_quote(self, item: Dict[str, Any], quote: Dict[str, Any]) -> None:
        """把行情字典写回自选项。"""
        current_price = quote.get("close")
        if current_price is None:
            current_price = quote.get("price")
        if current_price is None:
            current_price = quote.get("current_price")

        change_percent = quote.get("pct_chg")
        if change_percent is None:
            change_percent = quote.get("change_percent")

        item["current_price"] = current_price
        item["change_percent"] = change_percent
        item["volume"] = quote.get("volume")

    def _build_targets(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """为自选项补齐市场和标准化代码。"""
        targets: List[Dict[str, Any]] = []
        for item in items:
            market = self._normalize_market(item.get("market"), item.get("stock_code"))
            code = self._normalize_code_for_market(item.get("stock_code"), market)
            targets.append({
                "item": item,
                "market": market,
                "code": code,
            })
        return targets

    def _build_task_user_candidates(self, user_id: str) -> List[Any]:
        """兼容 analysis_tasks 中的字符串/ObjectId 用户ID。"""
        candidates: List[Any] = [user_id]
        try:
            candidates.append(ObjectId(user_id))
        except Exception:
            pass
        return candidates

    def _build_task_symbol_regex(self, code: str, market: str) -> str:
        """为最近分析报告查询构造股票代码匹配规则。"""
        escaped_code = re.escape(code)
        if market == "CN":
            return rf"^{escaped_code}(?:\.(?:SH|SZ|BJ))?$"
        if market == "HK":
            digits = re.escape(code.lstrip("0") or "0")
            return rf"^0*{digits}(?:\.HK)?$"
        return rf"^{escaped_code}(?:\.US)?$"

    async def _enrich_latest_report_actions(
        self,
        db,
        user_id: str,
        targets: List[Dict[str, Any]],
    ) -> None:
        """补充当前用户最近一份分析报告的执行建议。"""
        user_candidates = self._build_task_user_candidates(user_id)
        projection = {
            "_id": 0,
            "result.recommendation": 1,
            "result.decision": 1,
            "result.reports": 1,
        }

        for target in targets:
            code = target.get("code")
            market = target.get("market")
            if not code or not market:
                continue

            symbol_regex = self._build_task_symbol_regex(code, market)
            query = {
                "$and": [
                    {"status": "completed"},
                    {"result": {"$exists": True, "$ne": None}},
                    {"$or": [
                        {"user_id": {"$in": user_candidates}},
                        {"user": {"$in": user_candidates}},
                    ]},
                    {"$or": [
                        {"symbol": {"$regex": symbol_regex, "$options": "i"}},
                        {"stock_code": {"$regex": symbol_regex, "$options": "i"}},
                        {"result.stock_symbol": {"$regex": symbol_regex, "$options": "i"}},
                        {"result.stock_code": {"$regex": symbol_regex, "$options": "i"}},
                    ]},
                ]
            }

            doc = await db.analysis_tasks.find_one(
                query,
                projection,
                sort=[("completed_at", -1), ("created_at", -1)],
            )
            result = (doc or {}).get("result") or {}
            target["item"]["latest_report_action"] = extract_report_action(result) or None
            target["item"]["latest_report_target_price"] = extract_report_target_price(result)

    async def _load_quotes_from_mongo(
        self,
        db,
        collection_name: str,
        codes: List[str],
        market: str,
    ) -> Dict[str, Dict[str, Any]]:
        """从指定 market_quotes 集合批量读取行情。"""
        if not codes:
            return {}

        cursor = db[collection_name].find(
            {"code": {"$in": codes}},
            {"code": 1, "close": 1, "pct_chg": 1, "price": 1, "change_percent": 1, "volume": 1, "_id": 0},
        )
        docs = await cursor.to_list(length=None)
        return {
            self._normalize_code_for_market(doc.get("code"), market): doc
            for doc in (docs or [])
        }

    async def get_user_favorites_cn(
        self,
        db,
        targets: List[Dict[str, Any]],
        allow_online_quote_fallback: bool = True,
    ) -> None:
        """处理A股自选项的基础信息和行情富集。"""
        codes = [target["code"] for target in targets if target["code"]]
        if not codes:
            return

        try:
            # 只按当前优先级最高的A股数据源补基础信息。
            from app.core.unified_config import UnifiedConfigManager

            config = UnifiedConfigManager()
            data_source_configs = await config.get_data_source_configs_async()
            enabled_sources = [
                ds.type.lower() for ds in data_source_configs
                if ds.enabled and ds.type.lower() in ['tushare', 'akshare', 'baostock']
            ]
            if not enabled_sources:
                enabled_sources = ['tushare', 'akshare', 'baostock']

            preferred_source = enabled_sources[0] if enabled_sources else 'tushare'
            cursor = db["stock_basic_info"].find(
                {"code": {"$in": codes}, "source": preferred_source},
                {"code": 1, "sse": 1, "market": 1, "_id": 0}
            )
            basic_docs = await cursor.to_list(length=None)
            basic_map = {str(doc.get("code")).zfill(6): doc for doc in (basic_docs or [])}

            for target in targets:
                basic = basic_map.get(target["code"])
                if basic:
                    target["item"]["board"] = basic.get("market", "-")
                    target["item"]["exchange"] = basic.get("sse", "-")
                else:
                    target["item"]["board"] = "-"
                    target["item"]["exchange"] = "-"
        except Exception:
            for target in targets:
                target["item"]["board"] = "-"
                target["item"]["exchange"] = "-"

        quotes_map = await self._load_quotes_from_mongo(db, "market_quotes", codes, "CN")
        missing_codes: List[str] = []
        for target in targets:
            quote = quotes_map.get(target["code"])
            if quote:
                self._apply_quote(target["item"], quote)
            else:
                missing_codes.append(target["code"])

        if not allow_online_quote_fallback or not missing_codes:
            return

        try:
            quotes_online = await get_quotes_service().get_quotes(missing_codes)
            for target in targets:
                if target["item"].get("current_price") is not None:
                    continue
                quote = (quotes_online or {}).get(target["code"])
                if quote:
                    self._apply_quote(target["item"], quote)
        except Exception:
            pass

    async def get_user_favorites_foreign(
        self,
        db,
        targets: List[Dict[str, Any]],
        allow_online_quote_fallback: bool = True,
    ) -> None:
        """处理港股和美股自选项的行情富集。"""
        hk_codes = [target["code"] for target in targets if target["market"] == "HK" and target["code"]]
        us_codes = [target["code"] for target in targets if target["market"] == "US" and target["code"]]

        hk_quotes = await self._load_quotes_from_mongo(db, "market_quotes_hk", hk_codes, "HK")
        us_quotes = await self._load_quotes_from_mongo(db, "market_quotes_us", us_codes, "US")

        missing_targets: List[Dict[str, Any]] = []
        for target in targets:
            target["item"]["board"] = target["item"].get("board", "-")
            target["item"]["exchange"] = target["item"].get("exchange", "-")

            if target["market"] == "HK":
                quote = hk_quotes.get(target["code"])
            else:
                quote = us_quotes.get(target["code"])

            if quote:
                self._apply_quote(target["item"], quote)
            else:
                missing_targets.append(target)

        if not allow_online_quote_fallback or not missing_targets:
            return

        try:
            from app.services.foreign_stock_service import ForeignStockService

            foreign_service = ForeignStockService(db=db)
            for target in missing_targets:
                quote = await foreign_service.get_quote(
                    target["market"],
                    target["code"],
                    force_refresh=False,
                )
                if quote:
                    self._apply_quote(target["item"], quote)
        except Exception:
            pass

    async def get_user_favorites(
        self,
        user_id: str,
        allow_online_quote_fallback: bool = True,
    ) -> List[Dict[str, Any]]:
        """获取用户自选股列表，并批量拉取实时行情进行富集（兼容字符串ID与ObjectId）。"""
        db = await self._get_db()

        favorites: List[Dict[str, Any]] = []
        if self._is_valid_object_id(user_id):
            # 先尝试使用 ObjectId 查询
            user = await db.users.find_one({"_id": ObjectId(user_id)})
            # 如果 ObjectId 查询失败，尝试使用字符串查询
            if user is None:
                user = await db.users.find_one({"_id": user_id})
            favorites = (user or {}).get("favorite_stocks", [])
        else:
            doc = await db.user_favorites.find_one({"user_id": user_id})
            favorites = (doc or {}).get("favorites", [])

        # 先格式化基础字段
        items = [self._format_favorite(fav) for fav in favorites]
        if not items:
            return items

        targets = [target for target in self._build_targets(items) if target["code"]]
        if not targets:
            return items

        cn_targets = [target for target in targets if target["market"] == "CN"]
        foreign_targets = [target for target in targets if target["market"] in {"HK", "US"}]

        if cn_targets:
            await self.get_user_favorites_cn(
                db,
                cn_targets,
                allow_online_quote_fallback=allow_online_quote_fallback,
            )
        if foreign_targets:
            await self.get_user_favorites_foreign(
                db,
                foreign_targets,
                allow_online_quote_fallback=allow_online_quote_fallback,
            )

        await self._enrich_latest_report_actions(db, user_id, targets)

        return items

    async def add_favorite(
        self,
        user_id: str,
        stock_code: str,
        stock_name: str,
        market: str = "A股",
        tags: List[str] = None,
        notes: str = "",
        alert_price_high: Optional[float] = None,
        alert_price_low: Optional[float] = None
    ) -> bool:
        """添加股票到自选股（兼容字符串ID与ObjectId）"""
        import logging
        logger = logging.getLogger("webapi")

        try:
            logger.info(f"🔧 [add_favorite] 开始添加自选股: user_id={user_id}, stock_code={stock_code}")

            db = await self._get_db()
            logger.info("🔧 [add_favorite] 数据库连接获取成功")

            favorite_stock = {
                "stock_code": stock_code,
                "stock_name": stock_name,
                "market": market,
                "added_at": datetime.utcnow(),
                "tags": tags or [],
                "notes": notes,
                "alert_price_high": alert_price_high,
                "alert_price_low": alert_price_low
            }

            logger.info(f"🔧 [add_favorite] 自选股数据构建完成: {favorite_stock}")

            is_oid = self._is_valid_object_id(user_id)
            logger.info(f"🔧 [add_favorite] 用户ID类型检查: is_valid_object_id={is_oid}")

            if is_oid:
                logger.info("🔧 [add_favorite] 使用 ObjectId 方式添加到 users 集合")

                # 先尝试使用 ObjectId 查询
                result = await db.users.update_one(
                    {"_id": ObjectId(user_id)},
                    {
                        "$push": {"favorite_stocks": favorite_stock},
                        "$setOnInsert": {"favorite_stocks": []}
                    }
                )
                logger.info(f"🔧 [add_favorite] ObjectId查询结果: matched_count={result.matched_count}, modified_count={result.modified_count}")

                # 如果 ObjectId 查询失败，尝试使用字符串查询
                if result.matched_count == 0:
                    logger.info("🔧 [add_favorite] ObjectId查询失败，尝试使用字符串ID查询")
                    result = await db.users.update_one(
                        {"_id": user_id},
                        {
                            "$push": {"favorite_stocks": favorite_stock}
                        }
                    )
                    logger.info(f"🔧 [add_favorite] 字符串ID查询结果: matched_count={result.matched_count}, modified_count={result.modified_count}")

                success = result.matched_count > 0
                logger.info(f"🔧 [add_favorite] 返回结果: {success}")
                return success
            else:
                logger.info("🔧 [add_favorite] 使用字符串ID方式添加到 user_favorites 集合")
                result = await db.user_favorites.update_one(
                    {"user_id": user_id},
                    {
                        "$setOnInsert": {"user_id": user_id, "created_at": datetime.utcnow()},
                        "$push": {"favorites": favorite_stock},
                        "$set": {"updated_at": datetime.utcnow()}
                    },
                    upsert=True
                )
                logger.info(f"🔧 [add_favorite] 更新结果: matched_count={result.matched_count}, modified_count={result.modified_count}, upserted_id={result.upserted_id}")
                logger.info("🔧 [add_favorite] 返回结果: True")
                return True
        except Exception as e:
            logger.error(f"❌ [add_favorite] 添加自选股异常: {type(e).__name__}: {str(e)}", exc_info=True)
            raise

    async def remove_favorite(self, user_id: str, stock_code: str) -> bool:
        """从自选股中移除股票（兼容字符串ID与ObjectId）"""
        db = await self._get_db()

        if self._is_valid_object_id(user_id):
            # 先尝试使用 ObjectId 查询
            result = await db.users.update_one(
                {"_id": ObjectId(user_id)},
                {"$pull": {"favorite_stocks": {"stock_code": stock_code}}}
            )
            # 如果 ObjectId 查询失败，尝试使用字符串查询
            if result.matched_count == 0:
                result = await db.users.update_one(
                    {"_id": user_id},
                    {"$pull": {"favorite_stocks": {"stock_code": stock_code}}}
                )
            return result.modified_count > 0
        else:
            result = await db.user_favorites.update_one(
                {"user_id": user_id},
                {
                    "$pull": {"favorites": {"stock_code": stock_code}},
                    "$set": {"updated_at": datetime.utcnow()}
                }
            )
            return result.modified_count > 0

    async def update_favorite(
        self,
        user_id: str,
        stock_code: str,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        alert_price_high: Optional[float] = None,
        alert_price_low: Optional[float] = None
    ) -> bool:
        """更新自选股信息（兼容字符串ID与ObjectId）"""
        db = await self._get_db()

        # 统一构建更新字段（根据不同集合的字段路径设置前缀）
        is_oid = self._is_valid_object_id(user_id)
        prefix = "favorite_stocks.$." if is_oid else "favorites.$."
        update_fields: Dict[str, Any] = {}
        if tags is not None:
            update_fields[prefix + "tags"] = tags
        if notes is not None:
            update_fields[prefix + "notes"] = notes
        if alert_price_high is not None:
            update_fields[prefix + "alert_price_high"] = alert_price_high
        if alert_price_low is not None:
            update_fields[prefix + "alert_price_low"] = alert_price_low

        if not update_fields:
            return True

        if is_oid:
            result = await db.users.update_one(
                {
                    "_id": ObjectId(user_id),
                    "favorite_stocks.stock_code": stock_code
                },
                {"$set": update_fields}
            )
            return result.modified_count > 0
        else:
            result = await db.user_favorites.update_one(
                {
                    "user_id": user_id,
                    "favorites.stock_code": stock_code
                },
                {
                    "$set": {
                        **update_fields,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            return result.modified_count > 0

    async def is_favorite(self, user_id: str, stock_code: str) -> bool:
        """检查股票是否在自选股中（兼容字符串ID与ObjectId）"""
        import logging
        logger = logging.getLogger("webapi")

        try:
            logger.info(f"🔧 [is_favorite] 检查自选股: user_id={user_id}, stock_code={stock_code}")

            db = await self._get_db()

            is_oid = self._is_valid_object_id(user_id)
            logger.info(f"🔧 [is_favorite] 用户ID类型: is_valid_object_id={is_oid}")

            if is_oid:
                # 先尝试使用 ObjectId 查询
                user = await db.users.find_one(
                    {
                        "_id": ObjectId(user_id),
                        "favorite_stocks.stock_code": stock_code
                    }
                )

                # 如果 ObjectId 查询失败，尝试使用字符串查询
                if user is None:
                    logger.info("🔧 [is_favorite] ObjectId查询未找到，尝试使用字符串ID查询")
                    user = await db.users.find_one(
                        {
                            "_id": user_id,
                            "favorite_stocks.stock_code": stock_code
                        }
                    )

                result = user is not None
                logger.info(f"🔧 [is_favorite] 查询结果: {result}")
                return result
            else:
                doc = await db.user_favorites.find_one(
                    {
                        "user_id": user_id,
                        "favorites.stock_code": stock_code
                    }
                )
                result = doc is not None
                logger.info(f"🔧 [is_favorite] 字符串ID查询结果: {result}")
                return result
        except Exception as e:
            logger.error(f"❌ [is_favorite] 检查自选股异常: {type(e).__name__}: {str(e)}", exc_info=True)
            raise

    async def get_user_tags(self, user_id: str) -> List[str]:
        """获取用户使用的所有标签（兼容字符串ID与ObjectId）"""
        db = await self._get_db()

        if self._is_valid_object_id(user_id):
            pipeline = [
                {"$match": {"_id": ObjectId(user_id)}},
                {"$unwind": "$favorite_stocks"},
                {"$unwind": "$favorite_stocks.tags"},
                {"$group": {"_id": "$favorite_stocks.tags"}},
                {"$sort": {"_id": 1}}
            ]
            result = await db.users.aggregate(pipeline).to_list(None)
        else:
            pipeline = [
                {"$match": {"user_id": user_id}},
                {"$unwind": "$favorites"},
                {"$unwind": "$favorites.tags"},
                {"$group": {"_id": "$favorites.tags"}},
                {"$sort": {"_id": 1}}
            ]
            result = await db.user_favorites.aggregate(pipeline).to_list(None)

        return [item["_id"] for item in result if item.get("_id")]

    def _get_mock_price(self, stock_code: str) -> float:
        """获取模拟股价"""
        # 基于股票代码生成模拟价格
        base_price = hash(stock_code) % 100 + 10
        return round(base_price + (hash(stock_code) % 1000) / 100, 2)
    
    def _get_mock_change(self, stock_code: str) -> float:
        """获取模拟涨跌幅"""
        # 基于股票代码生成模拟涨跌幅
        change = (hash(stock_code) % 2000 - 1000) / 100
        return round(change, 2)
    
    def _get_mock_volume(self, stock_code: str) -> int:
        """获取模拟成交量"""
        # 基于股票代码生成模拟成交量
        return (hash(stock_code) % 10000 + 1000) * 100


# 创建全局实例
favorites_service = FavoritesService()
