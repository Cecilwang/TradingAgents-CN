#!/usr/bin/env python3
"""
将分析默认模型迁移到 gpt-5.4
"""

import asyncio
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient

from app.core.config import settings
from app.core.unified_config import unified_config

TARGET_MODEL = "gpt-5.4"
LEGACY_MODELS = {
    "qwen-flash",
    "qwen-turbo",
    "qwen-plus",
    "qwen-max",
    "qwen3-max",
}


def should_migrate(model_name: Any) -> bool:
    """空值或历史默认模型都迁移到目标模型。"""
    model = str(model_name or "").strip()
    return not model or model in LEGACY_MODELS


def migrate_settings_dict(system_settings: dict[str, Any]) -> tuple[dict[str, Any], dict[str, tuple[Any, Any]]]:
    """统一迁移系统设置中的分析模型字段。"""
    updated_settings = dict(system_settings or {})
    changed_fields: dict[str, tuple[Any, Any]] = {}

    for field_name in (
        "quick_analysis_model",
        "deep_analysis_model",
        "quick_think_llm",
        "deep_think_llm",
        "default_model",
    ):
        current_value = updated_settings.get(field_name)
        if should_migrate(current_value):
            updated_settings[field_name] = TARGET_MODEL
            if current_value != TARGET_MODEL:
                changed_fields[field_name] = (current_value, TARGET_MODEL)

    return updated_settings, changed_fields


async def migrate_database_settings() -> int:
    """迁移 MongoDB system_configs 集合。"""
    client = AsyncIOMotorClient(settings.MONGO_URI)
    db = client[settings.MONGO_DB]
    updated_docs = 0

    try:
        cursor = db["system_configs"].find({})
        async for system_config in cursor:
            system_settings = system_config.get("system_settings", {}) or {}
            migrated_settings, changed_fields = migrate_settings_dict(system_settings)

            if not changed_fields:
                print(f"⏭️ 跳过文档 {system_config.get('_id')}，分析模型已是 {TARGET_MODEL}")
                continue

            print(f"\n📝 更新 system_configs 文档 {system_config.get('_id')}:")
            for field_name, (before, after) in changed_fields.items():
                print(f"  - {field_name}: {before} -> {after}")

            result = await db["system_configs"].update_one(
                {"_id": system_config["_id"]},
                {
                    "$set": {
                        "system_settings": migrated_settings,
                        "version": system_config.get("version", 0) + 1,
                    }
                }
            )

            if result.modified_count > 0:
                updated_docs += 1

        return updated_docs
    finally:
        client.close()


def migrate_json_settings() -> bool:
    """迁移 config/settings.json。"""
    current_settings = unified_config.get_system_settings()
    migrated_settings, changed_fields = migrate_settings_dict(current_settings)

    if not changed_fields:
        print(f"\n⏭️ 跳过 config/settings.json，分析模型已是 {TARGET_MODEL}")
        return False

    print(f"\n📝 更新 config/settings.json:")
    for field_name, (before, after) in changed_fields.items():
        print(f"  - {field_name}: {before} -> {after}")

    if not unified_config.save_system_settings(migrated_settings):
        raise RuntimeError("保存 config/settings.json 失败")

    return True

async def main():
    """主函数"""
    print("=" * 60)
    print("📊 迁移分析模型默认值到 gpt-5.4")
    print("=" * 60)

    try:
        updated_docs = await migrate_database_settings()
        updated_json = migrate_json_settings()

        print(f"\n✅ 迁移完成")
        print(f"  - MongoDB 更新文档数: {updated_docs}")
        print(f"  - config/settings.json 已更新: {'是' if updated_json else '否'}")
        print(f"  - 目标模型: {TARGET_MODEL}")
        print("\n⚠️ 如果 Web API 进程已在运行，config_provider 可能会缓存旧值最多 60 秒；重启服务可立即生效。")
        print("\n" + "=" * 60)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
