export interface CodexSessionEntry {
  roleName: string
  sessionIds: string[]
}

const MODULE_ROLE_MAPPING: Record<string, string> = {
  market_report: 'Market Analyst',
  sentiment_report: 'Social Media Analyst',
  news_report: 'News Analyst',
  fundamentals_report: 'Fundamentals Analyst',
  bull_researcher: 'Bull Researcher',
  bear_researcher: 'Bear Researcher',
  research_team_decision: 'Research Manager',
  investment_plan: 'Research Manager',
  trader_investment_plan: 'Trader',
  risky_analyst: 'Risky Analyst',
  safe_analyst: 'Safe Analyst',
  neutral_analyst: 'Neutral Analyst',
  risk_management_decision: 'Risk Judge',
  final_trade_decision: 'Risk Judge',
  investment_debate_state: 'Research Manager',
  risk_debate_state: 'Risk Judge'
}

const ROLE_DISPLAY_NAMES: Record<string, string> = {
  'Market Analyst': '市场技术分析师',
  'Social Media Analyst': '市场情绪分析师',
  'News Analyst': '新闻分析师',
  'Fundamentals Analyst': '基本面分析师',
  'Bull Researcher': '多头研究员',
  'Bear Researcher': '空头研究员',
  'Research Manager': '研究经理',
  'Trader': '交易员',
  'Risky Analyst': '激进分析师',
  'Safe Analyst': '保守分析师',
  'Neutral Analyst': '中性分析师',
  'Risk Judge': '投资组合经理'
}

export const getModuleCodexSession = (
  moduleName: string,
  codexRoleSessions?: Record<string, string[]> | null
): CodexSessionEntry | null => {
  if (!codexRoleSessions || typeof codexRoleSessions !== 'object') {
    return null
  }

  const roleName = MODULE_ROLE_MAPPING[moduleName]
  if (!roleName) {
    return null
  }

  const sessionIds = codexRoleSessions[roleName]
  if (!Array.isArray(sessionIds) || sessionIds.length === 0) {
    return null
  }

  const normalizedSessionIds = sessionIds.filter(
    (sessionId): sessionId is string => typeof sessionId === 'string' && sessionId.length > 0
  )
  if (normalizedSessionIds.length === 0) {
    return null
  }

  return {
    roleName: ROLE_DISPLAY_NAMES[roleName] || roleName,
    sessionIds: normalizedSessionIds
  }
}
