export interface CodexSessionEntry {
  roleName: string
  sessionId: string
}

const MODULE_ROLE_MAPPING: Record<string, string[]> = {
  bull_researcher: ['Bull Researcher'],
  bear_researcher: ['Bear Researcher'],
  research_team_decision: ['Bull Researcher', 'Bear Researcher'],
  investment_debate_state: ['Bull Researcher', 'Bear Researcher'],
  investment_plan: ['Bull Researcher', 'Bear Researcher'],
  risky_analyst: ['Risky Analyst'],
  safe_analyst: ['Safe Analyst'],
  neutral_analyst: ['Neutral Analyst'],
  risk_management_decision: ['Risky Analyst', 'Safe Analyst', 'Neutral Analyst'],
  risk_debate_state: ['Risky Analyst', 'Safe Analyst', 'Neutral Analyst'],
  final_trade_decision: ['Risky Analyst', 'Safe Analyst', 'Neutral Analyst']
}

const ROLE_DISPLAY_NAMES: Record<string, string> = {
  'Bull Researcher': '多头研究员',
  'Bear Researcher': '空头研究员',
  'Risky Analyst': '激进分析师',
  'Safe Analyst': '保守分析师',
  'Neutral Analyst': '中性分析师'
}

export const getRelatedCodexSessions = (
  moduleName: string,
  codexRoleSessions?: Record<string, string> | null
): CodexSessionEntry[] => {
  if (!codexRoleSessions || typeof codexRoleSessions !== 'object') {
    return []
  }

  const relatedRoles = MODULE_ROLE_MAPPING[moduleName] || []

  return relatedRoles
    .map((roleName) => {
      const sessionId = codexRoleSessions[roleName]
      if (!sessionId) return null

      return {
        roleName: ROLE_DISPLAY_NAMES[roleName] || roleName,
        sessionId
      }
    })
    .filter((entry): entry is CodexSessionEntry => entry !== null)
}
