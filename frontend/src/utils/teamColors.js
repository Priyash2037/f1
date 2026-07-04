/** Team colors from OpenF1 API (team_colour field) as hex strings. */
export const TEAM_COLORS = {
  'McLaren':           '#F47600',
  'Red Bull Racing':   '#4781D7',
  'Ferrari':           '#ED1131',
  'Mercedes':          '#00D7B6',
  'Aston Martin':      '#229971',
  'Alpine':            '#00A1E8',
  'Williams':          '#1868DB',
  'Racing Bulls':      '#6C98FF',
  'Haas F1 Team':      '#9C9FA2',
  'Audi':              '#F50537',
  'Cadillac':          '#909090',
  'Kick Sauber':       '#00E701',
}

/**
 * Get team color from team_colour hex OR team_name fallback.
 * OpenF1 returns team_colour as a hex string (no leading #).
 */
export function getTeamColor(teamColour, teamName) {
  if (teamColour && teamColour.length === 6) return `#${teamColour}`
  return TEAM_COLORS[teamName] ?? '#FFFFFF'
}

/** Lighten a hex color by adding alpha transparency */
export function withAlpha(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  return `rgba(${r},${g},${b},${alpha})`
}

/** Compound → display label */
export const COMPOUND_LABELS = {
  SOFT:         'S',
  MEDIUM:       'M',
  HARD:         'H',
  INTERMEDIATE: 'I',
  WET:          'W',
  UNKNOWN:      '?',
}

/** Format seconds as M:SS.mmm */
export function formatLapTime(seconds) {
  if (seconds == null) return '—'
  const m  = Math.floor(seconds / 60)
  const s  = Math.floor(seconds % 60)
  const ms = Math.round((seconds % 1) * 1000)
  return `${m}:${String(s).padStart(2,'0')}.${String(ms).padStart(3,'0')}`
}

/** Format gap (float seconds) for display */
export function formatGap(gap) {
  if (gap == null) return '—'
  if (typeof gap === 'string') return gap
  return `+${gap.toFixed(3)}s`
}
