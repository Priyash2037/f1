import { getTeamColor, withAlpha } from '../utils/teamColors'

function ProbBar({ label, value, max, color, alpha = 0.85 }) {
  const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0
  const bg  = `${color}${Math.round(alpha * 255).toString(16).padStart(2, '0')}`
  const displayPct = (value * 100).toFixed(1)
  const showInner  = pct > 8   // Only show text if bar is wide enough

  return (
    <div className="prob-bar-row">
      <span className="prob-bar-label">{label}</span>
      <div className="prob-bar-bg">
        <div
          className="prob-bar-inner"
          style={{ width: `${pct}%`, background: withAlpha(color, alpha) }}
        >
          {showInner && (
            <span className="prob-bar-value">{displayPct}%</span>
          )}
        </div>
      </div>
    </div>
  )
}

export default function ProbabilityChart({ drivers, session }) {
  if (!drivers || drivers.length === 0) {
    return (
      <div className="card prob-chart-card">
        <div className="card-header">
          <span className="card-title">
            <svg className="card-title-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M18 20V10M12 20V4M6 20v-6"/>
            </svg>
            Win Probability
          </span>
        </div>
        <div className="empty-state">
          <div className="empty-state-icon">📊</div>
          <p>Probabilities will appear once live data loads.</p>
        </div>
      </div>
    )
  }

  const isQual = session?.session_type?.toLowerCase().includes('qual')
  const title  = isQual ? 'Pole Probability' : 'Win Probability'

  // Sort by win probability, show top 10
  const top10 = [...drivers]
    .sort((a, b) => b.win_probability - a.win_probability)
    .slice(0, 10)

  const maxWin    = Math.max(...top10.map(d => d.win_probability), 0.001)
  const maxPodium = Math.max(...top10.map(d => d.podium_probability), 0.001)

  return (
    <div className="card prob-chart-card">
      <div className="card-header">
        <span className="card-title">
          <svg className="card-title-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M18 20V10M12 20V4M6 20v-6"/>
          </svg>
          {title}
        </span>
        <span className="card-badge">Monte Carlo + ML</span>
      </div>

      {/* Legend */}
      <div style={{
        display: 'flex', gap: 16, padding: '8px 20px',
        borderBottom: '1px solid var(--border-dim)',
        fontSize: 11, color: 'var(--text-secondary)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <div style={{ width: 10, height: 10, borderRadius: 2, background: '#e10600', opacity: 0.85 }} />
          <span>Win</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <div style={{ width: 10, height: 10, borderRadius: 2, background: '#e10600', opacity: 0.35 }} />
          <span>Podium</span>
        </div>
      </div>

      <div className="prob-chart-body">
        {top10.map((driver) => {
          const color   = getTeamColor(driver.team_colour, driver.team_name)
          const winPct  = (driver.win_probability * 100).toFixed(1)

          return (
            <div key={driver.driver_number} className="prob-row">
              {/* Driver label */}
              <span className="prob-driver-label" style={{ color }}>
                {driver.name_acronym}
              </span>

              {/* Stacked bars */}
              <div className="prob-bars-stack">
                <ProbBar
                  label="W"
                  value={driver.win_probability}
                  max={maxWin}
                  color={color}
                  alpha={0.88}
                />
                <ProbBar
                  label="P"
                  value={driver.podium_probability}
                  max={maxPodium}
                  color={color}
                  alpha={0.35}
                />
              </div>

              {/* Win % label */}
              <span
                className="prob-pct-label"
                style={{ color: driver.win_probability > 0.1 ? color : 'var(--text-secondary)' }}
              >
                {winPct}%
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
