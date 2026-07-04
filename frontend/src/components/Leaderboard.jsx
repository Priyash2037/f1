import { getTeamColor, COMPOUND_LABELS, formatLapTime } from '../utils/teamColors'

const TYRE_LABELS = { S: 'SOFT', M: 'MEDIUM', H: 'HARD', I: 'INTER', W: 'WET', '?': 'UNKNOWN' }

function PosNumber({ pos }) {
  const cls = pos === 1 ? 'p1' : pos === 2 ? 'p2' : pos === 3 ? 'p3' : ''
  return <span className={`pos-number ${cls}`}>{pos}</span>
}

function TyreBadge({ compound, age }) {
  const abbr = COMPOUND_LABELS[compound] ?? '?'
  const cls  = `tyre-${compound ?? 'UNKNOWN'}`
  return (
    <span className={`tyre-badge ${cls}`}>
      {abbr} <span style={{ opacity: 0.7, fontSize: 10 }}>{age}L</span>
    </span>
  )
}

function WinBar({ prob, color }) {
  const pct = (prob * 100).toFixed(1)
  const w   = Math.max(prob * 100, 0)
  return (
    <div className="prob-bar-inline">
      <div className="prob-bar-track">
        <div
          className="prob-bar-fill"
          style={{ width: `${w}%`, background: color }}
        />
      </div>
      <span className="prob-pct" style={{ color }}>
        {prob > 0.001 ? `${pct}%` : '<0.1%'}
      </span>
    </div>
  )
}

export default function Leaderboard({ drivers, session }) {
  const isQual = session?.session_type?.toLowerCase().includes('qual')
                 || session?.session_type?.toLowerCase().includes('practice')
  if (!drivers || drivers.length === 0) {
    return (
      <div className="card leaderboard-card">
        <div className="card-header">
          <span className="card-title">
            <svg className="card-title-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 12h18M3 6h18M3 18h18"/>
            </svg>
            Live Leaderboard
          </span>
        </div>
        <div className="empty-state">
          <div className="empty-state-icon">🏎️</div>
          <p>Waiting for live timing data from OpenF1…</p>
        </div>
      </div>
    )
  }

  // Sort by current position
  const sorted = [...drivers].sort((a, b) => a.position - b.position)

  // In qualifying the decisive stat is the best lap, not last lap
  const lapTimeLabel = isQual ? 'BEST LAP' : 'LAST LAP'
  const gapLabel     = isQual ? 'GAP'      : 'GAP'

  return (
    <div className="card leaderboard-card">
      <div className="card-header">
        <span className="card-title">
          <svg className="card-title-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M3 12h18M3 6h18M3 18h18"/>
          </svg>
          Live Leaderboard
        </span>
        <span className="card-badge">{sorted.length} drivers</span>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table className="leaderboard-table">
          <thead>
            <tr>
              <th className="pos-cell">POS</th>
              <th>DRIVER</th>
              <th>{gapLabel}</th>
              <th>{lapTimeLabel}</th>
              <th>TYRE</th>
              <th>PITS</th>
              <th>WIN %</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map(driver => {
              const color = getTeamColor(driver.team_colour, driver.team_name)
              const lapCls = driver.is_overall_best
                ? 'overall-best'
                : driver.is_personal_best
                ? 'personal-best'
                : 'normal'
              const isLeader = driver.position === 1

              return (
                <tr key={driver.driver_number}>
                  {/* Position */}
                  <td className="pos-cell">
                    <PosNumber pos={driver.position} />
                  </td>

                  {/* Driver */}
                  <td>
                    <div className="driver-cell">
                      <div className="team-stripe" style={{ background: color }} />
                      <div>
                        <div className="driver-acronym" style={{ color }}>
                          {driver.name_acronym}
                          <span className="driver-number-badge" style={{ marginLeft: 6 }}>
                            #{driver.driver_number}
                          </span>
                        </div>
                        <div className="driver-team">{driver.team_name}</div>
                      </div>
                    </div>
                  </td>

                  {/* Gap / best lap delta */}
                  <td>
                    <span className={`gap-cell ${isLeader ? 'leader' : ''}`}>
                      {isLeader
                        ? (isQual ? 'POLE' : 'LEADER')
                        : (driver.gap_to_leader ?? '—')}
                    </span>
                  </td>

                  {/* Lap time — best in qualifying, last in race */}
                  <td>
                    <span className={`lap-time ${lapCls}`}>
                      {isQual
                        ? formatLapTime(driver.best_lap_time)
                        : formatLapTime(driver.last_lap_time)}
                    </span>
                  </td>

                  {/* Tyre */}
                  <td>
                    <TyreBadge compound={driver.compound} age={driver.tyre_age} />
                  </td>

                  {/* Pits */}
                  <td style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-f1)', fontSize: 13 }}>
                    {driver.pit_stop_count}
                  </td>

                  {/* Win probability */}
                  <td>
                    <WinBar prob={driver.win_probability} color={color} />
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
