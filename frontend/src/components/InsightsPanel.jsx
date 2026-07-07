import React from 'react'

export default function InsightsPanel({ drivers }) {
  if (!drivers || drivers.length === 0) return null

  // 1. Podium Contenders (highest podium_probability, top 3)
  const podiumContenders = [...drivers]
    .sort((a, b) => b.podium_probability - a.podium_probability)
    .slice(0, 3)

  // 2. DNF Risk (highest dnf_probability, top 3)
  const dnfRisks = [...drivers]
    .sort((a, b) => b.dnf_probability - a.dnf_probability)
    .slice(0, 3)

  // 3. Pit Strategy (lowest tyre_life_remaining_percent, top 3)
  // Only include drivers actually on track (not in pit, and have tyres)
  const pitWatch = [...drivers]
    .filter(d => d.tyre_life_remaining_percent < 30 && d.compound !== 'UNKNOWN')
    .sort((a, b) => a.tyre_life_remaining_percent - b.tyre_life_remaining_percent)
    .slice(0, 3)

  return (
    <div className="insights-panel">
      {/* Podium Contenders */}
      <div className="card insight-card">
        <div className="card-header">
          <span className="card-title" style={{ color: '#F4B41A' }}>
            <svg className="card-title-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M5 21V14L12 7L19 14V21"/>
            </svg>
            Podium Contenders
          </span>
        </div>
        <div className="insight-list">
          {podiumContenders.map((d, i) => (
            <div className="insight-row" key={d.driver_number}>
              <div className="insight-driver">
                <span className="pos-badge">{i + 1}</span>
                <span className="name" style={{ borderLeft: `3px solid #${d.team_colour}` }}>{d.name_acronym}</span>
              </div>
              <div className="insight-value">{(d.podium_probability * 100).toFixed(1)}%</div>
            </div>
          ))}
        </div>
      </div>

      {/* DNF Risk */}
      <div className="card insight-card">
        <div className="card-header">
          <span className="card-title" style={{ color: '#E10600' }}>
            <svg className="card-title-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
            </svg>
            DNF Risk Alert
          </span>
        </div>
        <div className="insight-list">
          {dnfRisks.map((d) => (
            <div className="insight-row" key={d.driver_number}>
              <div className="insight-driver">
                <span className="name" style={{ borderLeft: `3px solid #${d.team_colour}` }}>{d.name_acronym}</span>
              </div>
              <div className="insight-value risk-high">{(d.dnf_probability * 100).toFixed(1)}%</div>
            </div>
          ))}
        </div>
      </div>

      {/* Pit Strategy Watch */}
      <div className="card insight-card">
        <div className="card-header">
          <span className="card-title" style={{ color: '#00D2BE' }}>
            <svg className="card-title-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
            </svg>
            Pit Window Watch
          </span>
        </div>
        <div className="insight-list">
          {pitWatch.length === 0 ? (
            <div className="insight-empty">No drivers in critical window</div>
          ) : (
            pitWatch.map((d) => (
              <div className="insight-row" key={d.driver_number}>
                <div className="insight-driver">
                  <span className="name" style={{ borderLeft: `3px solid #${d.team_colour}` }}>{d.name_acronym}</span>
                  <span className={`tyre-badge badge-${d.compound}`}>{d.compound[0]}</span>
                </div>
                <div className="insight-value tyre-low">{d.tyre_life_remaining_percent.toFixed(0)}% left</div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}
