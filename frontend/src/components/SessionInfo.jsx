export default function SessionInfo({ session, weather }) {
  if (!session) {
    return (
      <div className="session-info-bar">
        <div className="empty-state" style={{ padding: '12px 0', flexDirection: 'row', gap: 12 }}>
          <span style={{ fontSize: 20 }}>🏁</span>
          <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
            No active session — waiting for OpenF1 live data…
          </span>
        </div>
      </div>
    )
  }

  const sessionTypeBg = {
    Race:              '#e10600',
    Sprint:            '#ff6b00',
    Qualifying:        '#9b4dca',
    'Sprint Qualifying': '#9b4dca',
    Practice:          '#1868db',
  }[session.session_type] ?? '#444'

  const trackTemp = weather?.track_temperature != null
    ? `${Math.round(weather.track_temperature)}°C`
    : '—'
  const airTemp = weather?.air_temperature != null
    ? `${Math.round(weather.air_temperature)}°C`
    : '—'
  const isWet = weather?.rainfall === 1

  return (
    <div className="session-info-bar">
      {/* Session type badge */}
      <span
        className="session-type-badge"
        style={{ background: sessionTypeBg }}
      >
        {session.session_type}
      </span>

      {/* Session name */}
      <span className="session-name">{session.country_name} Grand Prix</span>

      <div className="session-divider" />

      {/* Circuit */}
      <div className="session-meta-item">
        <span className="label">Circuit</span>
        <span className="value">{session.circuit_short_name}</span>
      </div>

      <div className="session-divider" />

      {/* Round info */}
      <div className="session-meta-item">
        <span className="label">Session</span>
        <span className="value">{session.session_name}</span>
      </div>

      <div className="session-divider" />

      <div className="session-meta-item">
        <span className="label">Season</span>
        <span className="value">{session.year}</span>
      </div>

      {/* Weather */}
      <div className="weather-badge">
        {isWet && <span title="Wet track" style={{ fontSize: 16 }}>🌧️</span>}
        <div className="weather-item">
          <span>🛣️</span>
          <span>{trackTemp}</span>
        </div>
        <div className="weather-item">
          <span>🌡️</span>
          <span>{airTemp}</span>
        </div>
      </div>
    </div>
  )
}
