const FLAG_CONFIG = {
  GREEN:      { dot: 'flag-GREEN',      label: null,      emoji: '🟢' },
  YELLOW:     { dot: 'flag-YELLOW',     label: 'YELLOW',  emoji: '🟡' },
  RED:        { dot: 'flag-RED',        label: 'RED',     emoji: '🔴' },
  SC:         { dot: 'flag-SC',         label: 'SC',      emoji: '🟡' },
  VSC:        { dot: 'flag-VSC',        label: 'VSC',     emoji: '🟡' },
  CHEQUERED:  { dot: 'flag-CHEQUERED',  label: null,      emoji: '🏁' },
}

function formatEventTime(dateStr) {
  if (!dateStr) return ''
  try {
    const d = new Date(dateStr)
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  } catch {
    return dateStr.slice(11, 19)
  }
}

function EventItem({ event }) {
  const cfg = FLAG_CONFIG[event.flag] ?? { dot: 'flag-default', label: null, emoji: '📋' }
  const hasLabel = cfg.label != null

  return (
    <div className="event-item">
      <div className={`event-flag-badge ${cfg.dot}`} />
      <div className="event-content">
        <div className="event-message">
          {cfg.emoji} {event.message}
        </div>
        <div className="event-time">
          {hasLabel && (
            <span className={`event-flag-label ${cfg.dot}`} style={{ marginRight: 6 }}>
              {cfg.label}
            </span>
          )}
          {formatEventTime(event.date)}
        </div>
      </div>
    </div>
  )
}

export default function EventFeed({ events }) {
  // Most recent first
  const sorted = [...(events ?? [])].reverse()

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title">
          <svg className="card-title-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
          </svg>
          Race Control
        </span>
        <span className="card-badge">{sorted.length} events</span>
      </div>

      <div className="event-feed-body">
        {sorted.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">📡</div>
            <p>No race control events yet.</p>
          </div>
        ) : (
          sorted.map((event, idx) => (
            <EventItem key={`${event.date}-${idx}`} event={event} />
          ))
        )}
      </div>
    </div>
  )
}
