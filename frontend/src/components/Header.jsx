import { useEffect, useState } from 'react'

export default function Header({ session, status, children }) {
  const [time, setTime] = useState(new Date())

  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(t)
  }, [])

  const hh = String(time.getHours()).padStart(2, '0')
  const mm = String(time.getMinutes()).padStart(2, '0')
  const ss = String(time.getSeconds()).padStart(2, '0')

  const connLabel = {
    connected:    'LIVE',
    connecting:   'CONNECTING',
    reconnecting: 'RECONNECTING',
    disconnected: 'OFFLINE',
  }[status] ?? 'CONNECTING'

  return (
    <header className="header">
      {/* Logo */}
      <div className="header-logo">
        <div className="logo-icon">F1</div>
        <div className="logo-text">F1 <span>Oracle</span></div>
      </div>

      {/* Session badge */}
      <div className="header-center">
        {session ? (
          <div className="session-badge">
            <span className={`live-dot ${status !== 'connected' ? 'offline' : ''}`} />
            <span>{session.circuit_short_name} — {session.session_name}</span>
          </div>
        ) : (
          <div className="session-badge">
            <span className={`live-dot ${status !== 'connected' ? 'offline' : ''}`} />
            <span>Waiting for session…</span>
          </div>
        )}
      </div>

      {/* Right side */}
      <div className="header-right">
        <div className={`conn-indicator ${status}`}>
          <div className="conn-dot" />
          <span>{connLabel}</span>
        </div>
        <div className="live-clock">{hh}:{mm}:{ss}</div>
        {children}
      </div>
    </header>
  )
}
