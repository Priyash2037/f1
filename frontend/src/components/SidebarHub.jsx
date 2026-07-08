import React, { useState, useEffect } from 'react'

export default function SidebarHub({ isOpen, onClose }) {
  const [activeTab, setActiveTab] = useState('drivers')
  const [data, setData] = useState({ drivers: [], constructors: [], races: [] })
  const [loading, setLoading] = useState(false)

  // Fetch data when a tab is opened (or just fetch all once on mount)
  useEffect(() => {
    if (isOpen && data.drivers.length === 0) {
      setLoading(true)
      Promise.all([
        fetch('/api/standings/drivers').then(r => r.json()),
        fetch('/api/standings/constructors').then(r => r.json()),
        fetch('/api/results/races').then(r => r.json())
      ]).then(([d, c, r]) => {
        setData({
          drivers: d.standings || [],
          constructors: c.standings || [],
          races: (r.races || []).reverse() // Show most recent first
        })
        setLoading(false)
      }).catch(err => {
        console.error("Failed to load hub data:", err)
        setLoading(false)
      })
    }
  }, [isOpen, data.drivers.length])

  return (
    <>
      {/* Overlay */}
      <div className={`sidebar-overlay ${isOpen ? 'open' : ''}`} onClick={onClose}></div>
      
      {/* Drawer */}
      <div className={`sidebar-drawer ${isOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <h2>Championship Hub</h2>
          <button className="close-btn" onClick={onClose}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          </button>
        </div>

        <div className="sidebar-tabs">
          <button className={`tab-btn ${activeTab === 'drivers' ? 'active' : ''}`} onClick={() => setActiveTab('drivers')}>Drivers</button>
          <button className={`tab-btn ${activeTab === 'constructors' ? 'active' : ''}`} onClick={() => setActiveTab('constructors')}>Constructors</button>
          <button className={`tab-btn ${activeTab === 'races' ? 'active' : ''}`} onClick={() => setActiveTab('races')}>Results</button>
        </div>

        <div className="sidebar-content">
          {loading ? (
            <div className="hub-loading"><div className="spinner"></div></div>
          ) : (
            <>
              {activeTab === 'drivers' && (
                <table className="hub-table">
                  <thead>
                    <tr>
                      <th>Pos</th>
                      <th>Driver</th>
                      <th>Pts</th>
                      <th>Wins</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.drivers.map(d => (
                      <tr key={d.Driver.driverId}>
                        <td className="center">{d.position}</td>
                        <td className="bold">{d.Driver.givenName} {d.Driver.familyName}</td>
                        <td className="center bold">{d.points}</td>
                        <td className="center">{d.wins}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}

              {activeTab === 'constructors' && (
                <table className="hub-table">
                  <thead>
                    <tr>
                      <th>Pos</th>
                      <th>Constructor</th>
                      <th>Pts</th>
                      <th>Wins</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.constructors.map(c => (
                      <tr key={c.Constructor.constructorId}>
                        <td className="center">{c.position}</td>
                        <td className="bold">{c.Constructor.name}</td>
                        <td className="center bold">{c.points}</td>
                        <td className="center">{c.wins}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}

              {activeTab === 'races' && (
                <div className="races-list">
                  {data.races.map(r => (
                    <div className="race-card" key={r.round}>
                      <div className="race-header">
                        <span className="round-badge">R{r.round}</span>
                        <span className="race-name">{r.raceName}</span>
                      </div>
                      <div className="race-podium">
                        {r.Results.slice(0, 3).map((res, i) => (
                          <div className="podium-row" key={res.position}>
                            <span className={`pos p${res.position}`}>P{res.position}</span>
                            <span className="name">{res.Driver.familyName}</span>
                            <span className="team">{res.Constructor.name}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </>
  )
}
