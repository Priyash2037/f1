import React, { useState } from 'react'
import './index.css'
import { useWebSocket } from './hooks/useWebSocket'
import Header from './components/Header'
import SessionInfo from './components/SessionInfo'
import Leaderboard from './components/Leaderboard'
import ProbabilityChart from './components/ProbabilityChart'
import EventFeed from './components/EventFeed'
import InsightsPanel from './components/InsightsPanel'
import SidebarHub from './components/SidebarHub'

export default function App() {
  const { liveState, status } = useWebSocket()
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)

  const session  = liveState?.session  ?? null
  const drivers  = liveState?.drivers  ?? []
  const events   = liveState?.events   ?? []
  const weather  = liveState?.weather  ?? null

  return (
    <div className="app-container">
      <Header session={session} status={status}>
        <div className="header-right">
          <button className="hamburger-btn" onClick={() => setIsSidebarOpen(true)}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
          </button>
        </div>
      </Header>

      {/* Sidebar Hub */}
      <SidebarHub isOpen={isSidebarOpen} onClose={() => setIsSidebarOpen(false)} />

      <main className="main-content">
        {/* Top bar: session + weather */}
        <SessionInfo session={session} weather={weather} />

        {/* Left column: leaderboard */}
        <Leaderboard drivers={drivers} session={session} />

        {/* Right column: chart + event feed */}
        <div className="right-column">
          <InsightsPanel drivers={drivers} />
          <ProbabilityChart drivers={drivers} session={session} />
          <EventFeed events={events} />
        </div>
      </main>
    </div>
  )
}
