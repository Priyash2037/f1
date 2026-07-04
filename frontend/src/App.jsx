import './index.css'
import { useWebSocket } from './hooks/useWebSocket'
import Header from './components/Header'
import SessionInfo from './components/SessionInfo'
import Leaderboard from './components/Leaderboard'
import ProbabilityChart from './components/ProbabilityChart'
import EventFeed from './components/EventFeed'

export default function App() {
  const { liveState, status } = useWebSocket()

  const session  = liveState?.session  ?? null
  const drivers  = liveState?.drivers  ?? []
  const events   = liveState?.events   ?? []
  const weather  = liveState?.weather  ?? null

  return (
    <div className="app-container">
      <Header session={session} status={status} />

      <main className="main-content">
        {/* Top bar: session + weather */}
        <SessionInfo session={session} weather={weather} />

        {/* Left column: leaderboard */}
        <Leaderboard drivers={drivers} session={session} />

        {/* Right column: chart + event feed */}
        <div className="right-column">
          <ProbabilityChart drivers={drivers} session={session} />
          <EventFeed events={events} />
        </div>
      </main>
    </div>
  )
}
