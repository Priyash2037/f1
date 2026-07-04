/**
 * useWebSocket — custom React hook for the F1 Oracle live state stream.
 *
 * Features:
 *  • Auto-connects to /ws/live on mount
 *  • Parses incoming JSON state payloads
 *  • Auto-reconnects with exponential backoff (1s → 2s → 4s … → 30s max)
 *  • Heartbeat ping every 20s to keep the connection alive
 *  • Exposes connection status: 'connecting' | 'connected' | 'reconnecting' | 'disconnected'
 */

import { useCallback, useEffect, useRef, useState } from 'react'

const WS_URL = 'ws://localhost:8000/ws/live'
const PING_INTERVAL = 20_000
const MAX_BACKOFF   = 30_000

export function useWebSocket() {
  const [liveState, setLiveState]   = useState(null)
  const [status, setStatus]         = useState('connecting')

  const wsRef         = useRef(null)
  const backoffRef    = useRef(1_000)
  const pingTimerRef  = useRef(null)
  const reconnectRef  = useRef(null)
  const mountedRef    = useRef(true)

  const clearTimers = () => {
    if (pingTimerRef.current)  clearInterval(pingTimerRef.current)
    if (reconnectRef.current)  clearTimeout(reconnectRef.current)
  }

  const connect = useCallback(() => {
    if (!mountedRef.current) return

    setStatus(prev => prev === 'connected' ? 'reconnecting' : 'connecting')
    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      if (!mountedRef.current) { ws.close(); return }
      setStatus('connected')
      backoffRef.current = 1_000   // reset backoff on success

      // Heartbeat
      pingTimerRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) ws.send('ping')
      }, PING_INTERVAL)
    }

    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data)
        if (data?.type === 'pong') return   // ignore heartbeat reply
        setLiveState(data)
      } catch (e) {
        // ignore malformed frames
      }
    }

    ws.onclose = (evt) => {
      clearTimers()
      if (!mountedRef.current) return
      setStatus('reconnecting')
      // Exponential backoff
      const delay = Math.min(backoffRef.current, MAX_BACKOFF)
      backoffRef.current = Math.min(backoffRef.current * 2, MAX_BACKOFF)
      reconnectRef.current = setTimeout(connect, delay)
    }

    ws.onerror = () => {
      ws.close()
    }
  }, [])

  useEffect(() => {
    mountedRef.current = true
    connect()
    return () => {
      mountedRef.current = false
      clearTimers()
      wsRef.current?.close()
    }
  }, [connect])

  return { liveState, status }
}
