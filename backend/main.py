"""
F1 Oracle — FastAPI Application
==================================
Main entry point. Exposes:
  • GET  /api/state          — full state snapshot (for initial page load)
  • GET  /api/session        — current session info
  • GET  /api/probabilities  — current probability results
  • GET  /health             — health check
  • WS   /ws/live            — real-time state stream

Run with:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import poller
import state
import stats
from models import LiveStatePayload

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("f1oracle")


# ─────────────────────────────────────────────────────────────────────────────
# Application lifespan — starts background poller
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("╔══════════════════════════════════╗")
    log.info("║   F1 Oracle Backend Starting…    ║")
    log.info("╚══════════════════════════════════╝")
    poller_task = asyncio.create_task(poller.run_poller(), name="openf1-poller")
    yield
    log.info("Shutting down poller…")
    poller_task.cancel()
    try:
        await poller_task
    except asyncio.CancelledError:
        pass
    log.info("F1 Oracle shut down cleanly.")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI application
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="F1 Oracle API",
    description="Real-time F1 race probability engine powered by OpenF1 + Monte Carlo",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the Vite dev server (localhost:5173) to call our API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# REST endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
async def health() -> dict:
    return {
        "status": "ok",
        "session_key": state.session.session_key if state.session else None,
        "drivers": len(state.drivers),
        "ws_clients": len(poller.ws_connections),
        "last_updated": state.last_updated,
    }


@app.get("/api/session", tags=["data"])
async def get_session():
    """Returns current session metadata."""
    return state.session


@app.get("/api/state", tags=["data"])
async def get_state():
    """
    Returns a full state snapshot — used by the frontend on initial load
    before the WebSocket stream takes over.
    """
    return LiveStatePayload(
        session         = state.session,
        drivers         = state.build_live_state(),
        events          = list(state.events),
        weather         = state.weather,
        probabilities   = list(state.probabilities.values()),
        last_updated    = state.last_updated,
        connection_count = len(poller.ws_connections),
        is_live         = state.session is not None,
    )


@app.get("/api/probabilities", tags=["data"])
async def get_probabilities():
    """Returns the latest probability results for all drivers."""
    return list(state.probabilities.values())


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    """
    Real-time state stream. The frontend connects here and receives
    a JSON payload every ~3 seconds as the poller runs its cycle.
    """
    await ws.accept()
    poller.ws_connections.add(ws)
    client = ws.client
    log.info("WebSocket connected: %s:%s  (total: %d)", client.host, client.port, len(poller.ws_connections))

    try:
        # Send an immediate snapshot on connect (don't wait for next poll cycle)
        initial = LiveStatePayload(
            session          = state.session,
            drivers          = state.build_live_state(),
            events           = list(state.events),
            weather          = state.weather,
            probabilities    = list(state.probabilities.values()),
            last_updated     = state.last_updated,
            connection_count = len(poller.ws_connections),
            is_live          = state.session is not None,
        )
        await ws.send_text(initial.model_dump_json())

        # Keep connection alive — the poller handles broadcasting
        while True:
            # Listen for pings / close frames from client
            msg = await ws.receive_text()
            if msg == "ping":
                await ws.send_text('{"type":"pong"}')

    except WebSocketDisconnect:
        log.info("WebSocket disconnected: %s:%s", client.host, client.port)
    except Exception as exc:
        log.warning("WebSocket error: %s", exc)
    finally:
        poller.ws_connections.discard(ws)
        log.info("WebSocket removed. Active connections: %d", len(poller.ws_connections))
