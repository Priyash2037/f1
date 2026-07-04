# F1 Oracle 🏎️

**Real-time Formula 1 race probability dashboard** powered by:
- **OpenF1 REST API** — free live F1 telemetry (no API key)
- **Monte Carlo simulation** — 500 simulations per tick with tyre degradation, pit stop modelling, and lap-time noise
- **XGBoost ML model** — trained on 14 seasons of historical F1 data, blended with MC results
- **FastAPI + WebSocket** — real-time state streaming
- **Vite + React** — premium dark racing-broadcast UI

---

## Project Structure

```
f1-oracle/
├── backend/          # FastAPI + probability engine
│   ├── main.py       # App entry point + WebSocket endpoint
│   ├── poller.py     # OpenF1 REST polling (every 3s)
│   ├── state.py      # In-memory state store
│   ├── engine.py     # Monte Carlo + softmax probability engine
│   ├── models.py     # Pydantic data models
│   └── requirements.txt
├── ml/               # Machine Learning pipeline
│   ├── preprocess.py # Data fetch (Ergast API) + feature engineering
│   ├── train.py      # XGBoost training + GridSearchCV
│   ├── predict.py    # Inference wrapper for FastAPI
│   └── requirements.txt
└── frontend/         # Vite + React dashboard
    ├── src/
    │   ├── App.jsx
    │   ├── index.css           # Design system
    │   ├── hooks/
    │   │   └── useWebSocket.js # Auto-reconnecting WS client
    │   └── components/
    │       ├── Header.jsx
    │       ├── SessionInfo.jsx
    │       ├── Leaderboard.jsx
    │       ├── ProbabilityChart.jsx
    │       └── EventFeed.jsx
    └── package.json
```

---

## Quick Start

### 1. Backend

```powershell
cd backend
pip install fastapi "uvicorn[standard]" httpx websockets python-dotenv joblib xgboost pandas
uvicorn main:app --reload --port 8000
```

The backend will immediately start polling OpenF1 and streaming data.

### 2. Frontend (in a new terminal)

```powershell
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

### 3. ML Training (optional — enhances predictions)

```powershell
cd ml
# Step 1: Fetch Ergast historical data (2010–2024) + build features
python preprocess.py

# Step 2: Train XGBoost model (fast mode skips GridSearch)
python train.py --no-grid-search --test-season 2024

# Full training with GridSearch (takes 20–40 min)
python train.py --test-season 2024
```

The trained model is automatically saved to `ml/models/f1_win_s2024_model.joblib`.
The backend loads it on startup and blends ML predictions with Monte Carlo results.

---

## Probability Model

The win probability for driver $i$ uses a softmax over expected remaining race times:

$$P(W_i | \mathbf{x}) = \frac{\exp(-\lambda \cdot E[T_i])}{\sum_{j=1}^{N} \exp(-\lambda \cdot E[T_j])}$$

Where $E[T_i]$ is estimated via Monte Carlo simulation:
- **Base pace**: median of last 5 clean laps
- **Tyre degradation**: compound-specific deg rate (SOFT: 0.08s/lap, MEDIUM: 0.045s/lap)
- **Pit time loss**: 22s estimated per remaining pit stop
- **Lap noise**: Gaussian σ=0.5s (calibrated from historical variance)
- **λ=2.0**: temperature parameter (higher = more decisive leader advantage)

ML blend: `α=0.35 × XGBoost + 0.65 × Monte Carlo`

---

## Key Features

| Feature | Detail |
|---|---|
| Live leaderboard | Position, gap, tyre compound + age, lap times |
| Lap time coloring | Purple = overall best, Green = personal best |
| Probability chart | Win + Podium bars, team-colored, animated |
| Race control feed | SC, VSC, red/yellow flags with timestamps |
| Auto-reconnect | WebSocket reconnects with exponential backoff |
| Session awareness | Adapts to Race / Qualifying / Sprint sessions |
| Weather strip | Track & air temperature in session bar |

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Health check + connection count |
| `GET /api/state` | Full state snapshot (initial load) |
| `GET /api/session` | Current session metadata |
| `GET /api/probabilities` | Latest probability results |
| `WS /ws/live` | Real-time state stream (~3s updates) |
