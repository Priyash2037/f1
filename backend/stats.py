import httpx
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api")

JOLPI_BASE_URL = "https://api.jolpi.ca/ergast/f1/current"

# We use httpx.AsyncClient without a session for simple one-off proxy requests
async def fetch_jolpi(endpoint: str):
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{JOLPI_BASE_URL}/{endpoint}.json", timeout=10.0)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Jolpi API error: {str(e)}")

@router.get("/standings/drivers")
async def get_driver_standings():
    data = await fetch_jolpi("driverStandings")
    try:
        standings = data["MRData"]["StandingsTable"]["StandingsLists"][0]["DriverStandings"]
        return {"standings": standings}
    except (KeyError, IndexError):
        return {"standings": []}

@router.get("/standings/constructors")
async def get_constructor_standings():
    data = await fetch_jolpi("constructorStandings")
    try:
        standings = data["MRData"]["StandingsTable"]["StandingsLists"][0]["ConstructorStandings"]
        return {"standings": standings}
    except (KeyError, IndexError):
        return {"standings": []}

@router.get("/results/races")
async def get_race_results():
    data = await fetch_jolpi("results")
    try:
        races = data["MRData"]["RaceTable"]["Races"]
        return {"races": races}
    except KeyError:
        return {"races": []}
