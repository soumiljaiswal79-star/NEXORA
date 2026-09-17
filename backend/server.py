from fastapi import FastAPI, APIRouter
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# The demo keeps its GIS-shaped fixtures in memory so it starts with zero setup.
# Production would swap this repository for PostGIS using the same response contract.

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


# Define Models
class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")  # Ignore MongoDB's _id field
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str

status_checks = []

class FieldReport(BaseModel):
    incident_type: str
    severity: str
    state: str
    notes: str
    latitude: float
    longitude: float
    image_data: Optional[str] = None

states = ["Assam", "Manipur", "Meghalaya", "Nagaland", "Mizoram"]
routes = [
    {"id":"NH-27 / Guwahati–Silchar", "state":"Assam", "risk":72, "accessibility":65, "terrain":"Hills", "delay":"+2h 10m", "incidents":4, "points":[[26.14,91.74],[25.57,91.88],[24.83,92.78]]},
    {"id":"NH-40 / Shillong–Jowai", "state":"Meghalaya", "risk":46, "accessibility":58, "terrain":"Mountains", "delay":"+48m", "incidents":2, "points":[[25.58,91.89],[25.45,92.20],[25.30,92.50]]},
    {"id":"NH-2 / Imphal–Moreh", "state":"Manipur", "risk":64, "accessibility":42, "terrain":"Mountains", "delay":"+1h 25m", "incidents":3, "points":[[24.82,93.94],[24.55,94.00],[24.25,94.30]]},
    {"id":"NH-29 / Dimapur–Kohima", "state":"Nagaland", "risk":38, "accessibility":61, "terrain":"Hills", "delay":"+32m", "incidents":2, "points":[[25.90,93.73],[25.67,94.10],[25.67,94.12]]},
    {"id":"NH-306 / Aizawl–Lunglei", "state":"Mizoram", "risk":57, "accessibility":39, "terrain":"Mountains", "delay":"+1h 05m", "incidents":3, "points":[[23.73,92.72],[23.20,92.72],[22.88,92.75]]},
    {"id":"SH-3 / Jorhat–Dibrugarh", "state":"Assam", "risk":22, "accessibility":78, "terrain":"Plains", "delay":"+12m", "incidents":1, "points":[[26.75,94.20],[27.47,94.91]]},
    {"id":"NH-102 / Imphal–Kakching", "state":"Manipur", "risk":28, "accessibility":70, "terrain":"Hills", "delay":"+20m", "incidents":1, "points":[[24.82,93.94],[24.60,93.98]]},
]
vehicles = [
    {"id":"V-014","state":"Assam","payload":"Medical supplies","status":"in-transit","eta":"02h 18m","route":"NH-27 / Guwahati–Silchar","position":[25.67,92.22],"type":"Cargo truck"},
    {"id":"V-021","state":"Manipur","payload":"Food grains","status":"rerouted","eta":"03h 42m","route":"NH-2 / Imphal–Moreh","position":[24.62,93.98],"type":"Cargo truck"},
    {"id":"V-033","state":"Meghalaya","payload":"Emergency fuel","status":"delayed","eta":"01h 06m","route":"NH-40 / Shillong–Jowai","position":[25.46,92.16],"type":"Van"},
    {"id":"V-047","state":"Nagaland","payload":"Construction materials","status":"in-transit","eta":"00h 54m","route":"NH-29 / Dimapur–Kohima","position":[25.78,93.95],"type":"Cargo truck"},
    {"id":"V-052","state":"Mizoram","payload":"Medical supplies","status":"in-transit","eta":"04h 11m","route":"NH-306 / Aizawl–Lunglei","position":[23.42,92.72],"type":"Van"},
    {"id":"V-061","state":"Assam","payload":"Food grains","status":"delivered","eta":"Delivered","route":"SH-3 / Jorhat–Dibrugarh","position":[27.10,94.55],"type":"Cargo truck"},
]
incidents = [
    {"state":"Assam","type":"Flooding","severity":"High","time":"12 min ago","location":[25.57,91.88]},
    {"state":"Manipur","type":"Bridge collapse","severity":"High","time":"28 min ago","location":[24.55,94.00]},
    {"state":"Meghalaya","type":"Landslide","severity":"Medium","time":"41 min ago","location":[25.45,92.20]},
    {"state":"Nagaland","type":"Road damage","severity":"Medium","time":"1h ago","location":[25.67,94.10]},
    {"state":"Mizoram","type":"Vehicle breakdown","severity":"Low","time":"2h ago","location":[23.20,92.72]},
]

# Add your routes to the router instead of directly to app
@api_router.get("/")
async def root():
    return {"message": "NE-LOGIX API online", "mode": "demo"}

@api_router.get("/overview")
async def overview():
    return {"states": states, "routes": routes, "vehicles": vehicles, "incidents": incidents,
            "sync_time":"just now", "uptime":"99.98%", "calibration":{"accuracy":89,"confidence":0.84}}

@api_router.post("/reports")
async def create_report(report: FieldReport):
    bump = 18 if report.severity == "High" else 10 if report.severity == "Medium" else 5
    target = next((route for route in routes if route["state"] == report.state), routes[0])
    target["risk"] = min(98, target["risk"] + bump)
    target["incidents"] += 1
    incidents.insert(0, {"state": report.state, "type": report.incident_type, "severity": report.severity,
                         "time": "just now", "location": [report.latitude, report.longitude]})
    for vehicle in vehicles:
        if vehicle["state"] == report.state and vehicle["status"] != "delivered":
            vehicle["status"] = "rerouted"
            vehicle["eta"] = "03h 08m"
    return {"ok": True, "report_id": str(uuid.uuid4()), "route": target,
            "alert": f"{target['id']} risk increased to {target['risk']} — 2 vehicles rerouted",
            "classification": report.severity, "synced": False}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.model_dump()
    status_obj = StatusCheck(**status_dict)
    
    # Convert to dict and serialize datetime to ISO string for MongoDB
    doc = status_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    
    status_checks.append(doc)
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    # Exclude MongoDB's _id field from the query results
    # Convert ISO string timestamps back to datetime objects
    for check in status_checks[:1000]:
        if isinstance(check['timestamp'], str):
            check['timestamp'] = datetime.fromisoformat(check['timestamp'])
    
    return status_checks

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
