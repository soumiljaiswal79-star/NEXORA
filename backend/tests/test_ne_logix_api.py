import os
import requests
from pathlib import Path


def _load_backend_url():
    url = os.environ.get("REACT_APP_BACKEND_URL")
    if not url:
        env_file = Path(__file__).resolve().parents[2] / "frontend" / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("REACT_APP_BACKEND_URL="):
                    url = line.split("=", 1)[1].strip()
                    break
    assert url, "REACT_APP_BACKEND_URL is not configured"
    return url.rstrip("/")


BASE_URL = _load_backend_url()


def test_overview_contract():
    response = requests.get(f"{BASE_URL}/api/overview", timeout=15)
    assert response.status_code == 200
    data = response.json()
    assert set(data["states"]) == {"Assam", "Manipur", "Meghalaya", "Nagaland", "Mizoram"}
    assert len(data["routes"]) >= 7
    assert any(r.get("is_alternate") for r in data["routes"])
    assert len(data["vehicles"]) == 6
    assert all("route_options" in v and len(v["route_options"]) >= 1 for v in data["vehicles"])
    assert len(data["incidents"]) >= 5
    assert data["uptime"] == "99.98%"
    assert data["calibration"]["accuracy"] == 89


def test_field_report_spikes_risk_and_reroutes_vehicle():
    before = requests.get(f"{BASE_URL}/api/overview", timeout=15).json()
    route = next(r for r in before["routes"] if r["state"] == "Assam")
    original_risk = route["risk"]
    original_incidents = route["incidents"]
    response = requests.post(
        f"{BASE_URL}/api/reports",
        json={
            "incident_type": "Landslide",
            "severity": "High",
            "state": "Assam",
            "notes": "TEST_report blocking lane",
            "latitude": 25.57,
            "longitude": 91.88,
        },
        timeout=15,
    )
    assert response.status_code == 200
    result = response.json()
    assert result["ok"] is True
    assert result["classification"] == "High"
    assert result["route"]["risk"] == min(98, original_risk + 18)
    assert result["route"]["incidents"] == original_incidents + 1
    assert "rerouted" in result["alert"]
    after = requests.get(f"{BASE_URL}/api/overview", timeout=15).json()
    assert next(v for v in after["vehicles"] if v["state"] == "Assam" and v["id"] == "V-014")["status"] == "rerouted"

def test_root():
    r = requests.get(f"{BASE_URL}/api/", timeout=15)
    assert r.status_code == 200
    assert r.json().get("message") == "NE-LOGIX API online"


def test_switch_route_valid_and_invalid():
    overview = requests.get(f"{BASE_URL}/api/overview", timeout=15).json()
    vehicle = next(v for v in overview["vehicles"] if v["state"] == "Manipur")
    alternates = [rid for rid in vehicle["route_options"] if rid != vehicle["route"]]
    assert alternates, "expected alternate route options"
    target = alternates[0]
    resp = requests.post(f"{BASE_URL}/api/vehicles/{vehicle['id']}/route", json={"route_id": target}, timeout=15)
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["vehicle"]["route"] == target
    assert body["vehicle"]["status"] == "rerouted"
    after = requests.get(f"{BASE_URL}/api/overview", timeout=15).json()
    fetched = next(v for v in after["vehicles"] if v["id"] == vehicle["id"])
    assert fetched["route"] == target
    assert fetched["status"] == "rerouted"

    # Invalid route id (route not in vehicle's state)
    bad = requests.post(f"{BASE_URL}/api/vehicles/{vehicle['id']}/route", json={"route_id": "NH-NOT-A-ROUTE"}, timeout=15)
    assert bad.status_code == 200
    assert bad.json()["ok"] is False

    # Invalid vehicle id
    bad2 = requests.post(f"{BASE_URL}/api/vehicles/DOES-NOT-EXIST/route", json={"route_id": target}, timeout=15)
    assert bad2.status_code == 200
    assert bad2.json()["ok"] is False
