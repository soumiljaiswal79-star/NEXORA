import os
import requests

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL").rstrip("/")


def test_overview_contract():
    response = requests.get(f"{BASE_URL}/api/overview", timeout=15)
    assert response.status_code == 200
    data = response.json()
    assert set(data["states"]) == {"Assam", "Manipur", "Meghalaya", "Nagaland", "Mizoram"}
    assert len(data["routes"]) == 7
    assert len(data["vehicles"]) == 6
    assert len(data["incidents"]) == 5
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