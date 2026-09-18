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

EXPECTED_FEATURES = {"rule_risk", "rainfall_mm", "wind_kmph", "terrain", "accessibility", "incidents"}


# --- Root / health -----------------------------------------------------------
def test_root():
    r = requests.get(f"{BASE_URL}/api/", timeout=15)
    assert r.status_code == 200
    assert r.json().get("message") == "NE-LOGIX API online"


# --- /api/overview contract (existing) --------------------------------------
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


# --- XGBoost calibration payload --------------------------------------------
def test_overview_calibration_is_real_xgb():
    data = requests.get(f"{BASE_URL}/api/overview", timeout=15).json()
    calib = data["calibration"]
    # Required keys
    for key in ["accuracy", "confidence", "mae", "r2", "top_feature",
                "feature_importance", "training_samples", "test_samples", "version"]:
        assert key in calib, f"missing calibration key: {key}"
    # Types & ranges
    assert isinstance(calib["accuracy"], (int, float))
    assert isinstance(calib["confidence"], (int, float))
    assert 0.0 <= calib["confidence"] <= 1.0, f"confidence out of [0,1]: {calib['confidence']}"
    assert isinstance(calib["mae"], (int, float)) and calib["mae"] > 0
    assert isinstance(calib["r2"], (int, float))
    assert calib["top_feature"] in EXPECTED_FEATURES
    assert isinstance(calib["feature_importance"], dict)
    assert set(calib["feature_importance"].keys()) == EXPECTED_FEATURES
    imp_sum = sum(calib["feature_importance"].values())
    assert 0.9 <= imp_sum <= 1.1, f"feature_importance sum not ~1: {imp_sum}"
    assert isinstance(calib["training_samples"], int) and calib["training_samples"] > 0
    assert isinstance(calib["test_samples"], int) and calib["test_samples"] > 0
    assert isinstance(calib["version"], str) and calib["version"].startswith("xgb-")
    # Must NOT be the previous hardcoded stub values
    assert calib["accuracy"] != 89, "accuracy is still the hardcoded 89"
    assert calib["confidence"] != 0.84, "confidence is still the hardcoded 0.84"


def test_overview_routes_have_calibrated_risk():
    data = requests.get(f"{BASE_URL}/api/overview", timeout=15).json()
    for r in data["routes"]:
        assert "calibrated_risk" in r, f"route missing calibrated_risk: {r['id']}"
        cr = r["calibrated_risk"]
        assert isinstance(cr, (int, float))
        assert 0.0 <= cr <= 100.0, f"calibrated_risk out of range: {cr}"
        assert "rainfall" in r and "wind" in r


# --- /api/reports ------------------------------------------------------------
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
    assert "calibrated" in result["alert"].lower()
    assert "calibrated_risk" in result["route"]
    assert isinstance(result["route"]["calibrated_risk"], (int, float))
    after = requests.get(f"{BASE_URL}/api/overview", timeout=15).json()
    assert next(v for v in after["vehicles"] if v["state"] == "Assam" and v["id"] == "V-014")["status"] == "rerouted"


def test_meghalaya_high_severity_report_calibrated_risk_not_lower():
    # Grab pre-report calibrated risk for Meghalaya primary route
    before = requests.get(f"{BASE_URL}/api/overview", timeout=15).json()
    pre = next(r for r in before["routes"] if r["state"] == "Meghalaya" and not r.get("is_alternate"))
    pre_cal = pre["calibrated_risk"]
    resp = requests.post(
        f"{BASE_URL}/api/reports",
        json={
            "incident_type": "Landslide",
            "severity": "High",
            "state": "Meghalaya",
            "notes": "TEST_meghalaya slide",
            "latitude": 25.45,
            "longitude": 92.20,
        },
        timeout=15,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "calibrated_risk" in body["route"]
    post_cal = body["route"]["calibrated_risk"]
    # Because rainfall + rule_risk + incidents all rose, calibrated should not drop meaningfully.
    assert post_cal >= pre_cal - 1.0, f"calibrated_risk dropped: pre={pre_cal} post={post_cal}"
    assert "calibrated" in body["alert"].lower()


# --- /api/vehicles/{id}/route -----------------------------------------------
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

    bad = requests.post(f"{BASE_URL}/api/vehicles/{vehicle['id']}/route", json={"route_id": "NH-NOT-A-ROUTE"}, timeout=15)
    assert bad.status_code == 200
    assert bad.json()["ok"] is False

    bad2 = requests.post(f"{BASE_URL}/api/vehicles/DOES-NOT-EXIST/route", json={"route_id": target}, timeout=15)
    assert bad2.status_code == 200
    assert bad2.json()["ok"] is False
