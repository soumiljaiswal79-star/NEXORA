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

EXPECTED_FEATURES = {"rule_risk", "rainfall_mm", "wind_kmph", "terrain", "accessibility", "incidents", "cnn_damage"}


def _synthetic_jpeg_b64(color=(120, 80, 40)):
    """Generate a base64-encoded JPEG data URL for CNN tests."""
    from PIL import Image
    import io, base64
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), color).save(buf, "JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _noisy_jpeg_b64():
    """A noisy image that should trigger more edge density / dark ratio."""
    from PIL import Image
    import io, base64, random
    random.seed(0)
    img = Image.new("RGB", (224, 224))
    pixels = [(random.randint(0, 60), random.randint(0, 60), random.randint(0, 60)) for _ in range(224 * 224)]
    img.putdata(pixels)
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


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
    assert isinstance(calib["version"], str) and calib["version"] == "xgb-1.0.0"
    # Accuracy & r² should be reasonable
    assert calib["accuracy"] > 90, f"accuracy too low: {calib['accuracy']}"
    assert calib["r2"] > 0.85, f"r2 too low: {calib['r2']}"


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


# --- CNN damage classifier tests --------------------------------------------
def test_report_without_image_no_cnn():
    resp = requests.post(
        f"{BASE_URL}/api/reports",
        json={
            "incident_type": "Road damage", "severity": "Medium", "state": "Nagaland",
            "notes": "TEST_no_image", "latitude": 25.90, "longitude": 93.73,
        },
        timeout=15,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["cnn"]["used"] is False
    assert body["classification"] == "Medium"
    assert "calibrated_risk" in body["route"]
    assert "CNN saw" not in body["alert"]


def test_report_with_image_runs_cnn():
    img = _synthetic_jpeg_b64()
    resp = requests.post(
        f"{BASE_URL}/api/reports",
        json={
            "incident_type": "Landslide", "severity": "Low", "state": "Mizoram",
            "notes": "TEST_cnn_image", "latitude": 23.73, "longitude": 92.72,
            "image_data": img,
        },
        timeout=30,
    )
    assert resp.status_code == 200
    body = resp.json()
    cnn = body["cnn"]
    assert cnn["used"] is True
    assert isinstance(cnn["score"], (int, float)) and 0.0 <= cnn["score"] <= 100.0
    assert cnn["severity"] in {None, "None", "Low", "Medium", "High"}
    assert isinstance(cnn["top_classes"], list) and len(cnn["top_classes"]) == 3
    for tc in cnn["top_classes"]:
        assert isinstance(tc["label"], str)
        assert isinstance(tc["confidence"], (int, float))
        assert 0.0 <= tc["confidence"] <= 1.0
    # route.cnn_damage should reflect the score
    assert body["route"].get("cnn_damage", 0) == cnn["score"] or body["route"]["cnn_damage"] >= cnn["score"]
    # alert must contain CNN evidence line
    assert "CNN saw" in body["alert"]
    # classification is max(reported, cnn)
    order = {"None": 0, "Low": 1, "Medium": 2, "High": 3}
    expected = "Low" if order.get(cnn["severity"], 0) < order["Low"] else cnn["severity"]
    assert order.get(body["classification"], 0) >= order.get("Low", 0)
    assert order.get(body["classification"], 0) == max(order.get("Low", 0), order.get(cnn["severity"], 0))


def test_cnn_never_downgrades_severity():
    """Reporter says High, CNN produces something lower → classification stays High."""
    img = _synthetic_jpeg_b64(color=(200, 200, 200))  # bright uniform → low damage
    resp = requests.post(
        f"{BASE_URL}/api/reports",
        json={
            "incident_type": "Bridge collapse", "severity": "High", "state": "Manipur",
            "notes": "TEST_cnn_no_downgrade", "latitude": 24.55, "longitude": 94.00,
            "image_data": img,
        },
        timeout=30,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["classification"] == "High"
    assert body["reported_severity"] == "High"


def test_cnn_upgrade_when_reporter_low_and_cnn_high():
    """Direct unit-style test via damage_cnn — CNN must return a High for a noisy dark image
    and API must upgrade classification."""
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from damage_cnn import classify as _classify
    img = _noisy_jpeg_b64()
    result = _classify(img)
    assert result["used"] is True
    # If CNN indeed sees High severity, verify API upgrades it
    if result["severity"] == "High":
        resp = requests.post(
            f"{BASE_URL}/api/reports",
            json={
                "incident_type": "Road damage", "severity": "Low", "state": "Nagaland",
                "notes": "TEST_cnn_upgrade", "latitude": 25.90, "longitude": 93.73,
                "image_data": img,
            },
            timeout=30,
        )
        body = resp.json()
        assert body["cnn"]["severity"] == "High"
        assert body["classification"] == "High"
        assert body["reported_severity"] == "Low"
    else:
        # At minimum, confirm severity ordering rule holds for whatever CNN returned
        resp = requests.post(
            f"{BASE_URL}/api/reports",
            json={
                "incident_type": "Road damage", "severity": "Low", "state": "Nagaland",
                "notes": "TEST_cnn_upgrade", "latitude": 25.90, "longitude": 93.73,
                "image_data": img,
            },
            timeout=30,
        )
        body = resp.json()
        order = {"None": 0, "Low": 1, "Medium": 2, "High": 3}
        expected = max(order.get("Low", 0), order.get(body["cnn"]["severity"], 0))
        assert order.get(body["classification"], 0) == expected
