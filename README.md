# NEXORA

NEXORA is a demo-ready Smart Logistics & Accessibility Intelligence dashboard for India’s North Eastern Region.

## Run

The project uses the provided React and FastAPI services. Start the frontend and backend with the workspace service runner; no external database, API keys, or setup is required.

## Demo story

1. Open the live corridor situation map to see risk-colored routes, incidents, and moving delivery vehicles across Assam, Manipur, Meghalaya, Nagaland, and Mizoram.
2. Use state filters or the role selector to focus the operational view.
3. Select **Field report**, submit a high-severity incident, and watch the risk score spike, a new alert appear, and vehicles in that state become rerouted.
4. Switch to Assamese using the language control for field-ready labels.

## Architecture

React + Leaflet renders the dashboard and map. FastAPI exposes `/api/overview` and `/api/reports`. The repository is intentionally an in-memory, GIS-shaped demo fixture so it starts in seconds with no Replit database setup. The production-shaped seam is ready for PostGIS route geometries.

Risk scoring is transparent rule-based logic: rainfall/terrain context, incident density, severity, and accessibility combine into a 0–100 route score. The dashboard exposes a synthetic calibration signal (89% accuracy, 0.84 confidence) to demonstrate the optional lightweight model layer without slowing the demo with model downloads.

IMD weather, Bhuvan DEM terrain, GPS, CNN photo classification, XGBoost calibration, and PostGIS persistence are **MOCKED** with realistic contextual data. The report form supports image selection and offline-first copy, while the demo endpoint synchronizes the report into the live incident feed.

## Known demo limitations

- Data resets when the backend process restarts.
- The map uses OpenStreetMap tiles and needs network access for the basemap; all operational data is local to the demo API.
- The exact incident-count assertion in the generated regression test is state-dependent after reports accumulate; dynamic product behavior is intentional.