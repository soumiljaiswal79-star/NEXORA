# NE—LOGIX Product Requirements Document

## Original problem statement

Build a scoped, end-to-end prototype of an AI-based Smart Logistics and Accessibility Intelligence Platform for India’s North Eastern Region. The five-minute demo must connect field data, disruption prediction, risk-aware route optimization, live delivery tracking, multilingual alerts, and responsive offline-first reporting across Assam, Manipur, Meghalaya, Nagaland, and Mizoram.

## Architecture decisions

- React dashboard with Leaflet for the first-viewport live map and route overlays.
- FastAPI `/api/overview` and `/api/reports` endpoints.
- Transparent rule-based risk scoring with a synthetic calibration signal representing the optional XGBoost layer.
- In-memory GIS-shaped fixtures for zero external setup; PostGIS is the production-oriented repository seam.
- English and Assamese labels, with report submission driving the visible alert/reroute loop.

## User personas

- Logistics manager: monitors corridor health, deliveries, risk, and reroutes.
- Field officer: submits a location-based damage report from a mobile-friendly form.
- District administrator: filters the regional view by state and sees live alerts and bottlenecks.

## Core requirements (static)

- Equal representation across five NER states.
- Live route risk, incident markers, vehicle status, payload, ETA, and accessibility context.
- State and role filtering, English/Assamese toggle, and report form with photo selection.
- Report → risk spike → alert → vehicle reroute visible in one session.
- Fast start with no credentials or database configuration.

## Implemented (2026-09-17)

- Tactical NE—LOGIX dashboard with responsive dark visual system, live metrics, route health, bottlenecks, alerts, and delivery pulse.
- Leaflet map with risk-colored corridors, incident markers, vehicle markers, popups, and monsoon context.
- Five-state mock dataset with seven corridors, six vehicles, incidents, mixed payloads, and statuses.
- Report API that appends incidents, increases route risk, reroutes matching vehicles, and refreshes the alert feed.
- English/Assamese toggle, role selector, state filters, mobile report modal, image input, and sync toast.
- README and end-to-end regression coverage.

## Prioritized backlog

### P0 remaining

- Replace in-memory fixtures with PostGIS-backed route geometry when persistent spatial storage is available.
- Add real model inference behind the existing calibration contract.

### P1 remaining

- Add route selection controls for switching a vehicle to alternate paths.
- Add a dedicated report history view with synced/unsynced states.
- Add time-series risk chart for rainfall and incident changes.

### P2 remaining

- Add real IMD/Bhuvan/GPS adapters.
- Add delivery export and district-level operational summaries.

## Next tasks

1. Add three alternate routes per delivery using persisted geometries.
2. Connect photo upload to a local CNN inference worker.
3. Add notification delivery adapters after the demo flow is stable.