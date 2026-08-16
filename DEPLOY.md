# Deployment notes

## Streamlit Community Cloud

Create an app from the repository and select `streamlit_app.py` as the main file. Community Cloud installs from `requirements.txt`; this project has no secrets, API keys, local model binaries, or external service dependencies.

The bundled demo files are loaded only after the visitor selects **Load bundled demo data**. Training is bounded to a configurable maximum (10,000 rows by default), uses one CPU process, and stores the model under the process temporary directory. Those decisions are deliberate: a Community Cloud instance can restart and its local filesystem is ephemeral.

## Operational expectations

- A browser refresh or instance restart can clear the current model and uploaded data.
- Do not upload personal or confidential credit data to this demo deployment.
- Use the app for portfolio demonstration and model-review exploration only.
- Downloaded CSV files are model outputs for review, not final lending decisions.

## Before a production deployment

Put storage, authentication, audit logging, encrypted data handling, model versioning, monitoring, access controls, approval workflows, and an independently governed adverse-action/reason-code process behind a service boundary. Validate with temporal splits, calibration, subgroup uncertainty, drift tests, and qualified legal/compliance review.
