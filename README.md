# MLOps Pipeline for MLOps Practical

This project demonstrates a MLOps lifecycle.

## Components:
1. **VCS & Data**: Version control with Git and dataset preparation (`data.py`).
2. **Training**: Model training and versioning (`train.py`).
3. **Containerization**: Dockerizing the FastAPI inference app (`Dockerfile`).
4. **Orchestration**: Airflow DAG for the ML pipeline (`dags/ml_pipeline.py`).
5. **CI/CD**: GitHub Actions for testing and building (`.github/workflows/ci.yml`).
6. **A/B Testing**: Random routing between two model versions in FastAPI (`main.py`).

## How to run:
1. `python data.py` (Generate data)
2. `python train.py` (Train models)
3. `uvicorn main:app --reload` (Start API)
