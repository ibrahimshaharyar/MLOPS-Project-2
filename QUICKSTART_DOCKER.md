# Quickstart: Running with Docker 🐳

This guide will help you build and run the Chicken Disease Classification API using Docker. This avoids local environment issues (like TensorFlow on Mac).

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

## 1. Build the Docker Image

Run this command in the project root:

```bash
docker build -t chicken-app .
```

## 2. Run the Container

Start the API server on port 8000:

```bash
docker run -p 8000:8000 --name chicken-classifier chicken-app
```

You should see output indicating `Uvicorn running on http://0.0.0.0:8000`.

## 3. Test the API

You can test the prediction endpoint using `curl`. Make sure you have a `test_image.jpg` handy.

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@test_image.jpg;type=image/jpeg'
```

**Expected Response:**

```json
{
  "label": "Healthy",
  "confidence": 0.98,
  "prob_healthy": 0.98
}
```

## Troubleshooting

- **Image not found**: Ensure you ran `docker build` successfully.
- **Port busy**: Change the port mapping, e.g., `-p 8001:8000`, and calculate `localhost:8001`.
- **Model not found**: Ensure `artifacts/model/best_model.keras` exists before building.
