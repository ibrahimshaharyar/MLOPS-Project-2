# Deploying to Render ☁️

This guide explains how to deploy the Chicken Disease Classification API to [Render](https://render.com/).

## Prerequisites

1.  **Push to GitHub**: Ensure your code (including `artifacts/model/best_model.keras`) is pushed to a GitHub repository.
2.  **Render Account**: Create an account at [render.com](https://render.com/).

## Step-by-Step Deployment

1.  **New Web Service**:
    - Go to your Render Dashboard.
    - Click **New +** -> **Web Service**.
    - Connect your GitHub repository.

2.  **Configure Settings**:
    - **Name**: `chicken-classifier` (or your preferred name)
    - **Runtime**: **Docker** (Render will automatically detect the `Dockerfile`)
    - **Region**: Choose the one closest to you.
    - **Branch**: `main` (or your working branch)

3.  **Environment Variables** (Optional):
    - Render automatically sets the `PORT` variable.
    - Our `Dockerfile` is configured to listen on `$PORT`.

4.  **Deploy**:
    - Click **Create Web Service**.
    - Render will build your Docker image and deploy it. This may take a few minutes.

## Verification

Once deployed, you will get a URL like `https://chicken-classifier.onrender.com`.

Test it using `curl`:

```bash
curl -X 'POST' \
  'https://YOUR-APP-NAME.onrender.com/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@test_image.jpg'
```

## Troubleshooting

- **Build Failed**: Check the Render logs. Common issues include missing files in Git (check `.gitignore`).
- **"Port binding" error**: Ensure your `Dockerfile` uses the `CMD` with `${PORT:-8000}` (we already configured this!).
