#!/bin/bash
# WC 2026 Forecast — GCP Deployment
# Run once to set up; re-run just the "deploy" step to push updates.
#
# Prerequisites:
#   1. gcloud CLI installed  →  brew install google-cloud-sdk
#   2. Authenticated         →  gcloud auth login
#   3. Project created       →  set PROJECT_ID below (must be globally unique)
#   4. Billing enabled on project (required for Cloud Run even in free tier)
#
# Usage:
#   chmod +x deploy.sh
#   GEMINI_API_KEY=your_key ./deploy.sh

set -euo pipefail

# ── Config — change PROJECT_ID to match your GCP project ──────────────────
PROJECT_ID="wc2026-forecast"
REGION="us-central1"
SERVICE="wc2026-forecast"
# ──────────────────────────────────────────────────────────────────────────

GROQ_KEY="${GROQ_API_KEY:-}"
TAVILY_KEY="${TAVILY_API_KEY:-}"
IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/wc2026/$SERVICE:latest"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  WC 2026 Forecast — GCP Deploy"
echo "  Project : $PROJECT_ID"
echo "  Region  : $REGION"
echo "  Service : $SERVICE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Step 1: Set active project ─────────────────────────────────────────────
echo ""
echo "[1/6] Setting project..."
gcloud config set project "$PROJECT_ID"

# ── Step 2: Enable required APIs ──────────────────────────────────────────
echo ""
echo "[2/6] Enabling APIs (run, artifactregistry, cloudbuild, cloudscheduler)..."
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  cloudscheduler.googleapis.com \
  --quiet

# ── Steps 3-5: Build + Deploy in one command ──────────────────────────────
# `gcloud run deploy --source .` builds via Cloud Build and deploys atomically.
echo ""
echo "[3/6] Skipping separate repo create (--source handles it)"
echo ""
echo "[4-5/6] Building image and deploying to Cloud Run..."

ENV_VARS="N_SIMS=50000"
if [ -n "$GROQ_KEY" ] && [ -n "$TAVILY_KEY" ]; then
  ENV_VARS="$ENV_VARS,GROQ_API_KEY=$GROQ_KEY,TAVILY_API_KEY=$TAVILY_KEY"
  echo "  Groq + Tavily keys: included ✓"
else
  echo "  Form keys: not set (StatsBomb + pedigree only)"
fi

gcloud run deploy "$SERVICE" \
  --source . \
  --platform managed \
  --region "$REGION" \
  --allow-unauthenticated \
  --memory 512Mi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 2 \
  --timeout 120 \
  --set-env-vars "$ENV_VARS" \
  --quiet

SERVICE_URL=$(gcloud run services describe "$SERVICE" \
  --region "$REGION" \
  --format "value(status.url)")

echo ""
echo "  ✓ Live at: $SERVICE_URL"

# ── Step 6: Cloud Scheduler — refresh every 6 hours ───────────────────────
echo ""
echo "[6/6] Setting up Cloud Scheduler (every 6 hours)..."

gcloud scheduler jobs create http wc2026-refresh \
  --schedule="0 */6 * * *" \
  --uri="$SERVICE_URL/refresh" \
  --http-method=POST \
  --location="$REGION" \
  --description="Refresh WC 2026 forecast every 6 hours" \
  --quiet 2>/dev/null || \
gcloud scheduler jobs update http wc2026-refresh \
  --schedule="0 */6 * * *" \
  --uri="$SERVICE_URL/refresh" \
  --http-method=POST \
  --location="$REGION" \
  --quiet

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  DONE"
echo ""
echo "  Dashboard : $SERVICE_URL"
echo "  API       : $SERVICE_URL/api/odds"
echo "  Refresh   : curl -X POST $SERVICE_URL/refresh"
echo ""
echo "  Cloud Scheduler fires every 6h → POST /refresh"
echo "  Trigger a manual refresh now:"
echo "    curl -X POST $SERVICE_URL/refresh"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
