#!/bin/bash

# Google Cloud Run Deployment Script for Legal RAG API
# This script builds and deploys your FastAPI application to Google Cloud Run

set -e  # Exit on error

# Configuration
PROJECT_ID="gen-lang-client-0304871297"  # Your Legify project
SERVICE_NAME="legal-rag-api"
REGION="us-central1"  # Change if needed (us-central1, europe-west1, asia-east1, etc.)
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Legal RAG API - Google Cloud Run Deploy${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI is not installed${NC}"
    echo "Install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if PROJECT_ID is set
if [ "$PROJECT_ID" = "your-gcp-project-id" ]; then
    echo -e "${RED}Error: Please set your PROJECT_ID in this script${NC}"
    echo "Edit deploy-gcloud.sh and replace 'your-gcp-project-id' with your actual GCP project ID"
    exit 1
fi

# Authenticate and set project
echo -e "${YELLOW}Step 1: Setting GCP project...${NC}"
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo -e "${YELLOW}Step 2: Enabling required APIs...${NC}"
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build the Docker image using Cloud Build
echo -e "${YELLOW}Step 3: Building Docker image with Cloud Build...${NC}"
gcloud builds submit --tag ${IMAGE_NAME}

# Deploy to Cloud Run
echo -e "${YELLOW}Step 4: Deploying to Cloud Run...${NC}"
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10 \
  --set-env-vars "MISTRAL_API_BASE_URL=${MISTRAL_API_BASE_URL}" \
  --set-env-vars "BASIC_AUTH_USER=${BASIC_AUTH_USER:-legal_user}" \
  --set-env-vars "BASIC_AUTH_PASS=${BASIC_AUTH_PASS:-super_secure_password123}"

# Get the service URL
echo -e "${YELLOW}Step 5: Getting service URL...${NC}"
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --platform managed --region ${REGION} --format 'value(status.url)')

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Deployment Successful! 🎉${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Service URL: ${GREEN}${SERVICE_URL}${NC}"
echo -e "Health Check: ${GREEN}${SERVICE_URL}/api/v1/health${NC}"
echo -e "API Docs: ${GREEN}${SERVICE_URL}/docs${NC}"
echo ""
echo -e "${YELLOW}Test your API:${NC}"
echo "curl ${SERVICE_URL}/api/v1/health"
echo ""
echo -e "${YELLOW}Query endpoint (with auth):${NC}"
echo "curl -X POST ${SERVICE_URL}/api/v1/query \\"
echo "  -u legal_user:super_secure_password123 \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"query\": \"What are property dispute requirements?\"}'"
echo ""
