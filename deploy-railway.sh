#!/bin/bash

# Railway Deployment (FREE - No Credit Card Needed!)
# 500 hours/month free tier

set -e

echo "🚂 Railway Deployment for Legal RAG API"
echo "========================================"
echo ""

# Check if railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "Installing Railway CLI..."
    npm install -g @railway/cli
fi

echo "Step 1: Login to Railway"
railway login

echo ""
echo "Step 2: Initialize Railway project"
railway init

echo ""
echo "Step 3: Link to Railway project"
railway link

echo ""
echo "Step 4: Add environment variables"
echo "Please set your MISTRAL_API_BASE_URL:"
read -p "Enter your Mistral API URL: " MISTRAL_URL
railway variables set MISTRAL_API_BASE_URL="$MISTRAL_URL"
railway variables set BASIC_AUTH_USER="legal_user"
railway variables set BASIC_AUTH_PASS="super_secure_password123"
railway variables set PORT="8080"

echo ""
echo "Step 5: Deploy!"
railway up

echo ""
echo "✅ Deployment complete!"
echo ""
echo "Get your URL with: railway domain"
echo "View logs with: railway logs"
