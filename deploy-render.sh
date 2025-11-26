#!/bin/bash

# Render.com Deployment (FREE - No Credit Card Needed!)
# Free tier for web services

set -e

echo "🎨 Render.com Deployment Instructions"
echo "======================================"
echo ""
echo "Render offers free tier with:"
echo "- 750 hours/month free"
echo "- Automatic HTTPS"
echo "- Auto-deploys from Git"
echo ""
echo "Setup Steps:"
echo ""
echo "1. Go to: https://render.com"
echo "2. Sign up with GitHub"
echo "3. Click 'New +' → 'Web Service'"
echo "4. Connect this GitHub repo: GX-47/Test"
echo "5. Configure:"
echo "   - Name: legal-rag-api"
echo "   - Environment: Docker"
echo "   - Branch: main"
echo "   - Instance Type: Free"
echo ""
echo "6. Add Environment Variables:"
echo "   MISTRAL_API_BASE_URL = your-ngrok-url"
echo "   BASIC_AUTH_USER = legal_user"
echo "   BASIC_AUTH_PASS = super_secure_password123"
echo "   PORT = 8080"
echo ""
echo "7. Click 'Create Web Service'"
echo ""
echo "Your API will be live at: https://legal-rag-api.onrender.com"
echo ""
echo "Note: Free tier sleeps after 15 min of inactivity (cold starts ~30s)"
