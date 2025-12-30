# Deploying SignBridge to Render (Free Web Service)

This guide explains how to deploy SignBridge to Render as a **free Web Service**.

## Architecture

SignBridge uses a **single web service** on Render:
- **Python FastAPI Backend** - Handles API requests and serves static files
- **Static Frontend** - HTML/CSS/JS files served by FastAPI

## Prerequisites

- GitHub repository with your code
- Render account (free tier)
- Trained model file (optional, but recommended)

## Quick Deployment (Free Web Service)

### Step 1: Push Code to GitHub

```bash
git add .
git commit -m "Ready for Render deployment"
git push origin master
```

### Step 2: Create Web Service on Render

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **"New +"** → **"Web Service"** (NOT Blueprint)
3. Connect your GitHub account (if not already connected)
4. Select your repository: `signlanguage` (or your repo name)
5. Click **"Connect"**

### Step 3: Configure Web Service

Fill in the following settings:

**Basic Settings:**
- **Name**: `signbridge` (or any name you prefer)
- **Region**: Choose closest to you
- **Branch**: `master` (or `main`)
- **Root Directory**: Leave empty (root of repo)

**Build & Deploy:**
- **Environment**: `Python 3`
- **Build Command**: `pip install -r python-backend/requirements.txt`
- **Start Command**: `cd python-backend && python server.py`

**Advanced Settings:**
- **Auto-Deploy**: `Yes` (deploys on every git push)
- **Health Check Path**: `/health`

**Environment Variables:**
- No variables needed! Port is auto-set by Render

### Step 4: Deploy

1. Click **"Create Web Service"**
2. Wait for build to complete (5-10 minutes first time)
3. Your app will be live at: `https://signbridge.onrender.com`

## Configuration Details

### Build Command
```
pip install -r python-backend/requirements.txt
```

### Start Command
```
cd python-backend && python server.py
```

### Health Check
- Path: `/health`
- Render uses this to verify service is running

## Model File Setup

Since model files are large and excluded from git:

### Option 1: Upload via Render Shell (Recommended)

1. After deployment, go to your service → **Shell** tab
2. Create models directory:
   ```bash
   mkdir -p python-backend/models
   ```
3. Upload model file using one of these methods:

   **Method A: Using wget (if model is hosted online)**
   ```bash
   wget -O python-backend/models/sign_language_cnn_model.h5 <your-model-url>
   ```

   **Method B: Using Render's file upload**
   - Use Render's file manager (if available)
   - Or use `scp` to upload from your computer

### Option 2: Use Render Disk (Persistent Storage)

1. In service settings → **Disk** tab
2. Enable **Persistent Disk**
3. Upload model file to disk
4. Model persists across deployments

### Option 3: Download on Startup

Modify `server.py` to download model from cloud storage (S3, etc.) on startup.

## Accessing Your Application

After deployment:
- **Main Site**: `https://signbridge.onrender.com`
- **API Docs**: `https://signbridge.onrender.com/docs`
- **Health Check**: `https://signbridge.onrender.com/health`
- **Recognition Page**: `https://signbridge.onrender.com/recognition.html`
- **API Status**: `https://signbridge.onrender.com/api`

## Troubleshooting

### Build Fails
- **Check build logs** for error messages
- Verify `requirements.txt` is correct
- Ensure Python version is compatible (3.11.0)

### Service Won't Start
- Check **Logs** tab for errors
- Verify start command is correct
- Ensure port is set correctly (auto-set by Render)

### Model Not Found
- Verify model file path: `python-backend/models/sign_language_cnn_model.h5`
- Check file permissions
- Service works without model (uses simulated predictions)

### Static Files Not Loading
- Check file paths in HTML
- Verify static file mounting in `server.py`
- Check browser console for 404 errors

### Camera Not Working
- HTTPS is required for camera access
- Render provides HTTPS automatically
- Check browser console for permission errors
- Allow camera permissions in browser

## Free Tier Limitations

- **Spins down** after 15 minutes of inactivity
- **Cold start** on first request after spin-down (may take 30-60 seconds)
- **750 hours/month** free
- **512 MB RAM** limit
- **No persistent disk** (unless upgraded)

## Updating Your Deployment

1. Make changes to your code
2. Push to GitHub:
   ```bash
   git add .
   git commit -m "Update code"
   git push origin master
   ```
3. Render automatically redeploys (if auto-deploy is enabled)

## File Structure

```
SignLanguage/
├── render.yaml              # Optional: for Blueprint (not needed for Web Service)
├── index.html              # Homepage
├── recognition.html         # Recognition page
├── config.js               # Frontend configuration
├── python-backend/
│   ├── server.py           # FastAPI server
│   ├── requirements.txt    # Python dependencies
│   ├── runtime.txt        # Python version
│   └── models/             # Model files (upload separately)
└── ...
```

## Notes

- Model files are excluded from git (too large)
- Upload model separately after deployment
- Free tier may have cold starts
- All static files are served by FastAPI
- Backend URL auto-detects (no configuration needed)

## Quick Reference

**Service Settings:**
- Environment: Python 3
- Build: `pip install -r python-backend/requirements.txt`
- Start: `cd python-backend && python server.py`
- Health: `/health`

**No environment variables needed!**

---

**Ready to deploy?** Follow Step 2 above to create your free Web Service! 🚀
