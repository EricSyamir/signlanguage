# Deploying SignBridge to Render

This guide explains how to deploy SignBridge to Render as a single web service.

## Architecture

SignBridge uses a **single web service** on Render:
- **Python FastAPI Backend** - Handles API requests and serves static files
- **Static Frontend** - HTML/CSS/JS files served by FastAPI

## Prerequisites

- GitHub repository with your code
- Render account (free tier works)
- Trained model file (optional, but recommended)

## Quick Deployment

### Step 1: Push Code to GitHub

```bash
git add render.yaml
git commit -m "Add Render deployment configuration"
git push origin master
```

### Step 2: Deploy on Render

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **"New +"** → **"Blueprint"**
3. Connect your GitHub repository
4. Render will automatically detect `render.yaml`
5. Click **"Apply"**

### Step 3: Wait for Deployment

- Render will install dependencies
- Build the Python service
- Deploy your application
- You'll get a URL like: `https://signbridge.onrender.com`

## Configuration

The `render.yaml` file configures:
- **Service Type**: Web service (Python)
- **Build Command**: `pip install -r python-backend/requirements.txt`
- **Start Command**: `cd python-backend && python server.py`
- **Port**: Automatically set by Render
- **Health Check**: `/health` endpoint

## Model File Setup

Since model files are large and excluded from git:

### Option 1: Upload via Render Shell (Recommended)

1. After deployment, go to your service → **Shell**
2. Create models directory:
   ```bash
   mkdir -p python-backend/models
   ```
3. Upload model file:
   ```bash
   # Using wget (if model is hosted online)
   wget -O python-backend/models/sign_language_cnn_model.h5 <model-url>
   
   # Or use Render's file upload feature
   ```

### Option 2: Use Render Disk

1. Enable **Persistent Disk** in service settings
2. Upload model file to disk
3. Model persists across deployments

### Option 3: Download on Startup

Modify `server.py` to download model from cloud storage on startup.

## Environment Variables

No environment variables required! The service auto-detects:
- Port (set by Render)
- Backend URL (same as service URL)

## Accessing Your Application

After deployment:
- **Main Site**: `https://signbridge.onrender.com`
- **API Docs**: `https://signbridge.onrender.com/docs`
- **Health Check**: `https://signbridge.onrender.com/health`
- **Recognition Page**: `https://signbridge.onrender.com/recognition.html`

## Troubleshooting

### Backend won't start
- Check build logs for dependency errors
- Ensure `requirements.txt` is correct
- Verify Python version (3.11.0)

### Model not found
- Verify model file path: `python-backend/models/sign_language_cnn_model.h5`
- Check file permissions
- Service will work without model (uses simulated predictions)

### Static files not loading
- Check file paths in HTML
- Verify static file mounting in `server.py`
- Check browser console for 404 errors

### Camera not working
- HTTPS is required for camera access
- Render provides HTTPS automatically
- Check browser console for permission errors

## Cost Considerations

- **Free Tier**: 
  - Service spins down after 15 minutes of inactivity
  - First request after spin-down may be slow (cold start)
  - 750 hours/month free
- **Paid Tier**: 
  - Always running
  - Better performance
  - No cold starts

## Updating Your Deployment

1. Make changes to your code
2. Push to GitHub:
   ```bash
   git add .
   git commit -m "Update code"
   git push origin master
   ```
3. Render automatically redeploys on push

## File Structure

```
SignLanguage/
├── render.yaml              # Render configuration
├── index.html              # Homepage
├── recognition.html         # Recognition page
├── config.js               # Frontend configuration
├── python-backend/
│   ├── server.py           # FastAPI server
│   ├── requirements.txt    # Python dependencies
│   ├── models/             # Model files (upload separately)
│   └── runtime.txt        # Python version
└── ...
```

## Notes

- Model files are excluded from git (too large)
- Upload model separately after deployment
- Free tier may have cold starts
- Consider Render Disk for persistent model storage
- All static files are served by FastAPI

---

**Ready to deploy?** Push your code to GitHub and follow the steps above! 🚀
