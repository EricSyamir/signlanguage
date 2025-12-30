# Deploying SignBridge to Render

This guide explains how to deploy SignBridge to Render with both backend and frontend services.

## Architecture

SignBridge requires two services on Render:
1. **Backend Service** - Python FastAPI server (web service)
2. **Frontend Service** - Static HTML/CSS/JS site (static site)

## Prerequisites

- GitHub repository with your code
- Render account (free tier works)
- Trained model file (optional, but recommended)

## Deployment Steps

### Option 1: Using render.yaml (Recommended)

1. **Push code to GitHub**
   ```bash
   git add render.yaml
   git commit -m "Add Render deployment configuration"
   git push origin master
   ```

2. **Create New Web Service on Render**
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New +" → "Blueprint"
   - Connect your GitHub repository
   - Render will automatically detect `render.yaml`
   - Click "Apply"

3. **Configure Services**
   - Render will create two services:
     - `signbridge-backend` (Python web service)
     - `signbridge-frontend` (Static site)

4. **Add Model File (Optional)**
   - If you have a trained model, upload it via Render's file system or use environment variables
   - Or use Render's persistent disk feature

### Option 2: Manual Setup

#### Backend Service

1. **Create New Web Service**
   - Go to Render Dashboard
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

2. **Configure Backend**
   - **Name**: `signbridge-backend`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r python-backend/requirements.txt`
   - **Start Command**: `cd python-backend && python server.py`
   - **Environment Variables**:
     - `PORT`: `8000` (Render sets this automatically)
     - `PYTHON_VERSION`: `3.11.0`

3. **Health Check**
   - Path: `/health`

#### Frontend Service

1. **Create New Static Site**
   - Go to Render Dashboard
   - Click "New +" → "Static Site"
   - Connect your GitHub repository

2. **Configure Frontend**
   - **Name**: `signbridge-frontend`
   - **Build Command**: (leave empty)
   - **Publish Directory**: `.` (root)

3. **Environment Variables**
   - Add: `BACKEND_URL` = `https://signbridge-backend.onrender.com`
   - This will be used by the frontend to connect to backend

## Environment Variables

### Backend Service
- `PORT` - Automatically set by Render
- `PYTHON_VERSION` - Python version (3.11.0 recommended)

### Frontend Service
- `BACKEND_URL` - Backend service URL (optional, auto-detected)

## Model File Setup

Since model files are large and excluded from git:

### Option 1: Upload via Render Shell
1. Go to your backend service → Shell
2. Create models directory: `mkdir -p python-backend/models`
3. Upload model file using `wget` or `curl`

### Option 2: Use Render Disk
1. Enable persistent disk in backend service settings
2. Upload model file to disk
3. Update model path in code

### Option 3: Host Model Separately
1. Upload model to cloud storage (S3, etc.)
2. Download on startup in `server.py`

## Updating Frontend Backend URL

The frontend automatically detects the backend URL. If you need to set it manually:

1. In Render Dashboard → Frontend Service → Environment
2. Add: `BACKEND_URL` = `https://your-backend-url.onrender.com`
3. Update `js/recognition.js` to use: `window.BACKEND_URL`

## Troubleshooting

### Backend won't start
- Check build logs for dependency errors
- Ensure `requirements.txt` is correct
- Verify Python version matches

### Frontend can't connect to backend
- Check backend URL in frontend environment variables
- Verify CORS is enabled in backend (already configured)
- Check backend service is running

### Model not found
- Verify model file path
- Check file permissions
- Ensure model is in `python-backend/models/`

### Camera not working
- HTTPS is required for camera access
- Render provides HTTPS automatically
- Check browser console for permissions

## Cost Considerations

- **Free Tier**: 
  - Backend: Spins down after 15 minutes of inactivity
  - Frontend: Always available
- **Paid Tier**: 
  - Backend: Always running
  - Better performance

## URLs After Deployment

- Backend: `https://signbridge-backend.onrender.com`
- Frontend: `https://signbridge-frontend.onrender.com`
- API Docs: `https://signbridge-backend.onrender.com/docs`

## Notes

- Model files are excluded from git (too large)
- You'll need to upload the model separately
- Free tier backend may have cold starts
- Consider using Render Disk for persistent model storage

---

**Ready to deploy?** Push your code to GitHub and follow the steps above!

