# How to Upload Model to Render

The model file is too large for Git, so you need to upload it directly to Render after deployment.

## Method 1: Using Render Shell (Recommended)

### Step 1: Deploy Your Service
First, deploy your service on Render (even without the model - it will use simulated predictions).

### Step 2: Open Render Shell
1. Go to your Render Dashboard
2. Click on your `signbridge` service
3. Go to **"Shell"** tab
4. Click **"Open Shell"**

### Step 3: Create Models Directory
```bash
mkdir -p python-backend/models
cd python-backend/models
```

### Step 4: Upload Model File

**Option A: Using wget (if model is hosted online)**
```bash
wget -O sign_language_cnn_model.h5 <your-model-url>
```

**Option B: Using curl**
```bash
curl -o sign_language_cnn_model.h5 <your-model-url>
```

**Option C: Using Render's File Manager**
- Some Render plans have a file manager
- Navigate to `python-backend/models/`
- Upload `sign_language_cnn_model.h5`

### Step 5: Verify Upload
```bash
ls -lh sign_language_cnn_model.h5
# Should show file size (usually 10-50 MB)
```

### Step 6: Restart Service
- Go to Render Dashboard → Your Service
- Click **"Manual Deploy"** → **"Deploy latest commit"**
- Or the service will auto-restart and detect the model

## Method 2: Using Render Disk (Persistent Storage)

### Step 1: Enable Persistent Disk
1. Go to your service → **Settings**
2. Scroll to **"Disk"** section
3. Enable **Persistent Disk**
4. Set size (1 GB is usually enough)

### Step 2: Upload via Shell
```bash
# Mount point is usually /opt/render/project/src
cd /opt/render/project/src/python-backend/models
# Upload your model file here
```

### Step 3: Restart Service
The model will persist across deployments.

## Method 3: Download Model on Startup

Modify `server.py` to download model from cloud storage:

```python
def download_model_from_url():
    """Download model from URL if not exists"""
    model_url = os.environ.get('MODEL_URL', '')
    if not model_url:
        return False
    
    model_path = 'python-backend/models/sign_language_cnn_model.h5'
    if os.path.exists(model_path):
        return True
    
    try:
        import urllib.request
        print(f"Downloading model from {model_url}...")
        urllib.request.urlretrieve(model_url, model_path)
        print("Model downloaded successfully!")
        return True
    except Exception as e:
        print(f"Failed to download model: {e}")
        return False
```

Then set environment variable in Render:
- `MODEL_URL` = `https://your-storage.com/model.h5`

## Method 4: Include Model in Git (Not Recommended)

If model is small enough (< 100 MB):
1. Remove from `.gitignore`
2. Commit model file
3. Push to GitHub
4. Render will include it in deployment

**Warning**: This increases repo size significantly.

## Quick Check: Is Model Loading?

After uploading, check logs:
1. Go to Render Dashboard → Your Service → **Logs**
2. Look for:
   - `✅ Model loaded successfully!` = Success
   - `⚠️ No trained model found` = Model not found
   - `❌ Error loading model` = Model file corrupted

## Troubleshooting

### Model Still Not Found
- Check file path: `python-backend/models/sign_language_cnn_model.h5`
- Verify file exists: `ls -la python-backend/models/`
- Check file permissions: `chmod 644 python-backend/models/sign_language_cnn_model.h5`

### Model File Too Large
- Render free tier has limits
- Consider using model compression
- Or use external storage (S3, etc.) and download on startup

### Model Loads But Predictions Fail
- Check model format (should be .h5)
- Verify model architecture matches code
- Check TensorFlow version compatibility

## Expected File Structure on Render

```
/opt/render/project/src/
├── python-backend/
│   ├── models/
│   │   └── sign_language_cnn_model.h5  ← Model file here
│   ├── server.py
│   └── requirements.txt
├── index.html
├── recognition.html
└── ...
```

---

**After uploading, restart your service and check the logs!** 🚀

