# Running SignBridge Locally

## Quick Start

### Option 1: Automated Setup (Recommended)

1. **First time setup:**
   ```batch
   SETUP_LOCAL.bat
   ```
   This will:
   - Check Python installation
   - Create virtual environment
   - Install all dependencies

2. **Start the server:**
   ```batch
   START_SERVER.bat
   ```

### Option 2: Quick Start (If already set up)

```batch
START_SERVER_QUICK.bat
```

## Manual Setup

If you prefer to set up manually:

### Prerequisites

- **Python 3.11** or higher
  - Download from: https://www.python.org/downloads/
  - Make sure to check "Add Python to PATH" during installation

### Steps

1. **Open Command Prompt or PowerShell** in the project directory

2. **Navigate to python-backend:**
   ```batch
   cd python-backend
   ```

3. **Create virtual environment:**
   ```batch
   python -m venv venv
   ```

4. **Activate virtual environment:**
   ```batch
   venv\Scripts\activate
   ```

5. **Install dependencies:**
   ```batch
   pip install -r requirements.txt
   ```

6. **Start the server:**
   ```batch
   python server.py
   ```

## Accessing the Application

Once the server is running:

- **Main page:** http://localhost:8000
- **Recognition page:** http://localhost:8000/recognition.html
- **API docs:** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health

## Troubleshooting

### "Python is not recognized"

- Make sure Python is installed
- Add Python to your PATH environment variable
- Restart Command Prompt after installing Python

### "pip is not recognized"

- Make sure Python is installed correctly
- Try: `python -m pip install -r requirements.txt`

### Port 8000 already in use

- Stop any other application using port 8000
- Or change the port in `server.py`:
  ```python
  port = int(os.environ.get("PORT", 8001))  # Change to 8001
  ```

### Dependencies fail to install

- Make sure you have internet connection
- Try upgrading pip: `python -m pip install --upgrade pip`
- Check Python version: `python --version` (should be 3.11+)

### Virtual environment issues

- Delete the `venv` folder and run `SETUP_LOCAL.bat` again
- Make sure you're using the correct Python version

## Stopping the Server

Press `Ctrl+C` in the terminal where the server is running.

## Development Tips

- The server auto-reloads on code changes (if using uvicorn with `--reload`)
- Check server logs in the terminal for debugging
- API documentation is available at `/docs` endpoint

## What Gets Installed

The following packages are installed:

- **FastAPI** - Web framework
- **Uvicorn** - ASGI server
- **sign-language-translator[mediapipe]** - Sign language translation library
- **OpenCV** - Image processing
- **Pillow** - Image handling
- **NumPy** - Numerical computing
- **WebSockets** - Real-time communication

---

**Need help?** Check the main README.md or open an issue on GitHub.

