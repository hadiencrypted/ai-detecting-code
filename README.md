This project runs a Flask-based image analysis API.

Quick start (Windows double-click):
- Install Python 3.11 and required packages:

```powershell
python -m pip install -r requirements.txt
```

- Double-click `run_app.bat` to start the server.

Quick start (Docker):

Windows (PowerShell):

```powershell
./run_docker.bat
```

Linux / macOS:

```bash
./run_docker.sh
```

Notes:
- The image uses an official TensorFlow base so model inference works out-of-the-box inside the container. The built image can be large (hundreds of MBs).
- Include `ai_model.h5` in the project root before building if you want the container to ship with the local model.

Model download helper
 - If you don't want to commit `ai_model.h5` to the repo, use the included downloader before running the app or building the image:

```bash
# Download to project root
python download_model.py "https://example.com/path/to/ai_model.h5" --sha256 "<optional-sha256>"
```

 - You can also instruct Docker to download the model at build time by passing `MODEL_URL` as a build-arg:

```bash
docker build --build-arg MODEL_URL="https://example.com/path/to/ai_model.h5" -t ai-detector:latest .
```

 - If you prefer a one-liner for POSIX shells, use `./download_model.sh <MODEL_URL> [sha256]`.

Notes:
- The app will attempt to load `ai_model.h5` if present.
- For stability on Windows the app defers TensorFlow import until startup and disables Flask's reloader.
- If you want "single-click" for other OSes, create a small launcher script (shell script or .desktop file).
