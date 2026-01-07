@echo off
REM Build and run Docker container (requires Docker Desktop installed)
docker build -t ai-detector:latest .
docker run -p 8000:8000 ai-detector:latest
