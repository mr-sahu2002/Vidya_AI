#!/usr/bin/env python3
"""
Voice RAG Agent Launcher
Run this script to start both the backend API and serve the frontend
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import fastapi
        import uvicorn
        print("✅ FastAPI and Uvicorn are installed")
        return True
    except ImportError:
        print("❌ Missing required packages. Please run:")
        print("pip install -r backend/requirements.txt")
        return False

def main():
    print("🚀 Starting Voice RAG Agent...")
    
    if not check_requirements():
        sys.exit(1)
    
    # Change to project root directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Check if directories exist
    if not (project_root / "backend").exists():
        print("❌ Backend directory not found!")
        sys.exit(1)
    
    if not (project_root / "frontend").exists():
        print("⚠️  Frontend directory not found, creating it...")
        (project_root / "frontend").mkdir(exist_ok=True)
    
    # Start the FastAPI server
    try:
        print("🔧 Starting FastAPI server...")
        print("📱 Frontend will be available at: http://localhost:8000")
        print("📚 API documentation at: http://localhost:8000/docs")
        print("❌ Press Ctrl+C to stop the server")
        print("-" * 50)
        
        # Run uvicorn from the backend directory
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "backend.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down server...")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    main()
