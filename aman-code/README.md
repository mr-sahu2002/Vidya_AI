# README.md


# Voice RAG Agent Backend with Google Gemini & TTS

This repository contains a FastAPI backend API for a Retrieval-Augmented Generation (RAG) application that allows:

- Uploading PDF documents which are vectorized and stored in Qdrant
- Querying the document content with Google Gemini Pro to generate concise answers
- Generating text-to-speech (TTS) audio responses either using Google Cloud Text-to-Speech or Google Gemini TTS API **(option configurable)**.
- Serving audio asynchronously with background processing and polling endpoints

There is also a simple static HTML frontend to interact with the API.

---

## Features

- Qdrant vector database for document storage and similarity search
- Integration with Google Gemini Pro for text generation
- Support for Google Cloud TTS or Google Gemini TTS API for audio generation
- Asynchronous processing with background audio generation
- REST API endpoints to control query, audio status, audio retrieval, document upload, and management
- Client-side polling mechanism for audio readiness
- Simple static frontend that works with these APIs

---

## Prerequisites

- Python 3.8 or higher
- Qdrant server (cloud or self-hosted) with URL and API key
- Google Gemini Pro API key (Google AI Studio)
- Google Cloud project for Text-to-Speech API (if using Google Cloud TTS)
- Google Cloud service account JSON key file for TTS (if using Google Cloud TTS)

---

## Installation

1. **Clone the repository**
```
git clone 
cd 
```

2. **Create and activate a virtual environment**
```
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows
```

3. **Install dependencies**
```
pip install -r requirements.txt
```

*(If you do not have a `requirements.txt`, install the following manually:)*

```
pip install fastapi uvicorn python-multipart \
fastembed qdrant-client langchain langchain-community \
google-generativeai google-cloud-texttospeech python-dotenv
```

---

## Configuration

1. **Create a `.env` file** in the project root directory using one of the templates above.

Example `.env` (Cloud TTS mode):

```
QDRANT_URL=https://your-qdrant-instance.com
QDRANT_API_KEY=your-qdrant-api-key

GOOGLE_API_KEY=your-google-api-key-for-gemini

GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/google-service-account.json

USE_GEMINI_TTS=false
```

Or for Gemini TTS mode:

```
QDRANT_URL=https://your-qdrant-instance.com
QDRANT_API_KEY=your-qdrant-api-key

GOOGLE_API_KEY=your-google-api-key-for-gemini

GOOGLE_APPLICATION_CREDENTIALS=

USE_GEMINI_TTS=true
```

2. **Ensure Google Cloud Text-to-Speech API is enabled** in your Google Cloud project if using Cloud TTS.

---

## Running the Backend API

```
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- The API will be available at `http://localhost:8000`.
- Interactive API docs available at `http://localhost:8000/docs`.

---

## Using the Static Frontend

1. Save the provided `index.html` (from the previous messages) in the project folder or a static server.

2. Serve it via:

- A simple HTTP server (if testing locally), from the directory containing `index.html`:

```
python3 -m http.server 8080
```

- Or, if you want to serve it through FastAPI, add this snippet inside `main.py`:

```
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="path_to_static_files", html=True), name="static")
```

Then access `http://localhost:8000/`.

---

## API Endpoints Overview

| Endpoint               | Method | Description                          |
|------------------------|--------|----------------------------------|
| `/upload-pdf`          | POST   | Upload PDF document               |
| `/query`               | POST   | Ask question, get text response, initiate TTS background generation |
| `/audio-status/{id}`   | GET    | Check whether TTS audio is ready  |
| `/audio/{id}`          | GET    | Retrieve the generated TTS audio  |
| `/voices`              | GET    | List available voices              |
| `/documents`           | DELETE | Clear all uploaded documents      |
| `/audio-cache`         | DELETE | Clear stored audio cache           |
| `/health`              | GET    | Health status                     |

---

## Example Usage (cURL)

### Upload PDF

```
curl -X POST "http://localhost:8000/upload-pdf" \
  -F "file=@your-document.pdf"
```

### Ask a Question (with TTS preparation)

```
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"How do I authenticate API requests?", "voice":"en-US-Journey-F", "prepare_audio": true}'
```

### Check Audio Status

```
curl "http://localhost:8000/audio-status/"
```

### Download Audio

```
curl "http://localhost:8000/audio/" --output response.mp3
```

---

## Notes

- When sending a query, you receive a `query_id`.
- You can poll `/audio-status/` to check if audio is ready.
- When ready, fetch audio from `/audio/`.
- The frontend handles these flows with UI controls.

---

## Troubleshooting & Tips

- Make sure your `.env` is loaded, or set environment variables before launch.
- Confirm that the Google API key and credentials have the right permissions.
- Ensure Qdrant service is accessible from the backend host.
- If using Gemini TTS mode (`USE_GEMINI_TTS=true`), backend requires less setup but ensure your Google API key supports Gemini TTS.
- The `audio_cache` stores all generated audio in memory; for production consider persistent storage or Redis caching.

---

## Contact / Support

For issues or customization help, please open an issue or contact the maintainer.

---

Enjoy your Voice RAG Agent powered by cutting-edge Google Gemini generative AI & Text-to-Speech!

```

If you want, I can also help you create a Dockerfile or deployment instructions for cloud environments. Just let me know!