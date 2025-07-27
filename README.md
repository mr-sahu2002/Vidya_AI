
# Vidya AI-Powered Educational Platform

A modular edtech platform that generates, personalizes, and tracks hyper-localized educational content for students and teachers in a rural and urban area.

---

## Table of Contents
- [Getting Started](#getting-started)
- [Core Setup & Example Commands](#core-setup--example-commands)
  - [Qdrant (Vector Database)](#qdrant-vector-database)
  - [Node.js Application](#nodejs-application)
  - [Python API (using Uvicorn)](#python-api-using-uvicorn)
  - [Imagen 4](#)
- [Firebase Authentication](#firebase-authentication)
- [Features & Implementation](#features--implementation)
- [Project Structure](#project-structure)
- [Running the Full Stack: Example Sequence](#running-the-full-stack-example-sequence)
- [Important Notes](#important-notes)

---

## Getting Started

**Prerequisites:**
- Docker
- Node.js and npm
- Python 3.x and pip
- Firebase account `.json` file with authentication & database access
- GCP service account

---

## Core Setup & Example Commands

### Qdrant (Vector Database)

```bash
docker run -p 6333:6333 -p 6334:6334 \
-v $(pwd)/qdrant_storage:/qdrant/storage:z \
qdrant/qdrant
```

- REST API: `http://localhost:6333`  
- GRPC API: `http://localhost:6334`  
- Data is stored on your machine in `./qdrant_storage`.

---

### Node.js Application

```bash
npm i         # Install dependencies
npm start     # Start the Node.js server
```

---

### Python API (using Uvicorn)

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

---

## Firebase Authentication

1. Obtain your **Firebase service account `.json` authentication file**.
2. Place this file securely in your backend directory.
3. Ensure your Firebase project has both authentication and Firestore access enabled.

---

## Features & Implementation

- **Hyper-Local Content Generator**  
  Creates materials tailored to your specific community, classroom, or school—ensuring relevance for local needs.

- **Instant Student Q&A Simplifier**  
  Uses AI to rephrase or answer student questions at an age-appropriate level.

- **White Board Animation**  
  Transforms concepts into animated video explainers (e.g., using Imagen or similar models).

- **Persona Profiling (with Firestore)**  
  Interactively builds individual student profiles through targeted questions, storing data in Firestore for adaptive learning.

- **On-the-Fly Educational Game Generator**  
  Combines persona and RAG (Retrieval-Augmented Generation) to create personalized educational game experiences that boost subject understanding by using llm js output to generate educational concept simulator.

- **AI-Powered Weekly Lesson Planner**  
  Automatically constructs optimized lesson plans based on curriculum, class performance, and student needs.

- **AI-Driven Recommendations**  
  Suggests resources and next steps based on analytics from student profiles and classroom progress.

- **Unified Teacher Dashboard**  
  Tracks each student’s learning persona, showing analytics, assessment, and progress in a unified interface.

- **Low-Tech Compatibility**  
  using Twilio we can eanable LLM chat through SMS  

---

## Project Structure

```
/qdrant_storage         # Local persistent storage for Qdrant
/frontend               # Node.js (likely React/Next.js)
/backend                # Python FastAPI server (main:app)
firebase_config.json    # Firebase service account (keep secret!)
```

- Firestore (Firebase) integration for persona and authentication  
- Docker for virtualization and data management  

---

## Running the Full Stack: Example Sequence

```bash
# 1. Start Qdrant for vector search
docker run -p 6333:6333 -p 6334:6334 \
-v $(pwd)/qdrant_storage:/qdrant/storage:z \
qdrant/qdrant

# 2. Install backend dependencies
pip install -r requirements.txt

# 3. Run Python backend
uvicorn main:app --reload

# 4. Install frontend dependencies
npm i

# 5. Start frontend
npm start

# 6. Place your Firebase service account .json file in the backend
```

---

## Important Notes

- 🔒 **Keep your Firebase authentication files secure**—do not commit them to source control.
- 🔁 Change all default passwords and ports before production use.
- 💾 Make sure data directories are persisted with Docker volumes.
- 📖 For more details, consult `/docs` or configuration files in the repo.

Thank you!
