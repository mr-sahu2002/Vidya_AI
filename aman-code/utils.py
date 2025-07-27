import os
import tempfile
import uuid
import asyncio
import io
import gc
import time
import warnings
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union, Any
from pathlib import Path

# Suppress FastEmbed deprecation warnings
warnings.filterwarnings("ignore", message=".*deprecated.*", module="fastembed")
logging.getLogger("fastembed.embedding").setLevel(logging.ERROR)

from fastapi import (
    FastAPI, UploadFile, File, HTTPException,
    Depends, BackgroundTasks, Form
)
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, HTMLResponse
from fastapi.requests import Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from fastembed.embedding import TextEmbedding

import google.generativeai as genai

# Conditional imports
try:
    from google.cloud import texttospeech
    from google.api_core import exceptions as google_exceptions
    GOOGLE_CLOUD_TTS_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_TTS_AVAILABLE = False
    texttospeech = None
    google_exceptions = None

try:
    from langdetect import detect
    LANGUAGE_DETECTION_AVAILABLE = True
except ImportError:
    LANGUAGE_DETECTION_AVAILABLE = False
    print("Warning: langdetect not available. Install with: pip install langdetect")

load_dotenv()

# Constants
COLLECTION_NAME = "voice-rag-agent"
USE_GEMINI_TTS = os.getenv("USE_GEMINI_TTS", "false").lower() == "true"

# Environment configs
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# MULTILINGUAL: Fixed voice mapping per language
LANGUAGE_VOICES = {
    "auto": "en-IN-Neural2-A",         # Default voice for auto-detection
    "english": "en-IN-Neural2-A",      # Indian English, Female
    "hindi": "hi-IN-Neural2-A",        # Hindi, Female  
    "kannada": "kn-IN-Standard-A",     # Kannada, Female
    "tamil": "ta-IN-Standard-A",       # Tamil, Female
    "telugu": "te-IN-Standard-A",      # Telugu, Female
    "bengali": "bn-IN-Standard-A",     # Bengali, Female
    "gujarati": "gu-IN-Standard-A",    # Gujarati, Female
    "marathi": "mr-IN-Standard-A",     # Marathi, Female
    "malayalam": "ml-IN-Standard-A",   # Malayalam, Female
    "punjabi": "pa-IN-Standard-A",     # Punjabi, Female
    "urdu": "hi-IN-Standard-B"         # Use Hindi voice for Urdu
}

# Language code mapping for TTS API
LANGUAGE_CODES = {
    "auto": "en-IN",
    "english": "en-IN",
    "hindi": "hi-IN",
    "kannada": "kn-IN", 
    "tamil": "ta-IN",
    "telugu": "te-IN",
    "bengali": "bn-IN",
    "gujarati": "gu-IN",
    "marathi": "mr-IN",
    "malayalam": "ml-IN",
    "punjabi": "pa-IN",
    "urdu": "hi-IN"
}

# FastAPI app
app = FastAPI(
    title="Voice RAG Agent API",
    description="Voice-powered RAG with Google Gemini Pro and Multilingual TTS",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def json_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": f"Internal server error: {str(exc)}",
            "error": str(exc)
        },
    )

# ThreadPoolExecutor for blocking calls
executor = ThreadPoolExecutor(max_workers=4)

# In-memory audio cache
audio_cache = {}

# Service globals
qdrant_client: Optional[QdrantClient] = None
embedding_model: Optional[TextEmbedding] = None
gemini_model: Optional[genai.GenerativeModel] = None
tts_client: Optional[Any] = None

# UPDATED: Pydantic models for multilingual support
class QueryRequest(BaseModel):
    query: str
    language: Optional[str] = "auto"  # Language selection instead of voice
    prepare_audio: Optional[bool] = True

class QueryResponse(BaseModel):
    status: str
    query_id: str
    text_response: Optional[str] = None
    detected_language: Optional[str] = None
    selected_voice: Optional[str] = None
    sources: Optional[List[str]] = None
    audio_preparing: Optional[bool] = False
    audio_ready: Optional[bool] = False
    tts_error: Optional[str] = None
    error: Optional[str] = None

class AudioResponse(BaseModel):
    status: str
    audio_ready: bool
    audio_url: Optional[str] = None
    error: Optional[str] = None

class UploadResponse(BaseModel):
    status: str
    message: str
    file_name: Optional[str] = None
    document_count: Optional[int] = None
    error: Optional[str] = None

# MULTILINGUAL: Language detection and voice selection functions
def detect_response_language(text: str) -> str:
    """Detect the language of the response text"""
    if not LANGUAGE_DETECTION_AVAILABLE:
        return "english"
    
    try:
        import re
        # Clean text for better detection
        clean_text = re.sub(r'[^\w\s]', '', text[:500])  # Use first 500 chars
        
        detected = detect(clean_text)
        
        # Map langdetect codes to our language keys
        lang_mapping = {
            'en': 'english',
            'hi': 'hindi',
            'kn': 'kannada',
            'ta': 'tamil',
            'te': 'telugu',
            'bn': 'bengali',
            'gu': 'gujarati',
            'mr': 'marathi',
            'ml': 'malayalam',
            'pa': 'punjabi',
            'ur': 'urdu'
        }
        
        return lang_mapping.get(detected, 'english')
        
    except Exception as e:
        print(f"Language detection failed: {e}, defaulting to English")
        return 'english'

def get_voice_for_language(language: str, response_text: str = "") -> tuple:
    """Get appropriate voice and language code based on selected language or auto-detection"""
    
    if language == "auto" and response_text:
        # Auto-detect language from response
        detected_lang = detect_response_language(response_text)
        print(f"🌍 Auto-detected language: {detected_lang}")
        return LANGUAGE_VOICES[detected_lang], LANGUAGE_CODES[detected_lang], detected_lang
    elif language in LANGUAGE_VOICES:
        # Use user-selected language
        return LANGUAGE_VOICES[language], LANGUAGE_CODES[language], language
    else:
        # Fallback to English
        return LANGUAGE_VOICES["english"], LANGUAGE_CODES["english"], "english"

# ENHANCED: Service initialization with robust TTS error handling
def initialize_services() -> bool:
    global qdrant_client, embedding_model, gemini_model, tts_client

    try:
        # Qdrant Cloud configuration
        if not QDRANT_URL:
            print("Error: QDRANT_URL not provided")
            return False
        
        if not QDRANT_API_KEY:
            print("Error: QDRANT_API_KEY not provided - required for Qdrant Cloud")
            return False

        print(f"Connecting to Qdrant Cloud at: {QDRANT_URL}")
        
        qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=30
        )
        
        # Test connection
        try:
            collections = qdrant_client.get_collections()
            print(f"✅ Successfully connected to Qdrant Cloud. Existing collections: {len(collections.collections)}")
        except Exception as e:
            print(f"❌ Failed to connect to Qdrant Cloud: {e}")
            return False

        # Initialize embedding model
        embedding_model = TextEmbedding()
        test_embedding = list(embedding_model.embed(["test"]))[0]
        embedding_dim = len(test_embedding)
        print(f"✅ Embedding model initialized with dimension: {embedding_dim}")

        # Collection handling
        try:
            collection_info = qdrant_client.get_collection(COLLECTION_NAME)
            print(f"✅ Collection '{COLLECTION_NAME}' already exists with {collection_info.points_count} points")
        except Exception as get_error:
            print(f"📝 Collection '{COLLECTION_NAME}' not found or error retrieving info: {get_error}")
            try:
                print(f"Creating new collection '{COLLECTION_NAME}'...")
                qdrant_client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
                )
                print(f"✅ Collection '{COLLECTION_NAME}' created successfully")
            except Exception as create_error:
                if "already exists" in str(create_error).lower() or "409" in str(create_error):
                    print(f"✅ Collection '{COLLECTION_NAME}' already exists (confirmed via create attempt)")
                else:
                    print(f"❌ Failed to create collection: {create_error}")
                    raise create_error

        # Google Gemini Pro setup
        if not GOOGLE_API_KEY:
            raise ValueError("Google API key not provided")
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-2.5-pro")
        print("✅ Google Gemini Pro initialized")

        # ENHANCED: TTS setup with comprehensive error handling
        if not USE_GEMINI_TTS:
            if not GOOGLE_CLOUD_TTS_AVAILABLE:
                print("⚠️ Google Cloud TTS not available. Install: pip install google-cloud-texttospeech")
                print("📝 TTS will be disabled. Text responses will still work.")
                tts_client = None
            else:
                try:
                    # Check credentials file
                    if GOOGLE_APPLICATION_CREDENTIALS:
                        cred_path = GOOGLE_APPLICATION_CREDENTIALS
                        if not os.path.isabs(cred_path):
                            cred_path = os.path.join(os.path.dirname(__file__), cred_path)
                        
                        if os.path.exists(cred_path):
                            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path
                            print(f"📁 Using credentials file: {cred_path}")
                            
                            # Test TTS client initialization and API access
                            tts_client = texttospeech.TextToSpeechClient()
                            
                            # Test with a minimal synthesis request
                            test_input = texttospeech.SynthesisInput(text="Test")
                            test_voice = texttospeech.VoiceSelectionParams(
                                language_code="en-IN",
                                name="en-IN-Neural2-A"
                            )
                            test_config = texttospeech.AudioConfig(
                                audio_encoding=texttospeech.AudioEncoding.MP3
                            )
                            
                            try:
                                test_response = tts_client.synthesize_speech(
                                    input=test_input,
                                    voice=test_voice,
                                    audio_config=test_config
                                )
                                print(f"✅ Google Cloud TTS initialized and tested successfully ({len(test_response.audio_content)} bytes)")
                            except google_exceptions.PermissionDenied as permission_error:
                                print(f"❌ Google Cloud TTS API not enabled in your project")
                                print(f"📋 Error details: {permission_error}")
                                if "SERVICE_DISABLED" in str(permission_error):
                                    print(f"🔧 Solution: Enable the Cloud Text-to-Speech API at:")
                                    print(f"   https://console.developers.google.com/apis/api/texttospeech.googleapis.com/overview")
                                    print(f"📝 TTS will be disabled. Text responses will still work.")
                                tts_client = None
                            except Exception as api_error:
                                print(f"❌ Google Cloud TTS API test failed: {api_error}")
                                print(f"📝 TTS will be disabled. Text responses will still work.")
                                tts_client = None
                                
                        else:
                            print(f"❌ Google Cloud credentials file not found at: {cred_path}")
                            print(f"📝 TTS will be disabled. Text responses will still work.")
                            tts_client = None
                    else:
                        print("❌ GOOGLE_APPLICATION_CREDENTIALS not set in .env")
                        print(f"📝 TTS will be disabled. Text responses will still work.")
                        tts_client = None
                        
                except Exception as tts_init_error:
                    print(f"❌ Failed to initialize Google Cloud TTS: {tts_init_error}")
                    print(f"📝 TTS will be disabled. Text responses will still work.")
                    tts_client = None
        else:
            tts_client = None
            print("⚠️ Gemini TTS mode enabled (not yet available)")

        return True
    except Exception as e:
        print(f"❌ Service initialization error: {e}")
        return False

def get_services():
    if not all([qdrant_client, embedding_model, gemini_model]):
        raise HTTPException(
            status_code=503,
            detail="Services not initialized properly. Check configurations."
        )
    return {
        "qdrant_client": qdrant_client,
        "embedding_model": embedding_model,
        "gemini_model": gemini_model,
        "tts_client": tts_client
    }

# PDF processing
def process_pdf_file(file_content: bytes, filename: str) -> List:
    print(f"🔧 Starting PDF processing for: {filename}")
    tmp_file_path = None
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
            print(f"📁 Created temporary file: {tmp_file_path}")

        time.sleep(0.1)
        documents = []
        
        try:
            print(f"📚 Loading PDF with PyPDFLoader...")
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            print(f"📄 Successfully loaded {len(documents)} pages with PyPDFLoader")
        except Exception as pdf_error:
            print(f"❌ PyPDFLoader failed: {pdf_error}")
            
            try:
                print(f"🔄 Trying PyPDF2 as fallback...")
                import PyPDF2
                with open(tmp_file_path, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    documents = []
                    for i, page in enumerate(pdf_reader.pages):
                        text = page.extract_text()
                        if text.strip():
                            from langchain.schema import Document
                            documents.append(Document(
                                page_content=text,
                                metadata={"page": i + 1, "source": filename}
                            ))
                print(f"📄 PyPDF2 processing successful: {len(documents)} pages")
            except Exception as pypdf2_error:
                print(f"❌ All PDF processing methods failed")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Could not process PDF with any method. Last error: {str(pypdf2_error)}"
                )

        if not documents:
            raise HTTPException(status_code=500, detail="No content could be extracted from PDF")

        print(f"📝 Adding metadata to {len(documents)} documents...")
        for i, doc in enumerate(documents):
            doc.metadata.update({
                "source_type": "pdf",
                "file_name": filename,
                "timestamp": datetime.now().isoformat(),
                "document_id": f"{filename}_{i}",
                "total_documents": len(documents)
            })

        print(f"✂️ Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"📝 Created {len(chunks)} chunks from {len(documents)} documents")
        
        return chunks
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ PDF processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF processing error: {str(e)}")
    
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                gc.collect()
                time.sleep(0.2)
                
                for attempt in range(3):
                    try:
                        os.unlink(tmp_file_path)
                        print(f"🗑️ Successfully cleaned up temporary file (attempt {attempt + 1})")
                        break
                    except Exception as cleanup_error:
                        if attempt < 2:
                            print(f"⚠️ Cleanup attempt {attempt + 1} failed, retrying...")
                            time.sleep(0.5)
                        else:
                            print(f"⚠️ Warning: Could not delete temporary file after 3 attempts: {cleanup_error}")
                            
            except Exception as final_cleanup_error:
                print(f"⚠️ Final cleanup error: {final_cleanup_error}")

def store_embeddings_in_qdrant(
    client: QdrantClient,
    embedding_model: TextEmbedding,
    documents: List,
    collection_name: str
):
    print(f"💾 Storing {len(documents)} embeddings in Qdrant...")
    points = []
    
    for i, doc in enumerate(documents):
        try:
            embedding = list(embedding_model.embed([doc.page_content]))[0]
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "content": doc.page_content,
                        "chunk_id": i,
                        **doc.metadata
                    }
                )
            )
        except Exception as embed_error:
            print(f"⚠️ Failed to create embedding for chunk {i}: {embed_error}")
            continue
    
    if not points:
        raise HTTPException(status_code=500, detail="No valid embeddings could be created")
    
    try:
        client.upsert(collection_name=collection_name, points=points)
        print(f"✅ Successfully stored {len(points)} embeddings in Qdrant")
    except Exception as store_error:
        print(f"❌ Failed to store embeddings: {store_error}")
        raise HTTPException(status_code=500, detail=f"Failed to store embeddings: {str(store_error)}")

# MULTILINGUAL: Enhanced Gemini processing with language-aware prompts
async def process_query_with_gemini(
    query: str,
    context: str,
    gemini_model: genai.GenerativeModel,
    selected_language: str = "auto"
) -> str:
    
    # Language-specific prompts
    language_instructions = {
        "hindi": "कृपया उत्तर हिंदी में दें। स्पष्ट और सरल भाषा का उपयोग करें।",
        "kannada": "ದಯವಿಟ್ಟು ಉತ್ತರವನ್ನು ಕನ್ನಡದಲ್ಲಿ ನೀಡಿ। ಸರಳ ಮತ್ತು ಸ್ಪಷ್ಟ ಭಾಷೆಯನ್ನು ಬಳಸಿ।",
        "tamil": "தயவுசெய்து தமிழில் பதிலளிக்கவும். எளிய மற்றும் தெளிவான மொழியைப் பயன்படுத்தவும்।",
        "telugu": "దయచేసి తెలుగులో సమాధానం ఇవ్వండి। సరళమైన మరియు స్పష్టమైన భాషను ఉపయోగించండి।",
        "bengali": "অনুগ্রহ করে বাংলায় উত্তর দিন। সহজ এবং স্পষ্ট ভাষা ব্যবহার করুন।",
        "gujarati": "કૃપા કરીને ગુજરાતીમાં જવાબ આપો. સરળ અને સ્પષ્ટ ભાષાનો ઉપયોગ કરો.",
        "marathi": "कृपया मराठीत उत्तर द्या. सोपी आणि स्पष्ट भाषा वापरा.",
        "malayalam": "ദയവായി മലയാളത്തിൽ ഉത്തരം നൽകുക. ലളിതവും വ്യക്തവുമായ ഭാഷ ഉപയോഗിക്കുക।",
        "punjabi": "ਕਿਰਪਾ ਕਰਕੇ ਪੰਜਾਬੀ ਵਿੱਚ ਜਵਾਬ ਦਿਓ। ਸਰਲ ਅਤੇ ਸਪੱਸ਼ਟ ਭਾਸ਼ਾ ਦੀ ਵਰਤੋਂ ਕਰੋ।",
        "english": "Please respond in clear, simple English suitable for Indian users.",
        "auto": "Detect the language of the query and respond in the same language. If uncertain, respond in English."
    }
    
    lang_instruction = language_instructions.get(selected_language, language_instructions["english"])
    
    prompt = f"""You are a helpful multilingual documentation assistant for Indian users. Your task is to:

1. Analyze the provided documentation content carefully
2. Answer the user's question clearly and concisely
3. {lang_instruction}
4. Include relevant examples when available
5. Cite the source files when referencing specific content
6. Keep responses natural and conversational
7. Format your response in a way that's easy to speak out loud
8. Use simple sentences that work well with text-to-speech

Documentation Context:
{context}

User Question: {query}

Please provide a clear, helpful answer following the language instruction above."""

    try:
        response = await asyncio.to_thread(gemini_model.generate_content, prompt)
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

# ENHANCED: TTS generation with comprehensive error handling
def generate_speech_with_google_cloud_tts(
    text: str,
    tts_client,
    voice_name: str,
    language_code: str
) -> bytes:
    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code, 
            name=voice_name
        )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        
        response = tts_client.synthesize_speech(
            input=synthesis_input, 
            voice=voice, 
            audio_config=audio_config
        )
        
        return response.audio_content
        
    except google_exceptions.PermissionDenied as permission_error:
        if "SERVICE_DISABLED" in str(permission_error):
            raise Exception("Google Cloud Text-to-Speech API is not enabled. Please enable it in Google Cloud Console.")
        else:
            raise Exception(f"Permission denied for TTS API. Check your credentials and project permissions.")
    except Exception as e:
        raise Exception(f"Google Cloud TTS error: {str(e)}")

async def generate_speech_with_gemini_tts(text: str, voice: str) -> bytes:
    raise HTTPException(
        status_code=501, 
        detail="Gemini TTS API is not yet available. Please set USE_GEMINI_TTS=false to use Google Cloud TTS"
    )

async def generate_speech(text: str, voice_name: str, language_code: str, tts_client) -> bytes:
    if USE_GEMINI_TTS:
        return await generate_speech_with_gemini_tts(text, voice_name)
    else:
        if not tts_client:
            raise Exception("TTS client not initialized - check Google Cloud credentials and API enablement")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, generate_speech_with_google_cloud_tts, text, tts_client, voice_name, language_code)

# ENHANCED: Background audio preparation with detailed error handling
async def prepare_audio_in_background(
    query_id: str,
    text_response: str,
    selected_language: str,
    tts_client,
):
    try:
        print(f"🎵 Starting audio preparation for query: {query_id}")
        print(f"📝 Text length: {len(text_response)} characters")
        print(f"🌍 Selected language: {selected_language}")
        print(f"🤖 TTS client available: {tts_client is not None}")
        
        audio_cache[query_id] = {
            "status": "preparing",
            "audio_content": None,
            "language": selected_language,
            "timestamp": datetime.now().isoformat(),
        }

        # ENHANCED: Check if TTS client is available with detailed logging
        if not tts_client:
            error_msg = "TTS client not initialized - Google Cloud Text-to-Speech API may not be enabled"
            print(f"❌ {error_msg}")
            raise Exception(error_msg)

        # Get appropriate voice and language code
        voice_name, language_code, detected_lang = get_voice_for_language(selected_language, text_response)
        
        print(f"🎤 Selected voice: {voice_name}")
        print(f"🗣️ Language code: {language_code}")
        print(f"🔍 Detected language: {detected_lang}")
        
        # Generate speech with enhanced error handling
        try:
            audio_content = await generate_speech(text_response, voice_name, language_code, tts_client)
            print(f"✅ Audio generated successfully: {len(audio_content)} bytes")
        except Exception as speech_error:
            error_msg = str(speech_error)
            print(f"❌ Speech generation failed: {error_msg}")
            
            # Provide specific error messages for common issues
            if "SERVICE_DISABLED" in error_msg or "not enabled" in error_msg:
                error_msg = "Google Cloud Text-to-Speech API is not enabled. Please visit Google Cloud Console to enable it."
            elif "PERMISSION_DENIED" in error_msg:
                error_msg = "Permission denied for TTS API. Check your Google Cloud credentials and project permissions."
            elif "credentials" in error_msg.lower():
                error_msg = "TTS credentials issue. Check your Google Cloud service account setup."
            
            raise Exception(error_msg)

        audio_cache[query_id] = {
            "status": "ready",
            "audio_content": audio_content,
            "language": selected_language,
            "voice_used": voice_name,
            "detected_language": detected_lang,
            "timestamp": datetime.now().isoformat(),
        }
        
        print(f"🎉 Audio preparation completed successfully for {query_id}")
        
    except Exception as e:
        error_msg = str(e) if str(e) else "Unknown TTS error occurred"
        print(f"❌ Audio preparation failed for {query_id}: {error_msg}")
        print(f"📍 Error type: {type(e).__name__}")
        
        # ENHANCED: Capture full error details for debugging
        import traceback
        traceback.print_exc()
        
        audio_cache[query_id] = {
            "status": "error",
            "audio_content": None,
            "language": selected_language,
            "error": error_msg,
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat(),
        }

# Routes


# @app.get("/api/health")
# async def health_check():
#     try:
#         qdrant_status = False
#         qdrant_info = {}
#         if qdrant_client:
#             try:
#                 collections = qdrant_client.get_collections()
#                 try:
#                     collection_info = qdrant_client.get_collection(COLLECTION_NAME)
#                     qdrant_status = True
#                     qdrant_info = {
#                         "total_collections": len(collections.collections),
#                         "points_count": collection_info.points_count,
#                         "collection_status": "accessible"
#                     }
#                 except Exception as collection_error:
#                     qdrant_status = True
#                     qdrant_info = {
#                         "total_collections": len(collections.collections),
#                         "points_count": "unknown (pydantic error)",
#                         "collection_status": "exists_but_info_unavailable"
#                     }
#             except Exception as e:
#                 qdrant_info = {"error": str(e)}
        
#         # Enhanced TTS status checking
#         tts_status = False
#         tts_info = {}
#         if tts_client:
#             tts_status = True
#             tts_info = {"status": "initialized", "client_available": True}
#         else:
#             tts_info = {"status": "disabled", "reason": "API not enabled or credentials missing"}
        
#         services_status = {
#             "qdrant": qdrant_status,
#             "embedding_model": embedding_model is not None,
#             "gemini": gemini_model is not None,
#             "tts": tts_status,
#             "google_cloud_tts_available": GOOGLE_CLOUD_TTS_AVAILABLE,
#             "language_detection_available": LANGUAGE_DETECTION_AVAILABLE
#         }
        
#         all_healthy = all([services_status["qdrant"], services_status["embedding_model"], services_status["gemini"]])
        
#         return {
#             "status": "healthy" if all_healthy else "degraded",
#             "services": services_status,
#             "qdrant_info": qdrant_info,
#             "tts_info": tts_info,
#             "audio_cache_size": len(audio_cache),
#             "supported_languages": list(LANGUAGE_VOICES.keys()),
#             "use_gemini_tts": USE_GEMINI_TTS
#         }
#     except Exception as e:
#         return {
#             "status": "error",
#             "error": str(e),
#             "services": {"all": False}
#         }

@app.post("/api/upload-pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...), services=Depends(get_services)):
    print(f"🔄 Starting PDF upload for: {file.filename}")
    try:
        if not file.filename.lower().endswith(".pdf"):
            print(f"❌ Invalid file type: {file.filename}")
            return UploadResponse(
                status="error",
                message="Only PDF files are allowed",
                error="Invalid file type"
            )

        content = await file.read()
        file_size_mb = len(content) / (1024 * 1024)
        print(f"📊 File size: {file_size_mb:.2f} MB")
        
        if file_size_mb > 50:
            return UploadResponse(
                status="error",
                message="File too large. Maximum size is 50MB.",
                error="File size limit exceeded"
            )

        print(f"🔧 Processing PDF...")
        documents = process_pdf_file(content, file.filename)
        print(f"📑 Extracted {len(documents)} document chunks")

        if not documents:
            print(f"❌ No content extracted from PDF")
            return UploadResponse(
                status="error",
                message="No content could be extracted from the PDF",
                error="Empty PDF or unsupported format"
            )

        print(f"💾 Storing {len(documents)} chunks in Qdrant...")
        store_embeddings_in_qdrant(services["qdrant_client"], services["embedding_model"], documents, COLLECTION_NAME)
        
        print(f"✅ PDF upload completed successfully!")
        return UploadResponse(
            status="success",
            message=f"Processed and stored {len(documents)} document chunks",
            file_name=file.filename,
            document_count=len(documents),
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ PDF Upload Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return UploadResponse(
            status="error",
            message="Failed to process PDF",
            error=str(e)
        )

# ENHANCED: Query endpoint with TTS error handling
@app.post("/api/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest, background_tasks: BackgroundTasks, services=Depends(get_services)
):
    query_id = str(uuid.uuid4())
    try:
        print(f"🔍 Processing query in {request.language} language: {request.query}")
        
        # Embedding and search
        query_embedding = list(services["embedding_model"].embed([request.query]))[0]
        print(f"📊 Generated embedding with dimension: {len(query_embedding)}")
        
        # Search using newer Qdrant client API
        try:
            search_response = services["qdrant_client"].search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding.tolist(),
                limit=5,
                with_payload=True
            )
            print(f"🔎 Search completed, found {len(search_response)} results")
        except AttributeError:
            # Fallback for older versions
            print("⚠️ Trying fallback method for older Qdrant client...")
            search_response = services["qdrant_client"].query_points(
                collection_name=COLLECTION_NAME,
                query=query_embedding.tolist(),
                limit=5,
                with_payload=True
            )
            search_response = search_response.points if hasattr(search_response, "points") else search_response
        
        results = search_response if isinstance(search_response, list) else []

        if not results:
            print("❌ No relevant documents found")
            return QueryResponse(
                status="error", query_id=query_id, error="No relevant documents found in the database"
            )

        print(f"📝 Building context from {len(results)} search results...")
        context = "Based on the following documentation:\n\n"
        sources = []
        for r in results:
            payload = r.payload
            if not payload:
                continue
            content = payload.get("content", "")
            source = payload.get("file_name", "Unknown Source")
            context += f"From {source}:\n{content}\n\n"
            if source not in sources:
                sources.append(source)
        context += f"\nUser Question: {request.query}\n\n"

        print(f"🤖 Generating response with Gemini for {request.language} language...")
        text_response = await process_query_with_gemini(
            request.query, 
            context, 
            services["gemini_model"],
            request.language
        )

        # Get voice info for response
        voice_name, language_code, detected_lang = get_voice_for_language(request.language, text_response)
        
        audio_preparing = False
        tts_error = None
        
        if request.prepare_audio:
            if services.get("tts_client"):
                print(f"🎵 Preparing audio in background with voice: {voice_name}")
                background_tasks.add_task(
                    prepare_audio_in_background, 
                    query_id, 
                    text_response, 
                    request.language, 
                    services.get("tts_client")
                )
                audio_preparing = True
            else:
                tts_error = "TTS service not available - Google Cloud Text-to-Speech API may not be enabled"
                print(f"⚠️ {tts_error}")

        print(f"✅ Query processed successfully!")
        return QueryResponse(
            status="success",
            query_id=query_id,
            text_response=text_response,
            detected_language=detected_lang if request.language == "auto" else request.language,
            selected_voice=voice_name,
            sources=sources,
            audio_preparing=audio_preparing,
            audio_ready=False,
            tts_error=tts_error
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Query processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return QueryResponse(status="error", query_id=query_id, error=f"Query processing error: {str(e)}")

# MULTILINGUAL: Updated languages endpoint instead of voices
@app.get("/api/languages")
async def get_supported_languages():
    languages = {
        "auto": {
            "name": "Auto-Detect",
            "native_name": "🤖 Auto-Detect",
            "code": "auto",
            "voice": LANGUAGE_VOICES["auto"]
        },
        "english": {
            "name": "English (India)",
            "native_name": "🇮🇳 English",
            "code": "en-IN",
            "voice": LANGUAGE_VOICES["english"]
        },
        "hindi": {
            "name": "Hindi",
            "native_name": "हिंदी",
            "code": "hi-IN",
            "voice": LANGUAGE_VOICES["hindi"]
        },
        "kannada": {
            "name": "Kannada",
            "native_name": "ಕನ್ನಡ",
            "code": "kn-IN",
            "voice": LANGUAGE_VOICES["kannada"]
        },
        "tamil": {
            "name": "Tamil",
            "native_name": "தமிழ்",
            "code": "ta-IN",
            "voice": LANGUAGE_VOICES["tamil"]
        },
        "telugu": {
            "name": "Telugu",
            "native_name": "తెలుగు",
            "code": "te-IN",
            "voice": LANGUAGE_VOICES["telugu"]
        },
        "bengali": {
            "name": "Bengali",
            "native_name": "বাংলা",
            "code": "bn-IN",
            "voice": LANGUAGE_VOICES["bengali"]
        },
        "gujarati": {
            "name": "Gujarati",
            "native_name": "ગુજરાતી",
            "code": "gu-IN",
            "voice": LANGUAGE_VOICES["gujarati"]
        },
        "marathi": {
            "name": "Marathi",
            "native_name": "मराठी",
            "code": "mr-IN",
            "voice": LANGUAGE_VOICES["marathi"]
        },
        "malayalam": {
            "name": "Malayalam",
            "native_name": "മലയാളം",
            "code": "ml-IN",
            "voice": LANGUAGE_VOICES["malayalam"]
        },
        "punjabi": {
            "name": "Punjabi",
            "native_name": "ਪੰਜਾਬੀ",
            "code": "pa-IN",
            "voice": LANGUAGE_VOICES["punjabi"]
        }
    }
    return {"languages": languages}

@app.get("/api/audio-status/{query_id}", response_model=AudioResponse)
async def audio_status(query_id: str):
    if query_id not in audio_cache:
        return AudioResponse(status="error", audio_ready=False, error="Query ID not found")

    audio_info = audio_cache[query_id]
    if audio_info["status"] == "ready":
        return AudioResponse(status="success", audio_ready=True, audio_url=f"/api/audio/{query_id}")
    elif audio_info["status"] == "preparing":
        return AudioResponse(status="preparing", audio_ready=False)
    else:
        error_msg = audio_info.get("error", "Unknown error")
        return AudioResponse(status="error", audio_ready=False, error=error_msg)

@app.get("/api/audio/{query_id}")
async def get_audio(query_id: str):
    if query_id not in audio_cache:
        raise HTTPException(status_code=404, detail="Query ID not found")

    audio_info = audio_cache[query_id]

    if audio_info["status"] != "ready" or not audio_info["audio_content"]:
        error_msg = audio_info.get("error", "Audio not ready or generation failed")
        raise HTTPException(status_code=400, detail=error_msg)

    return StreamingResponse(
        io.BytesIO(audio_info["audio_content"]),
        media_type="audio/mpeg",
        headers={"Content-Disposition": f"attachment; filename=response_{audio_info['language']}.mp3"},
    )

@app.delete("/api/documents")
async def clear_documents(services=Depends(get_services)):
    try:
        services["qdrant_client"].delete_collection(COLLECTION_NAME)
        test_embedding = list(services["embedding_model"].embed(["test"]))[0]
        embedding_dim = len(test_embedding)
        services["qdrant_client"].create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )
        return {"status": "success", "message": "All documents cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing documents: {str(e)}")

@app.delete("/api/audio-cache")
async def clear_audio_cache():
    count = len(audio_cache)
    audio_cache.clear()
    return {"status": "success", "message": f"Cleared {count} audio cache entries"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
