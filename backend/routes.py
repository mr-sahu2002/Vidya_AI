# Imports
from fastapi import APIRouter, UploadFile, HTTPException, Form, Depends, BackgroundTasks, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse, JSONResponse

from typing import Optional, List, Dict, Any
import tempfile
import os
import json
import uuid
import string
import random
import asyncio
import io
import gc
import time
import warnings
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import firebase_admin
from firebase_admin import credentials, firestore, auth
from pydantic import BaseModel
import logging

from starlette.routing import Router

# Suppress FastEmbed deprecation warnings
warnings.filterwarnings("ignore", message=".*deprecated.*", module="fastembed")
logging.getLogger("fastembed.embedding").setLevel(logging.ERROR)

# Additional imports for utils functionality
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from fastembed.embedding import TextEmbedding
import google.generativeai as genai

# Import config
from config import QDRANT_CONFIG


# Gemini client for LLM
try:
    from google import genai as google_genai
    from google.genai import types
    gemini_client = google_genai.Client(vertexai=True,
    project="vidya-ai-cdb33",
    location="us-east4" )
    GEMINI_AVAILABLE = True
except ImportError:
    print("Warning: google-genai not available. Install with: pip install google-genai")
    gemini_client = None
    GEMINI_AVAILABLE = False

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

from rag.extractor import extract_content_universal
from rag.embedder import get_jina_embeddings, get_single_jina_embedding
from rag.qdrant_service import (
    create_virtual_class,
    store_chunks_in_class,
    search_class_content,
    get_class_stats
)
# End of Imports

# Load environment variables
load_dotenv()

# Environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS","false").lower() == "true"

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate("../vidyaai-1dcfa-firebase-adminsdk-fbsvc-513f50d8d1.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()
security = HTTPBearer(auto_error=False)


# Constants from utils.py
COLLECTION_NAME = "voice-rag-agent"
USE_GEMINI_TTS = os.getenv("USE_GEMINI_TTS", "false").lower() == "true"

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

# ThreadPoolExecutor for blocking calls
executor = ThreadPoolExecutor(max_workers=4)

# In-memory audio cache
audio_cache = {}

# Service globals
qdrant_client: Optional[QdrantClient] = None
embedding_model: Optional[TextEmbedding] = None
gemini_model: Optional[genai.GenerativeModel] = None
tts_client: Optional[Any] = None

router = APIRouter()



# =====================
# PYDANTIC MODELS
# =====================
class UserCreate(BaseModel):
    email: str
    name: str
    role: str  # "teacher" or "student"

class ClassCreate(BaseModel):
    className: str
    subject: str
    description: Optional[str] = ""
    settings: Optional[Dict] = {}

class JoinClassRequest(BaseModel):
    joinCode: str

class PersonaResponse(BaseModel):
    responses: List[Dict[str, Any]]  # List of {questionId: str, response: str}

class TopicRequest(BaseModel):
    class_description: Optional[str] = ""
    difficulty_level: Optional[str] = "medium"  # easy, medium, hard
    grade_level: Optional[str] = "unknown"

class TeachingPlanRequest(BaseModel):
    topics: List[str]
    class_description: Optional[str] = ""
    duration_minutes: Optional[int] = 45
    teaching_style: Optional[str] = "interactive"  # interactive, lecture, hands-on
    difficulty_level: Optional[str] = "medium"

class GameGenerationRequest(BaseModel):
    plan_id: str
    simulation_type: str = "interactive"  # interactive, visual, experimental
    complexity: str = "medium"  # simple, medium, advanced
    interaction_mode: str = "sandbox"  # sandbox, guided, challenge

# Pydantic models from utils.py
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

######################## MULTILINGUAL: Language detection and voice selection functions ########################
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
        # Qdrant Docker configuration
        print(f"Connecting to Qdrant Docker at {QDRANT_CONFIG['host']}:{QDRANT_CONFIG['port']}")
        
        qdrant_client = QdrantClient(
            host=QDRANT_CONFIG['host'],
            port=QDRANT_CONFIG['port'],
            timeout=30
        )
        
        # Test connection
        try:
            collections = qdrant_client.get_collections()
            print(f"✅ Successfully connected to Qdrant Docker. Existing collections: {len(collections.collections)}")
        except Exception as e:
            print(f"❌ Failed to connect to Qdrant Docker: {e}")
            print("💡 Make sure Qdrant Docker container is running: docker run -p 6333:6333 qdrant/qdrant")
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

# =====================
# AUTHENTICATION HELPERS
# =====================
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(401, "Authentication required")
    
    try:
        decoded_token = auth.verify_id_token(credentials.credentials)
        user_id = decoded_token['uid']
        user_doc = db.collection('users').document(user_id).get()
        
        if not user_doc.exists:
            raise HTTPException(404, "User not found")
        
        user_data = user_doc.to_dict()
        user_data['uid'] = user_id
        return user_data
    except Exception as e:
        raise HTTPException(401, f"Invalid token: {str(e)}")

def generate_join_code() -> str:
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

def ensure_unique_join_code() -> str:
    while True:
        code = generate_join_code()
        existing = db.collection('classes').where('joinCode', '==', code).limit(1).get()
        if not existing:
            return code

# =====================
# PERSONA SYSTEM
# =====================
@router.get("/persona/questions")
async def get_persona_questions():
    """Get the 20 persona questions"""
    questions = [
        # Only including first 3 for brevity - full list would be here
        {
            "id": "q1",
            "question": "When your teacher explains something new, what helps you understand best?",
            "type": "multiple_choice",
            "options": [
                "Step-by-step explanations with clear examples",
                "Stories and analogies that relate to things I know",
                "Simple, direct explanations without extra details",
                "Detailed explanations that cover everything thoroughly"
            ]
        },
        {
            "id": "q2",
            "question": "If you had to learn about animals, which would you prefer?",
            "type": "multiple_choice",
            "options": [
                "Reading interesting facts and stories about animals",
                "Learning through comparisons (\"A lion is like a big house cat\")",
                "Short, simple descriptions of each animal",
                "Detailed explanations about how animals live and behave"
            ]
        },
        {
            "id": "q3",
            "question": "When you're trying to remember something important, what works best?",
            "type": "multiple_choice",
            "options": [
                "Breaking it into smaller, simpler parts",
                "Connecting it to stories or things I already know",
                "Repeating the main points several times",
                "Understanding all the details and reasons why"
            ]
        },
        # ... (all 20 questions would be here)
    ]
    return {"questions": questions}

# =====================
# ENHANCED PERSONA ANALYSIS WITH LLM
# =====================
def summarize_persona_with_llm(responses: List[Dict]) -> Dict:
    """Use LLM to analyze persona responses and generate learning profile"""
    try:
        # Format responses for LLM
        formatted_responses = "\n".join(
            [f"Q: {r['questionId']}: {r['response']}" for r in responses]
        )
        
        messages = [
            {
                "role": "system",
                "content": "Analyze these learning preference responses to create a comprehensive student persona profile. "
                           "Identify learning style (visual, auditory, kinesthetic, logical), "
                           "motivation factors, cognitive traits, and personalized teaching recommendations. "
                           "Return JSON format: {learning_style: str, motivation_factors: list, cognitive_traits: list, "
                           "personalization_tags: list, teaching_recommendations: list}"
            },
            {
                "role": "user",
                "content": formatted_responses
            }
        ]
        
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[messages[1]["content"]],  # Use the user message content
            config=types.GenerateContentConfig(
                temperature=0.1
            )
        )
        
        return json.loads(response.text)
    except Exception as e:
        logging.error(f"Persona analysis failed: {str(e)}")
        return {
            "learning_style": "general",
            "motivation_factors": [],
            "cognitive_traits": [],
            "personalization_tags": [],
            "teaching_recommendations": []
        }

@router.post("/persona/submit")
async def submit_persona_responses(
    responses: PersonaResponse,
    current_user: Dict = Depends(get_current_user)
):
    if current_user.get('role') != 'student':
        raise HTTPException(403, "Only students can submit persona responses")
    
    try:
        # Analyze responses with LLM
        persona_analysis = summarize_persona_with_llm(responses.responses)
        
        # Store in Firestore
        persona_data = {
            "userId": current_user['uid'],
            "responses": responses.responses,
            "analysis": persona_analysis,
            "createdAt": datetime.now()
        }
        db.collection('personas').document(current_user['uid']).set(persona_data)
        
        return {"status": "success", "analysis": persona_analysis}
    except Exception as e:
        raise HTTPException(500, f"Failed to save persona: {str(e)}")

# =====================
# CLASS MANAGEMENT
# =====================
@router.post("/classes/create")
async def create_class(
    class_data: ClassCreate,
    current_user: Dict = Depends(get_current_user)
):
    if current_user.get('role') != 'teacher':
        raise HTTPException(403, "Only teachers can create classes")
    
    try:
        class_id = f"class_{uuid.uuid4().hex[:12]}"
        join_code = ensure_unique_join_code()
        
        # Create class document
        class_doc = {
            "classId": class_id,
            "teacherId": current_user['uid'],
            "className": class_data.className,
            "subject": class_data.subject,
            "description": class_data.description,
            "joinCode": join_code,
            "createdAt": datetime.now(),
            "settings": class_data.settings
        }
        db.collection('classes').document(class_id).set(class_doc)
        
        # Update teacher's class list
        db.collection('users').document(current_user['uid']).update({
            "classes": firestore.ArrayUnion([class_id])
        })
        
        # Create Qdrant collection
        qdrant_result = create_virtual_class(
            teacher_id=current_user['uid'],
            class_name=class_data.className,
            subject=class_data.subject
        )
        
        return {
            "classId": class_id,
            "joinCode": join_code,
            "qdrantCollection": qdrant_result["collection"]
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to create class: {str(e)}")

@router.post("/classes/join")
async def join_class(
    join_request: JoinClassRequest,
    current_user: Dict = Depends(get_current_user)
):
    if current_user.get('role') != 'student':
        raise HTTPException(403, "Only students can join classes")
    
    try:
        # Find class by join code
        classes = db.collection('classes').where('joinCode', '==', join_request.joinCode).limit(1).get()
        if not classes:
            raise HTTPException(404, "Invalid join code")
        
        class_doc = classes[0]
        class_id = class_doc.id
        class_data = class_doc.to_dict()
        
        # Add student to class
        member_data = {
            "userId": current_user['uid'],
            "joinedAt": datetime.now(),
            "role": "student"
        }
        db.collection('classMembers').document(class_id).collection('students').document(current_user['uid']).set(member_data)
        
        # Update student's class list
        db.collection('users').document(current_user['uid']).update({
            "classes": firestore.ArrayUnion([class_id])
        })
        
        return {
            "classId": class_id,
            "className": class_data['className'],
            "subject": class_data['subject']
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to join class: {str(e)}")

# =====================
# ENHANCED CLASS UPLOAD WITH RAG INTEGRATION
# =====================
@router.post("/classes/{class_id}/upload")
async def upload_document(
    class_id: str,
    file: UploadFile,
    current_user: Dict = Depends(get_current_user),
    difficulty: str = Form("medium"),
    grade_level: str = Form("unknown"),
    language: str = Form("english")
):
    # Verify teacher owns the class
    class_ref = db.collection('classes').document(class_id)
    class_doc = class_ref.get()
    if not class_doc.exists:
        raise HTTPException(404, "Class not found")
    class_data = class_doc.to_dict()
    if class_data['teacherId'] != current_user['uid']:
        raise HTTPException(403, "Only class teacher can upload documents")
    
    # File validation
    if not file.filename.lower().endswith(('.pdf', '.txt')):
        raise HTTPException(400, "Only PDF/TXT files supported")
    
    try:
        # Read and validate file size
        content = await file.read()
        file_size_mb = len(content) / (1024 * 1024)
        if file_size_mb > 50:
            raise HTTPException(413, "File too large. Maximum size is 50MB.")
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        # Prepare metadata
        metadata = {
            "class_id": class_id,
            "subject": class_data['subject'],
            "filename": file.filename,
            "difficulty": difficulty,
            "grade_level": grade_level,
            "language": language,
            "uploaded_at": datetime.now().isoformat()
        }
        
        # Extract content using RAG functions
        chunks = extract_content_universal(tmp_path, metadata)
        if not chunks:
            raise HTTPException(400, "No content extracted")
        
        # Generate embeddings using RAG functions
        texts = [chunk["text"] for chunk in chunks]
        embeddings = get_jina_embeddings(texts)
        
        # Store in Qdrant using RAG functions
        collection_name = f"class_{class_id}_docs"
        result = store_chunks_in_class(
            class_id=class_id,
            chunks=chunks,
            embeddings=embeddings,
            collection_name=collection_name
        )
        
        # Create document record
        doc_data = {
            "classId": class_id,
            "fileName": file.filename,
            "uploadedBy": current_user['uid'],
            "uploadedAt": datetime.now(),
            "chunksCount": len(chunks),
            "status": "processed",
            "collection": collection_name  # Store Qdrant collection name
        }
        db.collection('documents').add(doc_data)
        
        return {
            "status": "success",
            "chunksStored": len(chunks),
            "collection": collection_name
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

# =====================
# ENHANCED EXPLAIN WITH RAG RETRIEVAL
# =====================
@router.post("/classes/{class_id}/explain")
async def explain_concept(
    class_id: str,
    concept: str = Form(...),
    analogy_context: Optional[str] = Form(None),
    current_user: Dict = Depends(get_current_user)
):
    # Verify class access
    if class_id not in current_user.get('classes', []):
        raise HTTPException(403, "Access denied to this class")
    
    try:
        # Get Qdrant collection name for class
        docs_ref = db.collection('documents').where("classId", "==", class_id).limit(1)
        docs = docs_ref.stream()
        collection_name = next(docs).to_dict().get("collection") if docs else None
        
        if not collection_name:
            raise HTTPException(404, "No documents found for this class")
        
        # Get persona profile
        persona_doc = db.collection('personas').document(current_user['uid']).get()
        persona_data = persona_doc.to_dict() if persona_doc.exists else {}
        
        # Get class subject
        class_doc = db.collection('classes').document(class_id).get()
        subject = class_doc.to_dict().get('subject', 'General') if class_doc.exists else 'General'
        
        # Generate embedding for concept using RAG functions
        concept_embedding = get_single_jina_embedding(concept)
        
        # Retrieve relevant context using RAG functions
        retrieved_results = search_class_content(class_id, concept_embedding, limit=10)
        
        # Build context from results
        context_chunks = []
        for result in retrieved_results:
            context_chunks.append({
                "text": result.get("text", ""),
                "source": result.get("filename", "Unknown")
            })
        
        # Build personalized prompt with Auto-CoT
        system_prompt, user_prompt = build_personalized_prompt(
            concept=concept,
            subject=subject,
            context_chunks=context_chunks,
            persona=persona_data,
            analogy_context=analogy_context
        )
        
        # Generate explanation
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[full_prompt],
            config=types.GenerateContentConfig(
                temperature=0.3
            )
        )
        
        explanation = response.text
        
        # Log interaction
        interaction_data = {
            "userId": current_user['uid'],
            "classId": class_id,
            "type": "explanation",
            "concept": concept,
            "timestamp": datetime.now()
        }
        db.collection('interactions').add(interaction_data)
        
        return {
            "explanation": explanation,
            "sources": list(set([chunk['source'] for chunk in context_chunks]))
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Explanation failed: {str(e)}")



# =====================
# ENHANCED AUTO-COT PROMPT ENGINEERING
# =====================
def build_personalized_prompt(
    concept: str,
    subject: str,
    context_chunks: List[str],
    persona: Dict,
    analogy_context: Optional[str] = None
) -> tuple:
    """Build personalized prompt with enhanced Auto-CoT structure"""
    # Extract persona insights
    analysis = persona.get("analysis", {})
    learning_style = analysis.get("learning_style", "general")
    teaching_recs = analysis.get("teaching_recommendations", [])
    tags = analysis.get("personalization_tags", [])
    
    # System prompt with detailed Auto-CoT instructions
    system_prompt = f"""
    You are an expert {subject} tutor. Explain concepts using comprehensive step-by-step reasoning 
    (Chain-of-Thought) and adapt to the student's learning profile:
    
    Student Profile:
    - Primary learning style: {learning_style}
    - Key motivation factors: {', '.join(analysis.get('motivation_factors', []))}
    - Cognitive traits: {', '.join(analysis.get('cognitive_traits', []))}
    
    Teaching Recommendations:
    {'. '.join(teaching_recs)}
    
    Always follow these reasoning steps:
    1. Start with a concise, precise definition
    2. Break concept into core components/first principles
    3. Provide 2-3 relatable examples (vary complexity)
    4. Explain real-world applications and significance
    5. Address common misconceptions
    6. Summarize key takeaways
    7. Suggest exploration paths for deeper understanding
    """
    
    # User prompt with formatted context
    user_prompt = f"Explain: {concept}\n\n"
    
    if analogy_context:
        user_prompt += f"## Analogy Request:\nRelate to {analogy_context}\n\n"
    
    if context_chunks:
        user_prompt += "## Relevant Class Materials:\n"
        user_prompt += "\n---\n".join([f"- {chunk}" for chunk in context_chunks])
        user_prompt += "\n\n"
    
    # Add personalized instructions
    personalization = []
    if "visual_learner" in tags:
        personalization.append("Include visual/spatial descriptions")
    if "detailed_explanations" in tags:
        personalization.append("Provide comprehensive technical details")
    if "real_world" in tags:
        personalization.append("Focus on practical applications")
    
    if personalization:
        user_prompt += f"## Personalization Instructions:\n{'. '.join(personalization)}"
    
    return system_prompt, user_prompt


# =====================
# OTHER ENDPOINTS
# =====================
@router.get("/classes/{class_id}/analytics")
async def get_analytics(
    class_id: str,
    current_user: Dict = Depends(get_current_user)
):
    # Verify teacher owns class
    class_doc = db.collection('classes').document(class_id).get()
    if not class_doc.exists:
        raise HTTPException(404, "Class not found")
    if class_doc.to_dict().get('teacherId') != current_user['uid']:
        raise HTTPException(403, "Access denied")
    
    try:
        # Get basic stats
        stats = get_class_stats(class_id)
        
        # Get interaction counts
        interactions = db.collection('interactions').where('classId', '==', class_id).get()
        
        return {
            "qdrantStats": stats,
            "interactionCount": len(interactions),
            "documentCount": len(db.collection('documents').where('classId', '==', class_id).get())
        }
    except Exception as e:
        raise HTTPException(500, f"Analytics failed: {str(e)}")

@router.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@router.get("/firebase/test")
async def firebase_test():
    db = firestore.client()
    db.collection("test").document("ping").set({"hello": "world"})
    return {"status": "written"}

# =====================
# MULTILINGUAL: Language detection and voice selection functions
# =====================
# MULTILINGUAL: Updated languages endpoint instead of voices
@router.get("/api/languages")
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

@router.get("/api/audio-status/{query_id}", response_model=AudioResponse)
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

@router.get("/api/audio/{query_id}")
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

@router.delete("/api/documents")
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

@router.delete("/api/audio-cache")
async def clear_audio_cache():
    count = len(audio_cache)
    audio_cache.clear()
    return {"status": "success", "message": f"Cleared {count} audio cache entries"}


# =====================
# LESSON PLANNER SYSTEM
# =====================

@router.get("/classes/{class_id}/topics")
async def get_class_topics(
    class_id: str,
    class_description: str = "",
    difficulty_level: str = "medium",
    grade_level: str = "unknown",
    current_user: Dict = Depends(get_current_user)
):
    """Extract and suggest topics from class materials using RAG"""
    
    # Verify class access
    class_doc = db.collection('classes').document(class_id).get()
    if not class_doc.exists:
        raise HTTPException(404, "Class not found")
    
    class_data = class_doc.to_dict()
    
    # Check access permissions
    if current_user.get('role') == 'teacher':
        if class_data['teacherId'] != current_user['uid']:
            raise HTTPException(403, "Access denied to this class")
    elif current_user.get('role') == 'student':
        if class_id not in current_user.get('classes', []):
            raise HTTPException(403, "Access denied to this class")
    else:
        raise HTTPException(403, "Invalid user role")
    
    try:
        subject = class_data.get('subject', 'General')
        
        # Create a broad query to get diverse content from the class materials
        topic_queries = [
            f"{subject} main concepts and topics",
            f"{subject} key learning objectives",
            f"{subject} fundamental principles",
            f"{subject} important topics to teach"
        ]
        
        all_content = []
        for query in topic_queries:
            query_embedding = get_single_jina_embedding(query)
            retrieved_results = search_class_content(class_id, query_embedding, limit=10)
            all_content.extend([item.get("text", "") for item in retrieved_results])
        
        # Remove duplicates and empty content
        unique_content = list(set([content for content in all_content if content.strip()]))
        
        if not unique_content:
            raise HTTPException(404, "No content found in class materials")
        
        # Use LLM to extract and organize topics
        content_sample = "\n".join(unique_content[:15])  # Limit content to avoid token limits
        
        system_prompt = f"""
        You are an educational content analyzer. Extract and organize key topics from the provided {subject} course materials.
        
        Instructions:
        1. Identify 8-15 distinct topics/concepts that can be taught
        2. Organize them by complexity (beginner to advanced)
        3. Ensure topics are specific enough to create focused lessons
        4. Consider the difficulty level: {difficulty_level}
        5. Consider the grade level: {grade_level}
        
        Return a JSON object with this structure:
        {{
            "beginner_topics": ["topic1", "topic2", ...],
            "intermediate_topics": ["topic1", "topic2", ...],
            "advanced_topics": ["topic1", "topic2", ...],
            "suggested_sequence": ["topic1", "topic2", "topic3", ...],
            "total_topics": number
        }}
        """
        
        user_prompt = f"""
        Subject: {subject}
        Class Description: {class_description}
        Difficulty Level: {difficulty_level}
        Grade Level: {grade_level}
        
        Course Materials Content:
        {content_sample}
        
        Extract and organize the key topics from this content.
        """
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[full_prompt],
            config=types.GenerateContentConfig(
                temperature=0.2
            )
        )
        
        topics_data = json.loads(response.text)
        
        # Log the interaction
        interaction_data = {
            "userId": current_user['uid'],
            "classId": class_id,
            "type": "topic_extraction",
            "timestamp": datetime.now(),
            "metadata": {
                "difficulty_level": difficulty_level,
                "grade_level": grade_level,
                "topics_count": topics_data.get("total_topics", 0)
            }
        }
        db.collection('interactions').add(interaction_data)
        
        return {
            "classId": class_id,
            "subject": subject,
            "topics": topics_data,
            "class_description": class_description
        }
        
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error in topic extraction: {str(e)}")
        raise HTTPException(500, "Failed to parse topic extraction results")
    except Exception as e:
        logging.error(f"Topic extraction failed: {str(e)}")
        raise HTTPException(500, f"Topic extraction failed: {str(e)}")


@router.post("/classes/{class_id}/planner")
async def create_daily_teaching_plan(
    class_id: str,
    plan_request: TeachingPlanRequest,
    current_user: Dict = Depends(get_current_user)
):
    """Generate a comprehensive daily teaching plan based on selected topics"""
    
    # Verify teacher access
    if current_user.get('role') != 'teacher':
        raise HTTPException(403, "Only teachers can create teaching plans")
    
    class_doc = db.collection('classes').document(class_id).get()
    if not class_doc.exists:
        raise HTTPException(404, "Class not found")
    
    class_data = class_doc.to_dict()
    if class_data['teacherId'] != current_user['uid']:
        raise HTTPException(403, "Access denied to this class")
    
    if not plan_request.topics:
        raise HTTPException(400, "At least one topic must be provided")
    
    try:
        subject = class_data.get('subject', 'General')
        
        # Get relevant content for the selected topics
        all_context = []
        for topic in plan_request.topics:
            query_embedding = get_single_jina_embedding(f"{topic} {subject}")
            retrieved_results = search_class_content(class_id, query_embedding, limit=5)
            topic_content = [item.get("text", "") for item in retrieved_results]
            all_context.extend(topic_content)
        
        # Remove duplicates
        unique_context = list(set([content for content in all_context if content.strip()]))
        context_sample = "\n".join(unique_context[:20])  # Limit context
        
        # Get class demographics for personalization
        students_count = len(db.collection('classMembers').document(class_id).collection('students').get())
        
        system_prompt = f"""
        You are an expert educational planner specializing in {subject}. Create a comprehensive, structured daily teaching plan.
        
        Plan Requirements:
        - Duration: {plan_request.duration_minutes} minutes
        - Teaching Style: {plan_request.teaching_style}
        - Difficulty Level: {plan_request.difficulty_level}
        - Subject: {subject}
        
        Structure your plan with these sections:
        1. Learning Objectives (3-5 specific, measurable goals)
        2. Materials Needed
        3. Lesson Structure (with time allocations)
        4. Teaching Activities (detailed instructions)
        5. Assessment Methods
        6. Homework/Follow-up Activities
        7. Differentiation Strategies
        8. Potential Challenges & Solutions
        
        Return a detailed JSON object with this structure:
        {{
            "lesson_title": "string",
            "duration_minutes": number,
            "learning_objectives": ["objective1", "objective2", ...],
            "materials_needed": ["material1", "material2", ...],
            "lesson_structure": [
                {{
                    "phase": "Opening/Introduction",
                    "duration_minutes": number,
                    "activities": ["activity1", "activity2"],
                    "teacher_notes": "detailed instructions"
                }},
                {{
                    "phase": "Main Content",
                    "duration_minutes": number,
                    "activities": ["activity1", "activity2"],
                    "teacher_notes": "detailed instructions"
                }},
                {{
                    "phase": "Practice/Application",
                    "duration_minutes": number,
                    "activities": ["activity1", "activity2"],
                    "teacher_notes": "detailed instructions"
                }},
                {{
                    "phase": "Closure/Summary",
                    "duration_minutes": number,
                    "activities": ["activity1", "activity2"],
                    "teacher_notes": "detailed instructions"
                }}
            ],
            "assessment_methods": [
                {{
                    "type": "formative/summative",
                    "method": "description",
                    "timing": "when to use"
                }}
            ],
            "homework_activities": ["activity1", "activity2", ...],
            "differentiation_strategies": [
                {{
                    "student_type": "advanced/struggling/ELL/etc",
                    "strategies": ["strategy1", "strategy2"]
                }}
            ],
            "potential_challenges": [
                {{
                    "challenge": "description",
                    "solution": "how to address"
                }}
            ],
            "resources_links": ["resource1", "resource2", ...],
            "extension_activities": ["activity1", "activity2", ...]
        }}
        """
        
        user_prompt = f"""
        Create a daily teaching plan for:
        
        Topics to Cover: {', '.join(plan_request.topics)}
        Class Description: {plan_request.class_description}
        Subject: {subject}
        Duration: {plan_request.duration_minutes} minutes
        Teaching Style: {plan_request.teaching_style}
        Difficulty Level: {plan_request.difficulty_level}
        Estimated Class Size: {students_count} students
        
        Available Course Materials Context:
        {context_sample}
        
        Create a comprehensive, practical teaching plan that a teacher can immediately implement.
        """
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[full_prompt],
            config=types.GenerateContentConfig(
                temperature=0.3
            )
        )
        
        teaching_plan = json.loads(response.text)
        
        # Save the teaching plan to database
        plan_data = {
            "classId": class_id,
            "teacherId": current_user['uid'],
            "topics": plan_request.topics,
            "teaching_plan": teaching_plan,
            "duration_minutes": plan_request.duration_minutes,
            "teaching_style": plan_request.teaching_style,
            "difficulty_level": plan_request.difficulty_level,
            "created_at": datetime.now(),
            "class_description": plan_request.class_description
        }
        
        # Store in Firestore
        plan_doc_ref = db.collection('teaching_plans').add(plan_data)
        plan_id = plan_doc_ref[1].id
        
        # Log the interaction
        interaction_data = {
            "userId": current_user['uid'],
            "classId": class_id,
            "type": "teaching_plan_creation",
            "timestamp": datetime.now(),
            "metadata": {
                "plan_id": plan_id,
                "topics_count": len(plan_request.topics),
                "duration_minutes": plan_request.duration_minutes
            }
        }
        db.collection('interactions').add(interaction_data)
        
        return {
            "plan_id": plan_id,
            "teaching_plan": teaching_plan,
            "status": "success",
            "created_at": datetime.now().isoformat()
        }
        
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error in teaching plan: {str(e)}")
        raise HTTPException(500, "Failed to parse teaching plan results")
    except Exception as e:
        logging.error(f"Teaching plan creation failed: {str(e)}")
        raise HTTPException(500, f"Teaching plan creation failed: {str(e)}")


@router.get("/classes/{class_id}/plans")
async def get_teaching_plans(
    class_id: str,
    limit: int = 10,
    current_user: Dict = Depends(get_current_user)
):
    """Get previously created teaching plans for a class"""
    
    # Verify teacher access
    if current_user.get('role') != 'teacher':
        raise HTTPException(403, "Only teachers can view teaching plans")
    
    class_doc = db.collection('classes').document(class_id).get()
    if not class_doc.exists:
        raise HTTPException(404, "Class not found")
    
    class_data = class_doc.to_dict()
    if class_data['teacherId'] != current_user['uid']:
        raise HTTPException(403, "Access denied to this class")
    
    try:
        plans_query = (db.collection('teaching_plans')
                      .where('classId', '==', class_id)
                      .where('teacherId', '==', current_user['uid'])
                      .order_by('created_at', direction=firestore.Query.DESCENDING)
                      .limit(limit))
        
        plans = plans_query.get()
        
        plans_list = []
        for plan_doc in plans:
            plan_data = plan_doc.to_dict()
            plans_list.append({
                "plan_id": plan_doc.id,
                "topics": plan_data.get('topics', []),
                "lesson_title": plan_data.get('teaching_plan', {}).get('lesson_title', 'Untitled'),
                "duration_minutes": plan_data.get('duration_minutes', 0),
                "teaching_style": plan_data.get('teaching_style', ''),
                "created_at": plan_data.get('created_at'),
                "difficulty_level": plan_data.get('difficulty_level', 'medium')
            })
        
        return {
            "class_id": class_id,
            "plans": plans_list,
            "total_count": len(plans_list)
        }
        
    except Exception as e:
        logging.error(f"Failed to retrieve teaching plans: {str(e)}")
        raise HTTPException(500, f"Failed to retrieve teaching plans: {str(e)}")


@router.get("/classes/{class_id}/plans/{plan_id}")
async def get_teaching_plan_detail(
    class_id: str,
    plan_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Get detailed view of a specific teaching plan"""
    
    # Verify teacher access
    if current_user.get('role') != 'teacher':
        raise HTTPException(403, "Only teachers can view teaching plans")
    
    try:
        plan_doc = db.collection('teaching_plans').document(plan_id).get()
        if not plan_doc.exists:
            raise HTTPException(404, "Teaching plan not found")
        
        plan_data = plan_doc.to_dict()
        
        # Verify ownership
        if (plan_data.get('classId') != class_id or 
            plan_data.get('teacherId') != current_user['uid']):
            raise HTTPException(403, "Access denied to this teaching plan")
        
        return {
            "plan_id": plan_id,
            "class_id": class_id,
            "teaching_plan": plan_data.get('teaching_plan', {}),
            "topics": plan_data.get('topics', []),
            "metadata": {
                "duration_minutes": plan_data.get('duration_minutes'),
                "teaching_style": plan_data.get('teaching_style'),
                "difficulty_level": plan_data.get('difficulty_level'),
                "created_at": plan_data.get('created_at'),
                "class_description": plan_data.get('class_description', '')
            }
        }
        
    except Exception as e:
        logging.error(f"Failed to retrieve teaching plan detail: {str(e)}")
        raise HTTPException(500, f"Failed to retrieve teaching plan detail: {str(e)}")
    



# =====================
# EDUCATIONAL GAME GENERATION
# =====================

@router.post("/classes/{class_id}/generate-game")
async def generate_educational_game(
    class_id: str,
    plan_id: str = Form(...),
    simulation_type: str = Form("interactive"),
    complexity: str = Form("medium"),
    interaction_mode: str = Form("sandbox"),
    current_user: Dict = Depends(get_current_user)
):
    """Generate an interactive educational simulation where students can manipulate parameters and explore concepts"""
    
    # Verify class access
    if class_id not in current_user.get('classes', []):
        raise HTTPException(403, "Access denied to this class")
    
    try:
        # Get the teaching plan
        plan_doc = db.collection('teaching_plans').document(plan_id).get()
        if not plan_doc.exists:
            raise HTTPException(404, "Teaching plan not found")
        
        plan_data = plan_doc.to_dict()
        
        # Verify the plan belongs to this class
        if plan_data.get('classId') != class_id:
            raise HTTPException(403, "Teaching plan does not belong to this class")
        
        # Get class information
        class_doc = db.collection('classes').document(class_id).get()
        class_data = class_doc.to_dict()
        subject = class_data.get('subject', 'General')
        
        # Extract information from teaching plan
        teaching_plan = plan_data.get('teaching_plan', {})
        topics = plan_data.get('topics', [])
        learning_objectives = teaching_plan.get('learning_objectives', [])
        lesson_title = teaching_plan.get('lesson_title', 'Interactive Learning Simulation')
        
        # Get persona profile for personalization (if student)
        persona_data = {}
        if current_user.get('role') == 'student':
            persona_doc = db.collection('personas').document(current_user['uid']).get()
            persona_data = persona_doc.to_dict() if persona_doc.exists else {}
        
        # Get additional context from class materials using RAG
        all_context = []
        for topic in topics:
            query_embedding = get_single_jina_embedding(f"{topic} {subject} parameters variables formulas relationships")
            retrieved_results = search_class_content(class_id, query_embedding, limit=4)
            topic_content = [item.get("text", "") for item in retrieved_results]
            all_context.extend(topic_content)
        
        # Remove duplicates and limit context
        unique_context = list(set([content for content in all_context if content.strip()]))
        context_sample = "\n".join(unique_context[:12])
        
        # Build interactive simulation generation prompt
        system_prompt = f"""
        You are an expert educational simulation developer specializing in interactive learning experiences.
        Create an interactive simulation where students can manipulate parameters, variables, and settings to understand how concepts work.
        
        Simulation Requirements:
        - Type: {simulation_type} simulation
        - Complexity: {complexity}
        - Mode: {interaction_mode}
        - Subject: {subject}
        
        Create a complete, functional HTML file with embedded CSS and JavaScript that includes:
        
        INTERACTIVE ELEMENTS:
        1. Parameter sliders/input controls (at least 3-5 adjustable parameters)
        2. Real-time visual feedback and updates
        3. Interactive graphs, charts, or visual representations
        4. Cause-and-effect demonstrations
        5. Live calculations and formula displays
        6. Parameter relationships visualization
        
        EDUCATIONAL FEATURES:
        7. Clear explanations of what each parameter does
        8. Tooltips and help text for guidance
        9. "What if" scenario exploration
        10. Real-world examples and applications
        11. Reset and preset configuration buttons
        12. Value ranges and realistic constraints
        
        TECHNICAL REQUIREMENTS:
        13. Responsive design for all devices
        14. Smooth animations and transitions
        15. Performance optimization for real-time updates
        16. Accessibility features (keyboard navigation, screen reader support)
        17. Error handling and input validation
        18. Export/save functionality for configurations
        
        SIMULATION TYPES:
        - Physics: Manipulate forces, velocities, masses, angles, etc.
        - Chemistry: Adjust concentrations, temperatures, pressures, pH levels
        - Mathematics: Change variables in equations, geometric parameters
        - Biology: Modify environmental factors, population parameters
        - Economics: Adjust market variables, supply/demand factors
        - Engineering: Control system parameters, design variables
        
        IMPORTANT: Return ONLY the complete HTML code with embedded CSS and JavaScript. 
        Do not include any markdown formatting or code blocks. The response should start with <!DOCTYPE html>
        
        Make the simulation educational, engaging, and focused on hands-on parameter exploration.
        Students should be able to see immediate visual feedback when they change values.
        """
        
        # Add personalization if student
        persona_info = ""
        if persona_data:
            analysis = persona_data.get("analysis", {})
            learning_style = analysis.get("learning_style", "general")
            persona_info = f"\nStudent Learning Profile: {learning_style} learner - adapt the interface and explanations accordingly"
        
        user_prompt = f"""
        Create an interactive parameter exploration simulation based on this lesson plan:
        
        Lesson Title: {lesson_title}
        Subject: {subject}
        Topics: {', '.join(topics)}
        Learning Objectives: {', '.join(learning_objectives)}
        Complexity Level: {complexity}
        Interaction Mode: {interaction_mode}
        {persona_info}
        
        Course Materials Context:
        {context_sample}
        
        Simulation Design Guidelines:
        1. Identify 3-7 key parameters/variables students can manipulate
        2. Create visual representations (graphs, diagrams, animations)
        3. Show real-time cause-and-effect relationships
        4. Include realistic value ranges and constraints
        5. Provide clear explanations of parameter effects
        6. Add preset scenarios for guided exploration
        7. Include measurement tools and data display
        8. Make abstract concepts tangible through interaction
        
        Focus on creating an exploratory environment where students can:
        - Experiment with different parameter combinations
        - Observe immediate visual/numerical feedback
        - Understand relationships between variables
        - Test hypotheses through manipulation
        - Learn through discovery and experimentation
        
        Example interactions based on subject:
        - Science: Adjust experimental conditions and see results
        - Math: Change equation parameters and see graph changes
        - History: Modify historical factors and see potential outcomes
        - Literature: Explore character motivations and plot variations
        
        Create a simulation that makes learning active, visual, and engaging through hands-on parameter manipulation.
        """
        
        # Generate the simulation using LLM
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[full_prompt],
            config=types.GenerateContentConfig(
                temperature=0.3
            )
        )
        
        simulation_html = response.text
        
        # Save the generated simulation to database
        simulation_data = {
            "classId": class_id,
            "planId": plan_id,
            "userId": current_user['uid'],
            "simulationType": simulation_type,
            "complexity": complexity,
            "interactionMode": interaction_mode,
            "lessonTitle": lesson_title,
            "topics": topics,
            "simulationHtml": simulation_html,
            "createdAt": datetime.now(),
            "subject": subject,
            "type": "interactive_simulation"
        }
        
        # Store in Firestore
        simulation_doc_ref = db.collection('educational_games').add(simulation_data)
        simulation_id = simulation_doc_ref[1].id
        
        # Log the interaction
        interaction_data = {
            "userId": current_user['uid'],
            "classId": class_id,
            "type": "interactive_simulation_generation",
            "timestamp": datetime.now(),
            "metadata": {
                "simulation_id": simulation_id,
                "simulation_type": simulation_type,
                "plan_id": plan_id,
                "complexity": complexity,
                "topics_count": len(topics)
            }
        }
        db.collection('interactions').add(interaction_data)
        
        return {
            "simulation_id": simulation_id,
            "simulation_html": simulation_html,
            "simulation_metadata": {
                "simulation_type": simulation_type,
                "complexity": complexity,
                "interaction_mode": interaction_mode,
                "lesson_title": lesson_title,
                "topics": topics,
                "subject": subject,
                "type": "interactive_simulation"
            },
            "status": "success",
            "created_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Interactive simulation generation failed: {str(e)}")
        raise HTTPException(500, f"Interactive simulation generation failed: {str(e)}")


@router.get("/classes/{class_id}/games")
async def get_class_simulations(
    class_id: str,
    limit: int = 20,
    simulation_type: Optional[str] = None,
    current_user: Dict = Depends(get_current_user)
):
    """Get all interactive simulations created for a class"""
    
    # Verify class access
    if class_id not in current_user.get('classes', []):
        raise HTTPException(403, "Access denied to this class")
    
    try:
        # Build query
        simulations_query = db.collection('educational_games').where('classId', '==', class_id)
        
        if simulation_type:
            simulations_query = simulations_query.where('simulationType', '==', simulation_type)
        
        simulations_query = simulations_query.order_by('createdAt', direction=firestore.Query.DESCENDING).limit(limit)
        
        simulations = simulations_query.get()
        
        simulations_list = []
        for sim_doc in simulations:
            sim_data = sim_doc.to_dict()
            simulations_list.append({
                "simulation_id": sim_doc.id,
                "lesson_title": sim_data.get('lessonTitle', 'Untitled Simulation'),
                "simulation_type": sim_data.get('simulationType', 'interactive'),
                "complexity": sim_data.get('complexity', 'medium'),
                "topics": sim_data.get('topics', []),
                "created_at": sim_data.get('createdAt'),
                "subject": sim_data.get('subject', 'General'),
                "interaction_mode": sim_data.get('interactionMode', 'sandbox'),
                "type": sim_data.get('type', 'interactive_simulation')
            })
        
        return {
            "class_id": class_id,
            "simulations": simulations_list,
            "total_count": len(simulations_list)
        }
        
    except Exception as e:
        logging.error(f"Failed to retrieve class simulations: {str(e)}")
        raise HTTPException(500, f"Failed to retrieve class simulations: {str(e)}")


@router.get("/classes/{class_id}/games/{simulation_id}")
async def get_simulation_detail(
    class_id: str,
    simulation_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Get detailed view of a specific interactive simulation"""
    
    # Verify class access
    if class_id not in current_user.get('classes', []):
        raise HTTPException(403, "Access denied to this class")
    
    try:
        sim_doc = db.collection('educational_games').document(simulation_id).get()
        if not sim_doc.exists:
            raise HTTPException(404, "Interactive simulation not found")
        
        sim_data = sim_doc.to_dict()
        
        # Verify the simulation belongs to this class
        if sim_data.get('classId') != class_id:
            raise HTTPException(403, "Simulation does not belong to this class")
        
        return {
            "simulation_id": simulation_id,
            "class_id": class_id,
            "simulation_html": sim_data.get('simulationHtml', ''),
            "metadata": {
                "lesson_title": sim_data.get('lessonTitle', ''),
                "simulation_type": sim_data.get('simulationType', ''),
                "complexity": sim_data.get('complexity', ''),
                "interaction_mode": sim_data.get('interactionMode', ''),
                "topics": sim_data.get('topics', []),
                "subject": sim_data.get('subject', ''),
                "created_at": sim_data.get('createdAt'),
                "plan_id": sim_data.get('planId', ''),
                "type": sim_data.get('type', 'interactive_simulation')
            }
        }
        
    except Exception as e:
        logging.error(f"Failed to retrieve simulation detail: {str(e)}")
        raise HTTPException(500, f"Failed to retrieve simulation detail: {str(e)}")


@router.post("/classes/{class_id}/games/{simulation_id}/interaction-session")
async def log_simulation_session(
    class_id: str,
    simulation_id: str,
    parameters_explored: str = Form(...),  # JSON string of parameters and values explored
    time_spent: int = Form(...),
    insights_gained: str = Form(""),
    configurations_tried: int = Form(1),
    current_user: Dict = Depends(get_current_user)
):
    """Log a simulation interaction session for learning analytics"""
    
    # Verify class access
    if class_id not in current_user.get('classes', []):
        raise HTTPException(403, "Access denied to this class")
    
    try:
        # Verify simulation exists and belongs to class
        sim_doc = db.collection('educational_games').document(simulation_id).get()
        if not sim_doc.exists or sim_doc.to_dict().get('classId') != class_id:
            raise HTTPException(404, "Simulation not found")
        
        # Parse parameters JSON
        try:
            parameters_data = json.loads(parameters_explored)
        except json.JSONDecodeError:
            parameters_data = {}
        
        # Create session record
        session_data = {
            "userId": current_user['uid'],
            "classId": class_id,
            "simulationId": simulation_id,
            "parametersExplored": parameters_data,
            "timeSpent": time_spent,
            "insightsGained": insights_gained,
            "configurationsTried": configurations_tried,
            "sessionAt": datetime.now(),
            "userRole": current_user.get('role', 'student'),
            "sessionType": "parameter_exploration"
        }
        
        # Store session
        session_ref = db.collection('simulation_sessions').add(session_data)
        session_id = session_ref[1].id
        
        # Log interaction
        interaction_data = {
            "userId": current_user['uid'],
            "classId": class_id,
            "type": "simulation_interaction",
            "timestamp": datetime.now(),
            "metadata": {
                "simulation_id": simulation_id,
                "session_id": session_id,
                "time_spent": time_spent,
                "configurations_tried": configurations_tried,
                "parameters_count": len(parameters_data)
            }
        }
        db.collection('interactions').add(interaction_data)
        
        return {
            "session_id": session_id,
            "status": "success",
            "message": "Simulation interaction session logged successfully"
        }
        
    except Exception as e:
        logging.error(f"Failed to log simulation session: {str(e)}")
        raise HTTPException(500, f"Failed to log simulation session: {str(e)}")
