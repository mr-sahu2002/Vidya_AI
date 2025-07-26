from fastapi import APIRouter, UploadFile, HTTPException, Form, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, List, Dict, Any
import tempfile
import os
import json
import uuid
import string
import random
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore, auth
from pydantic import BaseModel
import logging # Added logging import

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate("../vidyaai-1dcfa-firebase-adminsdk-fbsvc-513f50d8d1.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()
security = HTTPBearer(auto_error=False)

# Service imports
from rag.extractor import extract_content_universal
from rag.embedder import get_jina_embeddings, get_single_jina_embedding
from rag.qdrant_service import (
    create_virtual_class,
    store_chunks_in_class,
    search_class_content,
    get_class_stats
)
from groq import Groq

# Initialize Groq client
groq_client = Groq(api_key='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',)

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
        
        response = groq_client.chat.completions.create(
            messages=messages,
            model="llama3-70b-8192",
            temperature=0.1,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
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
# DOCUMENT UPLOAD
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
    class_doc = db.collection('classes').document(class_id).get()
    if not class_doc.exists:
        raise HTTPException(404, "Class not found")
    class_data = class_doc.to_dict()
    if class_data['teacherId'] != current_user['uid']:
        raise HTTPException(403, "Only class teacher can upload documents")
    
    if not file.filename.endswith(('.pdf', '.txt')):
        raise HTTPException(400, "Only PDF/TXT files supported")
    
    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
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
        
        # Extract content
        chunks = extract_content_universal(tmp_path, metadata)
        if not chunks:
            raise HTTPException(400, "No content extracted")
        
        # Generate embeddings
        texts = [chunk["text"] for chunk in chunks]
        embeddings = get_jina_embeddings(texts)
        
        # Store in Qdrant
        result = store_chunks_in_class(class_id, chunks, embeddings)
        if result["status"] != "success":
            raise HTTPException(500, result["message"])
        
        # Create document record
        doc_data = {
            "classId": class_id,
            "fileName": file.filename,
            "uploadedBy": current_user['uid'],
            "uploadedAt": datetime.now(),
            "chunksCount": len(chunks),
            "status": "processed"
        }
        db.collection('documents').add(doc_data)
        
        return {"status": "success", "chunksStored": len(chunks)}
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {str(e)}")
    finally:
        os.unlink(tmp_path)

# =====================
# PERSONALIZED RAG WITH AUTO-COT
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
        # Get persona profile
        persona_doc = db.collection('personas').document(current_user['uid']).get()
        persona_data = persona_doc.to_dict() if persona_doc.exists else {}
        
        # Get class subject
        class_doc = db.collection('classes').document(class_id).get()
        subject = class_doc.to_dict().get('subject', 'General') if class_doc.exists else 'General'
        
        # Retrieve relevant context
        query_embedding = get_single_jina_embedding(concept)
        retrieved_results = search_class_content(class_id, query_embedding, limit=5)
        context_chunks = [item.get("text", "") for item in retrieved_results]
        
        # Build personalized prompt with Auto-CoT
        system_prompt, user_prompt = build_personalized_prompt(
            concept=concept,
            subject=subject,
            context_chunks=context_chunks,
            persona=persona_data,
            analogy_context=analogy_context
        )
        
        # Generate explanation
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama3-70b-8192",
            temperature=0.3,
            max_tokens=2000
        )
        
        explanation = response.choices[0].message.content
        
        # Log interaction
        interaction_data = {
            "userId": current_user['uid'],
            "classId": class_id,
            "type": "explanation",
            "concept": concept,
            "timestamp": datetime.now()
        }
        db.collection('interactions').add(interaction_data)
        
        return {"explanation": explanation}
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

@router.get("/classes/{class_id}/planner")
async def get_analytics():
    pass
