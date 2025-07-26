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

# RAG service imports
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
groq_client = Groq(api_key='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')

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
        
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama3-70b-8192",
            temperature=0.2,
            max_tokens=1500,
            response_format={"type": "json_object"}
        )
        
        topics_data = json.loads(response.choices[0].message.content)
        
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
        
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama3-70b-8192",
            temperature=0.3,
            max_tokens=3000,
            response_format={"type": "json_object"}
        )
        
        teaching_plan = json.loads(response.choices[0].message.content)
        
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
    
# Add this Pydantic model with other models
class GameGenerationRequest(BaseModel):
    plan_id: str
    simulation_type: str = "interactive"  # interactive, visual, experimental
    complexity: str = "medium"  # simple, medium, advanced
    interaction_mode: str = "sandbox"  # sandbox, guided, challenge


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
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama3-70b-8192",
            temperature=0.3,
            max_tokens=4000
        )
        
        simulation_html = response.choices[0].message.content
        
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
