from fastapi import APIRouter, UploadFile, HTTPException, Form
from typing import Optional, List
import tempfile
import os
from datetime import datetime
import json # Import json for structured output

# Assume these services are correctly implemented in their respective files
from .services.extractor import extract_content_universal # Already there
from .services.embedder import get_jina_embeddings, get_single_jina_embedding # get_single_jina_embedding added
from .services.qdrant_service import ( # All these are already there
    create_virtual_class,
    store_chunks_in_class,
    search_class_content,
    get_class_stats,
    get_class_subject # We'll use this to get the subject for context
)

# New import for LLM interaction (e.g., Groq)
from groq import Groq # Assuming you've installed 'groq-sdk'

# Initialize Groq client
# Ensure GROQ_API_KEY is set in your environment variables
groq_client = Groq(api_key='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',)

router = APIRouter()

# --- Existing Routes (Copy-pasted for completeness, no changes) ---

@router.post("/create-class")
async def api_create_class(
    teacher_id: str = Form(...),
    class_name: str = Form(...),
    subject: str = Form(...)
):
    """Create a new virtual class"""
    try:
        result = create_virtual_class(teacher_id, class_name, subject)
        return {
            "status": "created",
            "class_id": result["class_id"],
            "collection": result["collection"],
            "subject": subject
        }
    except Exception as e:
        raise HTTPException(500, f"Class creation failed: {str(e)}")

@router.post("/upload")
async def upload_content(
    file: UploadFile,
    class_id: str = Form(...),
    difficulty: str = Form("medium"),
    grade_level: str = Form("unknown"),
    language: str = Form("english"),
):
    """Upload content to a virtual class"""
    if not file.filename.endswith(('.pdf', '.txt')):
        raise HTTPException(400, "Only PDF/TXT files supported")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Prepare metadata
        # We need to fetch the subject from the class registry if it's not provided explicitly during upload
        class_subject_info = get_class_subject(class_id)
        if class_subject_info.get("status") == "error":
            raise HTTPException(404, class_subject_info["message"])
        
        metadata = {
            "class_id": class_id,
            "subject": class_subject_info.get("subject", "unknown"), # Use subject from class registry
            "filename": file.filename,
            "difficulty": difficulty.lower(),
            "grade_level": grade_level.lower(),
            "language": language.lower(),
            "uploaded_at": datetime.now().isoformat()
        }
        
        # Extract content
        chunks = extract_content_universal(tmp_path, metadata)
        if not chunks:
            raise HTTPException(400, "No content extracted")
        
        # Generate embeddings
        texts = [chunk["text"] for chunk in chunks]
        embeddings = get_jina_embeddings(texts) # Assuming this takes a list of texts
        
        # Store in Qdrant
        result = store_chunks_in_class(class_id, chunks, embeddings)
        if result["status"] != "success":
            raise HTTPException(500, result["message"])
        
        return {
            "status": "success",
            "class_id": class_id,
            "chunks_stored": len(chunks),
            "filename": file.filename
        }
    finally:
        os.unlink(tmp_path)

@router.post("/search")
async def search_content_api( # Renamed to avoid conflict with service function
    class_id: str = Form(...),
    query: str = Form(...),
    difficulty: Optional[str] = Form(None),
    grade_level: Optional[str] = Form(None),
    content_type: Optional[str] = Form(None),
    limit: int = Form(5)
):
    """Search within a virtual class"""
    try:
        # Get single embedding for query
        query_embedding = get_single_jina_embedding(query) # Assuming this takes a single text
        
        # Prepare filters
        filters = {}
        if difficulty:
            filters["difficulty"] = difficulty.lower()
        if grade_level:
            filters["grade_level"] = grade_level.lower()
        if content_type:
            filters["content_type"] = content_type
        
        # Search class content
        results = search_class_content(class_id, query_embedding, filters, limit)
        
        return {
            "class_id": class_id,
            "query": query,
            "results": results
        }
    except Exception as e:
        raise HTTPException(500, f"Search failed: {str(e)}")

@router.get("/class-stats/{class_id}")
async def class_stats_api(class_id: str): # Renamed to avoid conflict
    """Get statistics for a virtual class"""
    stats = get_class_stats(class_id)
    if stats.get("status") == "error":
        raise HTTPException(404, stats["message"])
    return stats

@router.get("/health")
async def health_check():
    return {"status": "ok", "time": datetime.now().isoformat()}

async def explain_concept_with_rag(
    class_id: str,
    concept_to_explain: str,
    analogy_context: Optional[str] = None,
    subject_context: Optional[str] = None,
    user_prompt_details: Optional[str] = None,
    k_examples: int = 5,
    model_name: str = "llama3-8b-8192"
) -> str:
    """
    Performs RAG to explain a concept, mimicking Auto-CoT.
    """
    try:
        # 1. Get Subject Context if not explicitly provided
        # NOTE: Assumes get_class_subject returns Dict[str, Any] as per previous fix
        if not subject_context:
            class_info = get_class_subject(class_id)
            if class_info.get("status") == "success" and "subject" in class_info:
                subject_context = class_info["subject"]
            else:
                subject_context = "General Knowledge" # Default if not found
        
        # 2. Retrieval Phase: Fetch k examples from Qdrant
        print(f"[RAG] Retrieving {k_examples} examples for concept: '{concept_to_explain}' in class '{class_id}' (Subject: {subject_context})...")
        query_embedding = get_single_jina_embedding(concept_to_explain)
        
        # *** FIX START ***
        # Prepare filters for the Qdrant search
        retrieval_filters = {}
        if subject_context and subject_context != "General Knowledge": # Only filter if a specific subject is found
            retrieval_filters["subject"] = subject_context.lower() # Ensure it's lowercase for matching Qdrant index
        
        # You could also add difficulty/grade_level filters here if you plan to expose them
        # if difficulty: retrieval_filters["difficulty"] = difficulty.lower()
        # if grade_level: retrieval_filters["grade_level"] = grade_level.lower()

        retrieved_results = search_class_content(
            class_id,
            query_embedding,
            filters=retrieval_filters, # <--- Pass the new filters here
            limit=k_examples
        )
        # *** FIX END ***
        
        context_chunks: List[str] = [item.get("text", "") for item in retrieved_results] # Use .get for safety, though "text" should exist
        print(f"[RAG] Retrieved {len(context_chunks)} relevant chunks.")

        # 3. Reasoning & Generation Phase: Construct Auto-CoT inspired prompt for LLM
        # ... (rest of the function remains the same) ...

        system_instruction = (
            "You are an intelligent, helpful, and highly articulate AI assistant specializing in educational content. "
            "Your primary goal is to provide clear, comprehensive, and accurate explanations of concepts. "
            "You are operating within an educational platform where users (students or teachers) seek to understand specific topics. "
            "When explaining, mimic a thoughtful, step-by-step reasoning process (similar to Chain-of-Thought), "
            "and always ground your answers in the provided context, integrating it naturally. "
            "If an analogy context is provided, try to weave it into your explanation where appropriate. "
            "The subject of the class this content belongs to is important for framing your explanation."
        )

        user_message_parts: List[str] = [
            f"Please explain the concept of '{concept_to_explain}'.",
            f"The subject matter for this class is '{subject_context}'.",
        ]

        if analogy_context:
            user_message_parts.append(f"I am trying to understand '{concept_to_explain}' using an analogy related to '{analogy_context}'. Please incorporate this analogy or suggest a relevant one to aid understanding.")
        
        if user_prompt_details:
            user_message_parts.append(f"Additional instructions from the user: {user_prompt_details}")

        user_message_parts.append("\n\nLet's think step by step to provide a comprehensive and easy-to-understand explanation:")
        user_message_parts.append(f"1. **Define the concept:** Start with a clear and concise definition of '{concept_to_explain}'.")
        user_message_parts.append("2. **Break down key components:** Elaborate on its core principles, characteristics, or important sub-topics.")
        user_message_parts.append("3. **Provide examples/analogies:** Illustrate with real-world examples or a helpful analogy. If an analogy context was given, try to use or build upon it.")
        user_message_parts.append("4. **Discuss context/implications:** Explain its relevance, applications, or any broader implications within the subject matter.")
        user_message_parts.append("5. **Summarize:** Conclude with a brief summary of the main points.")

        if context_chunks:
            user_message_parts.append("\n\nHere is some highly relevant information retrieved from our knowledge base that you MUST use and integrate into your explanation:")
            for i, chunk in enumerate(context_chunks):
                user_message_parts.append(f"--- Retrieved Context {i+1} ---\n{chunk}")
            user_message_parts.append("--- End of Retrieved Contexts ---")
            user_message_parts.append("\n\nNow, generate the explanation by synthesizing your knowledge and the provided contexts, following the steps above.")
        else:
            user_message_parts.append("\n\nNo specific external context was retrieved. Please provide the explanation based on your general knowledge and the step-by-step guidance.")
        
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": "\n".join(user_message_parts)}
        ]

        print("[LLM] Sending structured prompt to Groq for explanation generation...")
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model=model_name,
            temperature=0.3, # Keeps the output more factual and less creative
            max_tokens=2000, # Allow for a comprehensive explanation
            response_format={"type": "text"} # Ensure text output
        )
        
        explanation = chat_completion.choices[0].message.content
        return explanation

    except Exception as e:
        print(f"Error in explain_concept_with_rag: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to explain concept: {str(e)}")

@router.post("/explain-concept")
async def api_explain_concept(
    class_id: str = Form(...),
    concept: str = Form(..., alias="conceptToExplain"), # Renamed for clarity, map to "conceptToExplain" from form
    analogy_context: Optional[str] = Form(None, alias="p1"), # p1 as analogy_context
    subject_context: Optional[str] = Form(None, alias="p2"), # p2 as subject_context
    user_prompt: Optional[str] = Form(None, alias="prompt"), # original prompt as user_prompt
    k_examples: int = Form(5)
):
    """
    Explains a concept using RAG (Retrieval Augmented Generation) and an Auto-CoT mimicked approach.
    Fetches relevant information from the class knowledge base and uses an LLM to generate an explanation.
    """
    try:
        explanation = await explain_concept_with_rag(
            class_id=class_id,
            concept_to_explain=concept,
            analogy_context=analogy_context,
            subject_context=subject_context,
            user_prompt_details=user_prompt,
            k_examples=k_examples
        )
        return {
            "status": "success",
            "concept": concept,
            "explanation": explanation,
            "class_id": class_id
        }
    except HTTPException as e:
        raise e # Re-raise HTTP exceptions directly
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")