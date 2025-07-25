import uuid
import logging
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchValue
)

# Initialize client
qdrant_client = QdrantClient(host="localhost", port=6333)

# Persistent class registry
CLASS_REGISTRY_FILE = "class_registry.json"
CLASS_REGISTRY = {}

# Load existing registry on startup
def load_registry():
    global CLASS_REGISTRY
    try:
        if os.path.exists(CLASS_REGISTRY_FILE):
            with open(CLASS_REGISTRY_FILE, 'r') as f:
                CLASS_REGISTRY = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load class registry: {str(e)}")

# Save registry to file
def save_registry():
    try:
        with open(CLASS_REGISTRY_FILE, 'w') as f:
            json.dump(CLASS_REGISTRY, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to save class registry: {str(e)}")

# Load registry immediately on import
load_registry()

# =====================
# CLASS MANAGEMENT
# =====================

def create_virtual_class(teacher_id: str, class_name: str, subject: str) -> Dict:
    """Create a new virtual class with unique collection"""
    class_id = f"class_{teacher_id}_{uuid.uuid4().hex[:8]}"
    collection_name = f"{class_id}_{subject.lower()}"
    
    CLASS_REGISTRY[class_id] = {
        "teacher_id": teacher_id,
        "class_name": class_name,
        "subject": subject,
        "collection": collection_name,
        "created_at": datetime.now().isoformat()
    }
    
    save_registry()  # Save to file after modification
    
    return {
        "class_id": class_id,
        "collection": collection_name,
        "subject": subject
    }

def get_class_collection(class_id: str) -> str:
    """Get collection name for a class"""
    return CLASS_REGISTRY.get(class_id, {}).get("collection", "")

# =====================
# COLLECTION MANAGEMENT
# =====================

def ensure_collection(collection_name: str, vector_dim: int) -> None:
    """Create collection if not exists with proper indexes"""
    existing_collections = [col.name for col in qdrant_client.get_collections().collections]
    
    if collection_name not in existing_collections:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE))
        create_payload_indexes(collection_name)

def create_payload_indexes(collection_name: str):
    """Create indexes for efficient filtering"""
    for field in ["class_id", "subject", "content_type", "grade_level", "difficulty"]:
        try:
            qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema="keyword"
            )
        except Exception as e:
            logging.info(f"Index exists for {field}: {e}")

# =====================
# DATA OPERATIONS
# =====================

def store_chunks_in_class(class_id: str, chunks: List[Dict], embeddings: List[List[float]]) -> Dict:
    """Store chunks in class-specific collection"""
    collection_name = get_class_collection(class_id)
    if not collection_name:
        return {"status": "error", "message": "Invalid class ID"}
    
    ensure_collection(collection_name, len(embeddings[0]))
    
    points = []
    for chunk, embedding in zip(chunks, embeddings):
        # Generate deterministic UUID from unique content identifiers
        unique_str = f"{class_id}_{chunk['filename']}_{chunk['page_number']}_{chunk['chunk_index']}"
        point_id = uuid.uuid5(uuid.NAMESPACE_DNS, unique_str)
        
        points.append(PointStruct(
            id=str(point_id),
            vector=embedding,
            payload={
                **chunk,
                "class_id": class_id,  # Critical for filtering
                "stored_at": datetime.now().isoformat()
            }
        ))
    
    try:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True
        )
        return {"status": "success", "stored": len(points)}
    except Exception as e:
        logging.error(f"Storage failed: {str(e)}")
        return {"status": "error", "message": str(e)}

def search_class_content(class_id: str, query_embedding: List[float], 
                        filters: Optional[Dict] = None, limit: int = 5) -> List[Dict]:
    """Search within a specific class collection"""
    collection_name = get_class_collection(class_id)
    if not collection_name:
        return []
    
    # Always filter by class ID
    must_conditions = [FieldCondition(key="class_id", match=MatchValue(value=class_id))]
    
    # Add additional filters
    if filters:
        for field, value in filters.items():
            if value and field in ["subject", "content_type", "difficulty", "grade_level"]:
                must_conditions.append(
                    FieldCondition(key=field, match=MatchValue(value=value))
                )
    
    search_filter = Filter(must=must_conditions)
    
    try:
        results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=search_filter,
            limit=limit
        )
        
        return [
            {
                "text": hit.payload.get("text", ""),
                "score": hit.score,
                "metadata": {
                    k: hit.payload.get(k) 
                    for k in ["subject", "content_type", "filename", 
                             "page_number", "difficulty", "grade_level"]
                }
            }
            for hit in results
        ]
    except Exception as e:
        logging.error(f"Search error: {str(e)}")
        return []

def get_class_stats(class_id: str) -> Dict:
    """Get statistics about a class collection"""
    collection_name = get_class_collection(class_id)
    if not collection_name:
        return {"status": "error", "message": "Invalid class ID"}
    
    try:
        info = qdrant_client.get_collection(collection_name)
        return {
            "class_id": class_id,
            "collection": collection_name,
            "chunk_count": info.points_count,
            "status": "active",
            "vectors": info.config.params.vectors
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
        
def get_class_subject(class_id: str) -> Dict[str, any]:
    """Get subject from class registry, returning a dict with status."""
    class_info = CLASS_REGISTRY.get(class_id)
    if class_info:
        return {"status": "success", "subject": class_info.get("subject", "general")}
    else:
        return {"status": "error", "message": f"Class ID {class_id} not found."}