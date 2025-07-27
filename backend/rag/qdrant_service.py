import uuid
import logging
from datetime import datetime
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchValue
)
import firebase_admin
from firebase_admin import firestore,credentials

# Initialize Firebase if not already initialized
if not firebase_admin._apps:
    # Initialize with service account key
    cred = credentials.Certificate("C:\\Users\\asp00\\Downloads\\vidyaai-1dcfa-firebase-adminsdk-fbsvc-513f50d8d1.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()
qdrant_client = QdrantClient(host="localhost", port=6333)

# =====================
# CLASS MANAGEMENT
# =====================
def create_virtual_class(teacher_id: str, class_name: str, subject: str) -> Dict:
    """Create a new virtual class with unique collection"""
    try:
        # Generate unique collection name
        collection_name = f"class_{teacher_id}_{uuid.uuid4().hex[:8]}_{subject.lower()}"
        
        # Create collection in Qdrant
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        create_payload_indexes(collection_name)
        
        return {
            "status": "success",
            "collection": collection_name,
            "subject": subject
        }
    except Exception as e:
        logging.error(f"Class creation failed: {str(e)}")
        return {"status": "error", "message": str(e)}

def get_class_collection(class_id: str) -> str:
    """Get collection name for a class from Firestore"""
    try:
        class_doc = db.collection('classes').document(class_id).get()
        if class_doc.exists:
            return class_doc.to_dict().get('qdrant_collection', '')
        return ""
    except Exception as e:
        logging.error(f"Firestore access failed: {str(e)}")
        return ""

# =====================
# COLLECTION MANAGEMENT
# =====================
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
    
    points = []
    for chunk, embedding in zip(chunks, embeddings):
        # Generate deterministic UUID
        unique_str = f"{class_id}_{chunk['filename']}_{chunk['page_number']}_{chunk['chunk_index']}"
        point_id = uuid.uuid5(uuid.NAMESPACE_DNS, unique_str)
        
        points.append(PointStruct(
            id=str(point_id),
            vector=embedding,
            payload={
                **chunk,
                "class_id": class_id  # Critical for filtering
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

def search_class_content(
    class_id: str, 
    query_embedding: List[float], 
    filters: Optional[Dict] = None, 
    limit: int = 5
) -> List[Dict]:
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
    
    search_filter = Filter(must=must_conditions) if must_conditions else None
    
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
            "vectors": info.config.params.vectors
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}