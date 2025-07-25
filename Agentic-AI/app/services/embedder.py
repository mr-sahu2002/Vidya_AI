import requests
import logging
from typing import List

JINA_API_KEY = "jina_16c2883ab0a04e7292f7b3d56ba6c387LOXnRrP073vyx6ZpVe0CXEe7c9I3"  
JINA_ENDPOINT = "https://api.jina.ai/v1/embeddings"

def get_jina_embeddings(texts: List[str], task: str = "text-matching") -> List[List[float]]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}"
    }

    payload = {
        "model": "jina-embeddings-v3",
        "task": task,
        "input": texts
    }

    try:
        response = requests.post(JINA_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]
    except Exception as e:
        logging.error(f"Jina error: {str(e)}")
        raise

def get_single_jina_embedding(text: str, task: str = "text-matching") -> List[float]:
    return get_jina_embeddings([text], task)[0]