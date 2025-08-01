import requests
import logging
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

JINA_API_KEY = os.getenv('EMBEDDING_KEY')
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