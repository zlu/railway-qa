import requests
import json
from typing import List

class OllamaEmbeddings:
    def __init__(self, model_name="bge-m3", base_url="http://localhost:11434/api"):
        self.model_name = model_name
        self.base_url = base_url
        self.embed_url = f"{base_url}/embeddings"
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using Ollama"""
        embeddings = []
        for text in texts:
            data = {
                "model": self.model_name,
                "prompt": text
            }
            try:
                response = requests.post(self.embed_url, json=data)
                response.raise_for_status()
                embedding = response.json().get("embedding", [])
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error getting embedding: {e}")
                # Return zero vector if there's an error
                embeddings.append([0.0] * 1024)  # bge-m3 uses 1024 dimensions
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query (same as document for bge-m3)"""
        return self.embed_documents([text])[0]
