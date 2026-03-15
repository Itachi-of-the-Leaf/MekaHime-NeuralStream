import chromadb
from chromadb.config import Settings
import numpy as np
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class SpeakerDB:
    """
    Persistent Vector Database for Speaker Voiceprints using ChromaDB.
    Strictly follows Rule 4 (Cosine Space) and Rule 10 (Confidence Mapping).
    """
    def __init__(self, persist_directory: str = "./vector_store"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Rule 4: Ensure metadata is strictly set to hnsw:space: cosine
        self.collection = self.client.get_or_create_collection(
            name="mekahime_voiceprints",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Initialized SpeakerDB at {persist_directory} with cosine space.")

    def add_voiceprint(self, speaker_id: str, embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        """
        Adds a pre-enrolled voiceprint embedding to the collection.
        embedding: NeMo TitaNet-L vector (list or numpy array)
        """
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
            
        if metadata is None:
            metadata = {"speaker_id": speaker_id}
            
        self.collection.add(
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[speaker_id]
        )
        logger.info(f"Added voiceprint for speaker: {speaker_id}")

    def map_confidence(self, distance: float) -> str:
        """
        Rule 10: Confidence-Based Identification mapping.
        < 0.15 = 'For sure'
        0.15 - 0.35 = 'Probably'
        > 0.35 = 'Likely' or 'Unknown'
        """
        if distance < 0.15:
            return "For sure"
        elif 0.15 <= distance <= 0.35:
            return "Probably"
        else:
            return "Likely" or "Unknown"

    def match_voiceprint(self, query_embedding: np.ndarray, n_results: int = 1) -> Dict[str, Any]:
        """
        Queries the collection with a new audio embedding.
        Returns the best match with speaker_id and confidence label.
        """
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
            
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        if not results['ids'] or not results['ids'][0]:
            return {"speaker_id": "Unknown", "confidence": "Unknown", "distance": None}
        
        # Get the top match
        speaker_id = results['ids'][0][0]
        distance = results['distances'][0][0]
        confidence_label = self.map_confidence(distance)
        
        # Rule 10: Output precise logging
        logger.info(f"Speaker Match: {speaker_id} | Distance: {distance:.4f} | Confidence: {confidence_label}")
        
        return {
            "speaker_id": speaker_id,
            "confidence": confidence_label,
            "distance": distance,
            "metadata": results['metadatas'][0][0]
        }
