# src/database.py - SIMPLIFIED VERSION
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import json
import os
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class ProductDatabase:
    """Simplified product database"""
    
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB
        try:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Initialize embedding function
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            
            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name="products",
                embedding_function=self.embedding_function
            )
            
            logger.info(f"✅ Database initialized at {persist_directory}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ChromaDB: {e}")
            self.client = None
            self.collection = None
    
    def add_products(self, products: List[Dict[str, Any]]):
        """Add products to database"""
        if self.collection is None:
            logger.error("Database not initialized")
            return 0
        
        if not products:
            return 0
        
        documents = []
        metadatas = []
        ids = []
        
        for product in products:
            # Create text for embedding
            text = f"{product.get('name', '')} {product.get('description', '')} {product.get('category', '')}"
            
            # Prepare metadata
            metadata = {
                "name": product.get("name", ""),
                "description": product.get("description", ""),
                "price": float(product.get("price", 0)),
                "category": product.get("category", ""),
                "brand": product.get("brand", ""),
                "rating": float(product.get("rating", 0)),
                "image_url": product.get("image_url", ""),
                "specs": json.dumps(product.get("specs", {}))
            }
            
            documents.append(text)
            metadatas.append(metadata)
            ids.append(product.get("id", f"prod_{len(ids)}"))
        
        try:
            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(products)} products")
            return len(products)
            
        except Exception as e:
            logger.error(f"Error adding products: {e}")
            return 0
    
    def search_products(self, query: str, n_results: int = 5, 
                       filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Search products"""
        if self.collection is None:
            logger.error("Database not initialized")
            return {"query": query, "results": [], "total_results": 0}
        
        try:
            if filters:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=filters
                )
            else:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results
                )
            
            # Parse results
            parsed_results = []
            if results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    metadata = results['metadatas'][0][i]
                    parsed_results.append({
                        "id": results['ids'][0][i],
                        "name": metadata["name"],
                        "description": metadata["description"],
                        "price": metadata["price"],
                        "category": metadata["category"],
                        "brand": metadata["brand"],
                        "rating": metadata["rating"],
                        "image_url": metadata["image_url"],
                        "specs": json.loads(metadata["specs"]),
                        "similarity_score": 1 - results['distances'][0][i]
                    })
            
            return {
                "query": query,
                "results": parsed_results,
                "total_results": len(parsed_results)
            }
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return {"query": query, "results": [], "total_results": 0}

# Create instance
product_db = ProductDatabase()