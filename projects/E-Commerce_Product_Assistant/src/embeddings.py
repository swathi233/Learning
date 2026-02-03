# src/embeddings.py (Updated)
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from typing import Union, List, Optional
import logging
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.WARNING)  # Reduce verbosity
logger = logging.getLogger(__name__)

class TextEmbedder:
    """Handles text embedding using Sentence Transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize text embedding model"""
        logger.info(f"Initializing text embedder with model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"✅ Text embedder initialized")
        except Exception as e:
            logger.error(f"❌ Error initializing text embedder: {e}")
            raise
    
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text"""
        try:
            if isinstance(text, str):
                text = [text]
            
            embeddings = self.model.encode(text, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            return np.random.randn(384)  # Fallback
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for batch of texts"""
        try:
            embeddings = self.model.encode(
                texts, 
                batch_size=batch_size, 
                convert_to_numpy=True,
                show_progress_bar=False  # Hide progress bar
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding batch: {e}")
            return np.random.randn(len(texts), 384)

class LightImageEmbedder:
    """Lightweight image embedder for demo purposes"""
    
    def __init__(self):
        """Initialize lightweight image embedding model"""
        logger.info("Initializing lightweight image embedder")
        self.device = torch.device("cpu")
        
        # Use a VERY lightweight model or mock for demo
        try:
            # Try to load a small model
            self.model = models.resnet18(pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.model.eval()
            self.model.to(self.device)
            logger.info("✅ Image embedder initialized with ResNet18")
        except:
            # Fallback to mock model
            self.model = None
            logger.warning("⚠️ Using mock image embedder (for demo)")
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Smaller size
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def embed_image(self, image_path: str) -> Optional[np.ndarray]:
        """Generate embedding for a single image"""
        try:
            if self.model is None:
                # Return random vector for demo
                return np.random.randn(512)
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                embedding = self.model(image_tensor)
                embedding = embedding.squeeze().cpu().numpy()
            
            # Flatten and normalize
            embedding = embedding.flatten()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error embedding image {image_path}: {e}")
            return np.random.randn(512)  # Fallback
    
    def embed_image_pil(self, image: Image.Image) -> Optional[np.ndarray]:
        """Generate embedding from PIL Image"""
        try:
            if self.model is None:
                return np.random.randn(512)
            
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model(image_tensor)
                embedding = embedding.squeeze().cpu().numpy()
            
            embedding = embedding.flatten()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error embedding PIL image: {e}")
            return np.random.randn(512)

# Singleton instances
text_embedder = TextEmbedder()
image_embedder = LightImageEmbedder()  # Changed to lightweight