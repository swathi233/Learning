# src/utils.py
import os
import json
import pickle
import shutil
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import numpy as np
from PIL import Image
import io
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and processing of product data"""
    
    @staticmethod
    def load_sample_products() -> List[Dict[str, Any]]:
        """Load sample product data"""
        products = [
            {
                "id": "1",
                "name": "Sony WH-1000XM4 Wireless Headphones",
                "description": "Industry-leading noise cancellation with Dual Noise Sensor technology",
                "price": 349.99,
                "category": "Electronics",
                "brand": "Sony",
                "rating": 4.8,
                "stock": 50,
                "image_url": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=400",
                "specs": {
                    "color": "black",
                    "weight": "254g",
                    "battery": "30 hours",
                    "bluetooth": "5.0",
                    "noise_cancellation": "Yes"
                }
            },
            {
                "id": "2",
                "name": "Nike Air Zoom Pegasus 38",
                "description": "Responsive running shoes with Zoom Air cushioning",
                "price": 119.99,
                "category": "Sports",
                "brand": "Nike",
                "rating": 4.6,
                "stock": 100,
                "image_url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400",
                "specs": {
                    "size": "7-13",
                    "color": "blue/white",
                    "material": "mesh",
                    "weight": "285g",
                    "gender": "Unisex"
                }
            },
            {
                "id": "3",
                "name": "Hydro Flask Wide Mouth Water Bottle",
                "description": "Insulated stainless steel water bottle keeps drinks cold for 24 hours",
                "price": 44.95,
                "category": "Outdoor",
                "brand": "Hydro Flask",
                "rating": 4.7,
                "stock": 200,
                "image_url": "https://images.unsplash.com/photo-1523362628745-0c100150b504?w=400",
                "specs": {
                    "capacity": "32oz",
                    "color": "stainless",
                    "material": "stainless steel",
                    "lid_type": "wide mouth",
                    "insulation": "double wall"
                }
            },
            {
                "id": "4",
                "name": "Apple MacBook Pro 14-inch",
                "description": "Professional laptop with M1 Pro chip and Liquid Retina XDR display",
                "price": 1999.00,
                "category": "Computers",
                "brand": "Apple",
                "rating": 4.9,
                "stock": 25,
                "image_url": "https://images.unsplash.com/photo-1517336714731-489689fd1ca8?w=400",
                "specs": {
                    "processor": "M1 Pro",
                    "ram": "16GB",
                    "storage": "512GB SSD",
                    "display": "14.2-inch",
                    "battery": "17 hours"
                }
            },
            {
                "id": "5",
                "name": "Levi's 501 Original Fit Jeans",
                "description": "Classic straight leg jeans in authentic denim",
                "price": 89.50,
                "category": "Fashion",
                "brand": "Levi's",
                "rating": 4.5,
                "stock": 150,
                "image_url": "https://images.unsplash.com/photo-1542272604-787c3835535d?w=400",
                "specs": {
                    "size": "28-38",
                    "color": "blue",
                    "material": "100% cotton",
                    "fit": "straight",
                    "closure": "button fly"
                }
            },
            {
                "id": "6",
                "name": "Instant Pot Duo 7-in-1",
                "description": "Electric pressure cooker that replaces 7 kitchen appliances",
                "price": 99.95,
                "category": "Home",
                "brand": "Instant Pot",
                "rating": 4.7,
                "stock": 75,
                "image_url": "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=400",
                "specs": {
                    "capacity": "6 quarts",
                    "functions": "7-in-1",
                    "material": "stainless steel",
                    "programs": "14",
                    "warranty": "1 year"
                }
            }
        ]
        
        return products
    
    @staticmethod
    def load_products_from_json(file_path: str) -> List[Dict[str, Any]]:
        """Load products from JSON file"""
        try:
            with open(file_path, 'r') as f:
                products = json.load(f)
            
            logger.info(f"Loaded {len(products)} products from {file_path}")
            return products
            
        except Exception as e:
            logger.error(f"Error loading products from JSON: {e}")
            return []
    
    @staticmethod
    def save_products_to_json(products: List[Dict[str, Any]], 
                             file_path: str):
        """Save products to JSON file"""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(products, f, indent=2)
            
            logger.info(f"Saved {len(products)} products to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving products to JSON: {e}")

class ImageUtils:
    """Utilities for image processing"""
    
    @staticmethod
    def load_image(image_path: str) -> Optional[Image.Image]:
        """Load image from file path"""
        try:
            return Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    @staticmethod
    def save_image(image: Image.Image, image_path: str):
        """Save image to file path"""
        try:
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            image.save(image_path)
            logger.info(f"Image saved to {image_path}")
        except Exception as e:
            logger.error(f"Error saving image: {e}")
    
    @staticmethod
    def image_to_base64(image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        try:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return img_str
        except Exception as e:
            logger.error(f"Error converting image to base64: {e}")
            return ""
    
    @staticmethod
    def base64_to_image(base64_str: str) -> Optional[Image.Image]:
        """Convert base64 string to PIL Image"""
        try:
            img_data = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(img_data))
            return image.convert('RGB')
        except Exception as e:
            logger.error(f"Error converting base64 to image: {e}")
            return None
    
    @staticmethod
    def download_image(url: str, save_path: str) -> bool:
        """Download image from URL"""
        try:
            import requests
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            
            logger.info(f"Image downloaded to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading image: {e}")
            return False

class CacheManager:
    """Manages caching of embeddings and results"""
    
    def __init__(self, cache_dir: str = "./data/cache"):
        """Initialize cache manager"""
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def save_embeddings(self, embeddings: np.ndarray, key: str):
        """Save embeddings to cache"""
        cache_path = os.path.join(self.cache_dir, f"{key}_embeddings.pkl")
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embeddings, f)
            
            logger.debug(f"Embeddings saved to cache: {key}")
            
        except Exception as e:
            logger.error(f"Error saving embeddings to cache: {e}")
    
    def load_embeddings(self, key: str) -> Optional[np.ndarray]:
        """Load embeddings from cache"""
        cache_path = os.path.join(self.cache_dir, f"{key}_embeddings.pkl")
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                embeddings = pickle.load(f)
            
            logger.debug(f"Embeddings loaded from cache: {key}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error loading embeddings from cache: {e}")
            return None
    
    def save_results(self, results: Any, key: str):
        """Save search results to cache"""
        cache_path = os.path.join(self.cache_dir, f"{key}_results.pkl")
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(results, f)
            
            logger.debug(f"Results saved to cache: {key}")
            
        except Exception as e:
            logger.error(f"Error saving results to cache: {e}")
    
    def load_results(self, key: str) -> Any:
        """Load search results from cache"""
        cache_path = os.path.join(self.cache_dir, f"{key}_results.pkl")
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                results = pickle.load(f)
            
            logger.debug(f"Results loaded from cache: {key}")
            return results
            
        except Exception as e:
            logger.error(f"Error loading results from cache: {e}")
            return None

class Config:
    """Application configuration"""
    
    # Database
    PERSIST_DIRECTORY = "./data/chroma_db"
    
    # Embeddings
    TEXT_MODEL = "all-MiniLM-L6-v2"
    IMAGE_MODEL = "resnet50"
    
    # Search
    DEFAULT_RESULTS = 6
    SIMILARITY_THRESHOLD = 0.3
    
    # UI
    ITEMS_PER_PAGE = 6
    ENABLE_IMAGE_SEARCH = True
    ENABLE_FILTERS = True
    
    # Cache
    CACHE_DIR = "./data/cache"
    CACHE_TTL = 3600  # 1 hour
    
    @classmethod
    def to_dict(cls):
        """Convert config to dictionary"""
        return {key: value for key, value in cls.__dict__.items() 
                if not key.startswith('_') and not callable(value)}