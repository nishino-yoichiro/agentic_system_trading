"""
API Utilities
Common functions for API key loading and management
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def load_api_keys() -> Dict[str, str]:
    """Load API keys from .env file"""
    try:
        from dotenv import load_dotenv
        import os
        
        # Load .env file
        load_dotenv()
        
        # Extract API keys from environment variables
        api_keys = {
            'newsapi': os.getenv('NEWSAPI_KEY', ''),
            'polygon': os.getenv('POLYGON_API_KEY', ''),
            'coingecko': os.getenv('COINGECKO_API_KEY', ''),
            'binance': os.getenv('BINANCE_API_KEY', ''),
            'binance_secret': os.getenv('BINANCE_SECRET_KEY', '')
        }
        
        # Filter out empty keys
        api_keys = {k: v for k, v in api_keys.items() if v}
        
        logger.info(f"Loaded API keys for: {list(api_keys.keys())}")
        return api_keys
        
    except Exception as e:
        logger.error(f"Error loading API keys from .env: {e}")
        return {}

def get_api_key(api_keys: Dict[str, Any], service: str) -> str:
    """Get API key for a specific service"""
    return api_keys.get(service, "")

def validate_api_keys(api_keys: Dict[str, Any]) -> Dict[str, bool]:
    """Validate that required API keys are present"""
    required_keys = ['newsapi', 'polygon', 'coingecko']
    validation = {}
    
    for key in required_keys:
        validation[key] = bool(api_keys.get(key))
    
    return validation
