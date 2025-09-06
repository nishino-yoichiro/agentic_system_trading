#!/usr/bin/env python3
"""
Test script to verify API key loading
"""

import os
import yaml
from dotenv import load_dotenv

def test_api_key_loading():
    """Test API key loading from various sources"""
    print("Testing API key loading...")
    
    # Load environment variables from current directory
    load_dotenv('.env')
    
    # Check environment variable
    newsapi_key = os.getenv('NEWSAPI_KEY')
    print(f"NEWSAPI_KEY from env: {newsapi_key[:10] if newsapi_key else 'None'}...")
    
    # Check local config file
    local_config_path = "config/api_keys_local.yaml"
    if os.path.exists(local_config_path):
        try:
            with open(local_config_path, 'r') as f:
                local_config = yaml.safe_load(f)
                print(f"Local config keys: {list(local_config.keys())}")
                if 'newsapi' in local_config:
                    print(f"NewsAPI key from local config: {local_config['newsapi'][:10]}...")
        except Exception as e:
            print(f"Error loading local config: {e}")
    else:
        print("No local config file found")
    
    # Check if key looks valid
    if newsapi_key:
        if newsapi_key == 'api_key':
            print("❌ ERROR: API key is still the placeholder 'api_key'")
            print("Please set your actual NewsAPI key in .env file or config/api_keys_local.yaml")
        elif len(newsapi_key) < 20:
            print("❌ WARNING: API key seems too short")
        else:
            print("✅ API key looks valid")
    else:
        print("❌ ERROR: No API key found")

if __name__ == "__main__":
    test_api_key_loading()
