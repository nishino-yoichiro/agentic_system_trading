"""
API Keys Setup Script
Helps you configure API keys for the data collection system
"""

import yaml
from pathlib import Path
import getpass

def setup_api_keys():
    """Interactive API keys setup"""
    print("=" * 50)
    print("API KEYS SETUP")
    print("=" * 50)
    print()
    print("This script will help you set up API keys for data collection.")
    print("You can get free API keys from:")
    print("- NewsAPI: https://newsapi.org/register")
    print("- Polygon: https://polygon.io/pricing")
    print("- CoinGecko: https://www.coingecko.com/en/api")
    print()
    
    # Load existing config
    config_file = Path("config/api_keys_local.yaml")
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Create default config
        config = {
            'newsapi': {'api_key': '', 'base_url': 'https://newsapi.org/v2'},
            'polygon': {'api_key': '', 'base_url': 'https://api.polygon.io'},
            'coingecko': {'api_key': '', 'base_url': 'https://api.coingecko.com/api/v3'}
        }
    
    # Get API keys
    print("Enter your API keys (press Enter to skip):")
    print()
    
    # NewsAPI
    newsapi_key = input("NewsAPI key: ").strip()
    if newsapi_key:
        config['newsapi']['api_key'] = newsapi_key
        print("✓ NewsAPI key saved")
    else:
        print("⚠ NewsAPI key skipped (news collection will be limited)")
    
    print()
    
    # Polygon
    polygon_key = input("Polygon API key: ").strip()
    if polygon_key:
        config['polygon']['api_key'] = polygon_key
        print("✓ Polygon key saved")
    else:
        print("⚠ Polygon key skipped (stock data will be limited)")
    
    print()
    
    # CoinGecko
    coingecko_key = input("CoinGecko API key (optional): ").strip()
    if coingecko_key:
        config['coingecko']['api_key'] = coingecko_key
        print("✓ CoinGecko key saved")
    else:
        print("✓ CoinGecko key skipped (free tier will be used)")
    
    # Save config
    config['polygon']['base_url'] = 'https://api.polygon.io'
    config['newsapi']['base_url'] = 'https://newsapi.org/v2'
    config['coingecko']['base_url'] = 'https://api.coingecko.com/api/v3'
    
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print()
    print("=" * 50)
    print("API KEYS SAVED SUCCESSFULLY!")
    print("=" * 50)
    print(f"Config saved to: {config_file}")
    print()
    print("You can now run:")
    print("  python quick_setup.py --days 365")
    print()

if __name__ == "__main__":
    setup_api_keys()