#!/usr/bin/env python3
"""
Test Script for Interactive Trading Module
=========================================

Tests the interactive trading module components without requiring full setup.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from src.interactive_trading_module import InteractiveTradingModule
        print("[OK] InteractiveTradingModule imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import InteractiveTradingModule: {e}")
        return False
    
    try:
        from src.crypto_signal_integration import CryptoSignalIntegration
        print("[OK] CryptoSignalIntegration imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import CryptoSignalIntegration: {e}")
        return False
    
    try:
        from src.data_ingestion.websocket_price_feed import WebSocketPriceFeed
        print("[OK] WebSocketPriceFeed imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import WebSocketPriceFeed: {e}")
        return False
    
    return True

def test_dependencies():
    """Test optional dependencies"""
    print("\nTesting optional dependencies...")
    
    # Test pygame
    try:
        import pygame
        print("[OK] pygame available - sound notifications enabled")
        pygame_available = True
    except ImportError:
        print("[WARN] pygame not available - sound notifications disabled")
        pygame_available = False
    
    # Test plyer
    try:
        from plyer import notification
        print("[OK] plyer available - desktop notifications enabled")
        plyer_available = True
    except ImportError:
        print("[WARN] plyer not available - desktop notifications disabled")
        plyer_available = False
    
    return pygame_available, plyer_available

def test_signal_framework():
    """Test signal framework initialization"""
    print("\nTesting signal framework...")
    
    try:
        from src.crypto_signal_integration import CryptoSignalIntegration
        
        # Initialize with minimal setup
        integration = CryptoSignalIntegration(selected_strategies=['btc_ny_session'])
        
        print(f"[OK] Signal framework initialized with {len(integration.framework.strategies)} strategies")
        
        # List available strategies
        strategies = list(integration.framework.strategies.keys())
        print(f"   Available strategies: {', '.join(strategies)}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize signal framework: {e}")
        return False

def test_data_files():
    """Test for required data files"""
    print("\nTesting data files...")
    
    data_dir = Path("data")
    required_files = [
        "BTC_1m_historical.parquet",
        "ETH_1m_historical.parquet"
    ]
    
    found_files = []
    missing_files = []
    
    for file in required_files:
        file_path = data_dir / file
        if file_path.exists():
            found_files.append(file)
            print(f"[OK] Found {file}")
        else:
            missing_files.append(file)
            print(f"[WARN] Missing {file}")
    
    if missing_files:
        print(f"\n[INFO] To generate missing data files, run:")
        print(f"   python main.py data-collection")
    
    return len(found_files) > 0

def test_interactive_module():
    """Test interactive module initialization"""
    print("\nTesting interactive module...")
    
    try:
        from src.interactive_trading_module import InteractiveTradingModule
        
        # Initialize module
        module = InteractiveTradingModule()
        
        print("[OK] InteractiveTradingModule initialized successfully")
        print(f"   Data directory: {module.data_dir}")
        print(f"   Sound system enabled: {module.sound_system.enabled}")
        print(f"   Notification system enabled: {module.notification_system.enabled}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize InteractiveTradingModule: {e}")
        return False

def main():
    """Run all tests"""
    print("Interactive Trading Module Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Dependency Tests", test_dependencies),
        ("Signal Framework Tests", test_signal_framework),
        ("Data File Tests", test_data_files),
        ("Interactive Module Tests", test_interactive_module)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"[OK] {test_name} PASSED")
            else:
                print(f"[FAIL] {test_name} FAILED")
        except Exception as e:
            print(f"[ERROR] {test_name} ERROR: {e}")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] All tests passed! Interactive trading module is ready to use.")
        print("\nTo start the module, run:")
        print("   python main.py interactive-trading")
    else:
        print("[WARN] Some tests failed. Please check the errors above.")
        print("\nTo fix common issues:")
        print("   1. Install missing dependencies: python scripts/setup_interactive_trading.py")
        print("   2. Generate data files: python main.py data-collection")
        print("   3. Check logs for detailed error information")

if __name__ == "__main__":
    main()
