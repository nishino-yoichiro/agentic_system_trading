"""
Test script for BTC Daily Brief
"""

import asyncio
from pathlib import Path
from reports.btc_daily_brief import BTCDailyBriefGenerator

async def main():
    """Test BTC Daily Brief generation"""
    print("🚀 Generating BTC Daily Brief...")
    
    generator = BTCDailyBriefGenerator()
    timestamp = generator.generate_daily_brief(days_back=7)
    
    if timestamp:
        print(f"✅ BTC Daily Brief generated successfully!")
        print(f"📁 Check reports/btc_briefs/ for the generated files")
        print(f"🕐 Timestamp: {timestamp}")
    else:
        print("❌ Failed to generate BTC Daily Brief")

if __name__ == "__main__":
    asyncio.run(main())
