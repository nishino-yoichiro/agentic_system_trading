"""
Schwab Options Data Manager
===========================

Optional module for fetching and managing options chain data from Schwab.
Provides Greeks, IV, open interest, and volume for options analysis.

Features:
- Options chain retrieval (calls/puts)
- Greeks calculation (delta, gamma, vega, theta)
- Implied volatility tracking
- Open interest and volume monitoring
- Daily snapshots for backtesting
- Real-time streaming for intraday updates
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
import pandas as pd
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class OptionsDataManager:
    """
    Manages options chain data from Schwab.
    
    Features:
    - Fetch options chain for a symbol
    - Track Greeks (delta, gamma, vega, theta)
    - Monitor implied volatility
    - Store daily snapshots for backtesting
    """
    
    def __init__(self, client, symbol: str, storage_path: Optional[str] = None):
        """
        Initialize options data manager.
        
        Args:
            client: SchwabClient instance
            symbol: Underlying symbol (e.g., 'AAPL')
            storage_path: Optional path for storing options data
        """
        self.client = client
        self.symbol = symbol
        
        if storage_path is None:
            storage_dir = Path("data") / "options_data"
            storage_dir.mkdir(exist_ok=True, parents=True)
            storage_path = str(storage_dir)
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True, parents=True)
    
    async def get_options_chain(
        self,
        expiration_date: Optional[str] = None,
        strike_count: int = 50,
        include_quotes: bool = True
    ) -> pd.DataFrame:
        """
        Fetch options chain for the underlying symbol.
        
        Args:
            expiration_date: Target expiration date (YYYY-MM-DD). If None, returns nearest expiration
            strike_count: Number of strikes above and below ATM
            include_quotes: If True, include bid/ask quotes
        
        Returns:
            DataFrame with options chain data
        """
        try:
            def fetch_chain():
                """Synchronous fetch function using Schwabdev's option_chains endpoint (safe signature)."""
                try:
                    if hasattr(self.client, "option_chains"):
                        # Keep only the args guaranteed to exist
                        response = self.client.option_chains(
                            symbol=self.symbol,
                            contractType="ALL",       # Both calls and puts
                            strikeCount=strike_count  # ±N strikes around ATM
                        )

                        # Decode if it's a requests.Response
                        if hasattr(response, "json"):
                            data = response.json()
                        else:
                            data = response

                        return data

                    elif hasattr(self.client, "option_expiration_chain"):
                        base = self.client.option_expiration_chain(symbol=self.symbol)
                        logger.info(f"Fetched expiration chain fallback: {list(base.keys())[:5]}")
                        return base

                    raise AttributeError("No supported options endpoint found on Schwab client.")
                except Exception as e:
                    logger.error(f"Error in fetch_chain: {e}")
                    raise

            response = await asyncio.to_thread(fetch_chain)
            
            # Parse response into DataFrame
            options_data = []
            
            if response:
                # Extract chain data
                if isinstance(response, dict):
                    call_exp_date_map = response.get('callExpDateMap', {})
                    put_exp_date_map = response.get('putExpDateMap', {})
                elif hasattr(response, 'callExpDateMap'):
                    call_exp_date_map = response.callExpDateMap
                    put_exp_date_map = response.putExpDateMap
                else:
                    call_exp_date_map = {}
                    put_exp_date_map = {}
                
                # Process calls
                for exp_date, strikes in call_exp_date_map.items():
                    for strike, contracts in strikes.items():
                        if contracts and len(contracts) > 0:
                            contract = contracts[0]  # Take first contract
                            options_data.append(self._parse_option_contract(contract, 'CALL', exp_date, strike))
                
                # Process puts
                for exp_date, strikes in put_exp_date_map.items():
                    for strike, contracts in strikes.items():
                        if contracts and len(contracts) > 0:
                            contract = contracts[0]  # Take first contract
                            options_data.append(self._parse_option_contract(contract, 'PUT', exp_date, strike))
            
            if options_data:
                df = pd.DataFrame(options_data)
                return df
            else:
                logger.warning(f"No options data found for {self.symbol}")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching options chain: {e}")
            return pd.DataFrame()
    
    def _parse_option_contract(
        self,
        contract: Dict[str, Any],
        contract_type: str,
        expiration_date: str,
        strike: str
    ) -> Dict[str, Any]:
        """Parse a single option contract into standardized format"""
        try:
            # Extract Greeks
            greeks = contract.get('greeks', {}) if isinstance(contract, dict) else getattr(contract, 'greeks', {})
            
            # Extract quote
            quote = contract.get('quote', {}) if isinstance(contract, dict) else getattr(contract, 'quote', {})
            
            # Parse strike and expiration
            strike_price = float(strike)
            exp_date = pd.to_datetime(expiration_date.split(':')[0])  # Handle format like "2025-01-17:0"
            
            return {
                'symbol': self.symbol,
                'expiration': exp_date,
                'strike': strike_price,
                'type': contract_type,
                'bid': float(quote.get('bid', contract.get('bid', 0))) if quote else float(contract.get('bid', 0)),
                'ask': float(quote.get('ask', contract.get('ask', 0))) if quote else float(contract.get('ask', 0)),
                'iv': float(greeks.get('iv', contract.get('impliedVolatility', 0))) if greeks else float(contract.get('impliedVolatility', 0)),
                'delta': float(greeks.get('delta', 0)) if greeks else 0.0,
                'gamma': float(greeks.get('gamma', 0)) if greeks else 0.0,
                'vega': float(greeks.get('vega', 0)) if greeks else 0.0,
                'theta': float(greeks.get('theta', 0)) if greeks else 0.0,
                'oi': float(contract.get('openInterest', contract.get('open_interest', 0))),
                'volume': float(contract.get('totalVolume', contract.get('volume', 0))),
                'timestamp': datetime.now(timezone.utc)
            }
        except Exception as e:
            logger.error(f"Error parsing option contract: {e}")
            return {}
    
    async def save_daily_snapshot(self, chain_df: pd.DataFrame):
        """
        Save daily options chain snapshot for backtesting.
        
        Args:
            chain_df: Options chain DataFrame
        """
        try:
            if chain_df.empty:
                return
            
            # Save to date-based file
            date_str = datetime.now().strftime('%Y%m%d')
            snapshot_file = self.storage_path / f"{self.symbol}_options_{date_str}.parquet"
            
            chain_df.to_parquet(snapshot_file)
            logger.info(f"Saved options snapshot for {self.symbol} on {date_str}")
            
        except Exception as e:
            logger.error(f"Error saving options snapshot: {e}")
    
    def load_snapshot(self, date: datetime) -> Optional[pd.DataFrame]:
        """
        Load a historical options snapshot.
        
        Args:
            date: Date to load
        
        Returns:
            DataFrame with options chain, or None if not found
        """
        try:
            date_str = date.strftime('%Y%m%d')
            snapshot_file = self.storage_path / f"{self.symbol}_options_{date_str}.parquet"
            
            if snapshot_file.exists():
                df = pd.read_parquet(snapshot_file)
                return df
            else:
                logger.info(f"No options snapshot found for {self.symbol} on {date_str}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading options snapshot: {e}")
            return None
    
    async def stream_options_updates(
        self,
        symbols: List[str],
        on_update: callable
    ):
        """
        Stream real-time options updates.
        
        Args:
            symbols: List of option symbols to subscribe to
            on_update: Callback function for updates
        """
        # Implementation would depend on Schwab's streaming API for options
        # This is a placeholder for future implementation
        logger.info(f"Options streaming not yet implemented for {self.symbol}")
        pass

