"""
Feature Store for managing and storing engineered features
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from loguru import logger
import os
import pickle
from datetime import datetime, timedelta


@dataclass
class FeatureMetadata:
    """Metadata for a feature"""
    name: str
    description: str
    category: str
    created_at: datetime
    version: str
    dependencies: List[str]


class FeatureStore:
    """Store and manage engineered features"""
    
    def __init__(self, store_path: str = 'data/features'):
        self.store_path = store_path
        self.metadata = {}
        self._ensure_store_exists()
    
    def _ensure_store_exists(self):
        """Ensure feature store directory exists"""
        os.makedirs(self.store_path, exist_ok=True)
    
    def store_feature(self, name: str, data: pd.DataFrame, metadata: FeatureMetadata):
        """Store a feature with metadata"""
        try:
            # Save data
            data_path = os.path.join(self.store_path, f"{name}.parquet")
            data.to_parquet(data_path)
            
            # Save metadata
            metadata_path = os.path.join(self.store_path, f"{name}_metadata.pkl")
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            self.metadata[name] = metadata
            logger.info(f"Stored feature: {name}")
            
        except Exception as e:
            logger.error(f"Error storing feature {name}: {e}")
    
    def load_feature(self, name: str) -> Optional[pd.DataFrame]:
        """Load a feature from store"""
        try:
            data_path = os.path.join(self.store_path, f"{name}.parquet")
            if os.path.exists(data_path):
                return pd.read_parquet(data_path)
            else:
                logger.warning(f"Feature {name} not found in store")
                return None
        except Exception as e:
            logger.error(f"Error loading feature {name}: {e}")
            return None
    
    def list_features(self) -> List[str]:
        """List all available features"""
        features = []
        for file in os.listdir(self.store_path):
            if file.endswith('.parquet') and not file.endswith('_metadata.pkl'):
                features.append(file.replace('.parquet', ''))
        return features


class FeatureManager:
    """Manage feature lifecycle and dependencies"""
    
    def __init__(self, feature_store: FeatureStore):
        self.feature_store = feature_store
        self.dependencies = {}
    
    def add_dependency(self, feature: str, depends_on: List[str]):
        """Add dependency information for a feature"""
        self.dependencies[feature] = depends_on
    
    def get_feature_dependencies(self, feature: str) -> List[str]:
        """Get dependencies for a feature"""
        return self.dependencies.get(feature, [])
    
    def validate_dependencies(self, feature: str) -> bool:
        """Validate that all dependencies are available"""
        deps = self.get_feature_dependencies(feature)
        available_features = self.feature_store.list_features()
        
        for dep in deps:
            if dep not in available_features:
                logger.error(f"Dependency {dep} not available for feature {feature}")
                return False
        
        return True
