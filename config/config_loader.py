"""
Configuration loader for Geo3DVQA.

This module provides centralized configuration management using YAML files.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """
    Configuration loader that reads from YAML files and environment variables.
    
    Environment variables take precedence over config file values.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration loader.
        
        Args:
            config_path: Path to config YAML file. If None, looks for config.yaml
                         in the config directory or current directory.
        """
        if config_path is None:
            # Try to find config.yaml in common locations
            possible_paths = [
                Path(__file__).parent / "config.yaml",
                Path.cwd() / "config.yaml",
                Path.cwd() / "config" / "config.yaml",
            ]
            
            config_path = None
            for path in possible_paths:
                if path.exists():
                    config_path = str(path)
                    break
        
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        
        if config_path and os.path.exists(config_path):
            self._load_from_file(config_path)
        else:
            # Use defaults if no config file found
            self.config = self._get_defaults()
    
    def _load_from_file(self, config_path: str):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Could not load config file {config_path}: {e}")
            print("Using default configuration.")
            self.config = self._get_defaults()
    
    def _get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            "random_seed": int(os.getenv("RANDOM_SEED", 42)),
            "processing": {
                "mode": "both",
                "num_processes": 0,
                "batch_size": 8,
                "max_files": 3000,
                "qa_multiplier": 15,
                "hard_ratio": 0.3,
            },
            "inference": {
                "model": "gpt-4o",
                "temperature": 0.0,
                "max_new_tokens": 512,
                "batch_size": 4,
            },
            "logging": {
                "level": "INFO",
                "console": True,
            },
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Example:
            config.get("processing.mode")  # Returns "both"
            config.get("processing.batch_size", 10)  # Returns batch_size or 10
        
        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                # Check environment variable override
                env_key = key.upper().replace('.', '_')
                env_value = os.getenv(env_key)
                if env_value is not None:
                    # Try to convert to appropriate type
                    try:
                        if env_value.lower() in ('true', 'false'):
                            return env_value.lower() == 'true'
                        try:
                            return int(env_value)
                        except ValueError:
                            try:
                                return float(env_value)
                            except ValueError:
                                return env_value
                    except Exception:
                        return env_value
                
                return default
        
        return value
    
    def get_path(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get path configuration value and expand user/home directory.
        
        Args:
            key: Configuration key
            default: Default path if key not found
            
        Returns:
            Expanded path string or None
        """
        path = self.get(key, default)
        if path:
            return os.path.expanduser(os.path.expandvars(path))
        return None
    
    def update(self, key: str, value: Any):
        """
        Update configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: New value
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary."""
        return self.config.copy()


# Global configuration instance
_global_config: Optional[ConfigLoader] = None


def get_config(config_path: Optional[str] = None) -> ConfigLoader:
    """
    Get or create the global configuration loader.
    
    Args:
        config_path: Path to config file (only used on first call)
        
    Returns:
        Global ConfigLoader instance
    """
    global _global_config
    if _global_config is None:
        _global_config = ConfigLoader(config_path)
    return _global_config


def reload_config(config_path: Optional[str] = None):
    """
    Reload configuration from file.
    
    Args:
        config_path: Path to config file (uses default if None)
    """
    global _global_config
    _global_config = ConfigLoader(config_path)

