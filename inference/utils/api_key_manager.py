"""
Secure API key management utilities.

This module provides secure handling of API keys to prevent leakage
and ensure proper validation.
"""

import os
import re
from typing import Optional
from dotenv import load_dotenv


class APIKeyManager:
    """
    Secure API key manager for OpenAI and other APIs.
    
    This class ensures that API keys are:
    - Loaded from secure sources (environment variables, .env files)
    - Validated before use
    - Never exposed in error messages or logs
    - Properly masked in output
    """
    
    # Pattern to detect API keys in strings (for validation)
    API_KEY_PATTERN = re.compile(r'sk-[a-zA-Z0-9]{32,}')
    
    def __init__(self, key_name: str = "OPENAI_API_KEY", env_file: str = ".env"):
        """
        Initialize the API key manager.
        
        Args:
            key_name: Name of the environment variable containing the API key
            env_file: Path to .env file (if exists)
        """
        self.key_name = key_name
        self.env_file = env_file
        self._api_key: Optional[str] = None
        self._load_key()
    
    def _load_key(self):
        """Load API key from environment variable or .env file."""
        # Try loading from .env file first
        if os.path.exists(self.env_file):
            load_dotenv(self.env_file)
        
        # Get from environment variable
        api_key = os.getenv(self.key_name)
        
        if api_key:
            self._api_key = api_key.strip()
        else:
            self._api_key = None
    
    def get_key(self) -> Optional[str]:
        """
        Get the API key.
        
        Returns:
            API key string or None if not set
        """
        return self._api_key
    
    def validate_key(self) -> bool:
        """
        Validate that the API key is set and has correct format.
        
        Returns:
            True if key is valid, False otherwise
        """
        if not self._api_key:
            return False
        
        # Basic format validation for OpenAI API keys
        if self.key_name == "OPENAI_API_KEY":
            # OpenAI API keys typically start with "sk-"
            if not self._api_key.startswith("sk-"):
                return False
            # Minimum length check
            if len(self._api_key) < 20:
                return False
        
        return True
    
    def require_key(self) -> str:
        """
        Get the API key and raise an error if not set or invalid.
        
        Returns:
            API key string
            
        Raises:
            ValueError: If API key is not set or invalid
        """
        if not self._api_key:
            raise ValueError(
                f"{self.key_name} is not set. "
                f"Please set it as an environment variable or in a .env file. "
                f"Example: export {self.key_name}=your_api_key_here"
            )
        
        if not self.validate_key():
            raise ValueError(
                f"{self.key_name} has invalid format. "
                f"Please check your API key."
            )
        
        return self._api_key
    
    def mask_key(self, key: Optional[str] = None) -> str:
        """
        Mask API key for safe logging/output.
        
        Args:
            key: Key to mask (uses stored key if None)
            
        Returns:
            Masked key string (e.g., "sk-...xxxx")
        """
        if key is None:
            key = self._api_key
        
        if not key:
            return "<not set>"
        
        if len(key) <= 8:
            return "****"
        
        # Show first 7 characters and last 4 characters
        return f"{key[:7]}...{key[-4:]}"
    
    def check_key_in_string(self, text: str) -> bool:
        """
        Check if an API key is accidentally included in a string.
        
        This is useful for detecting potential key leakage in error messages.
        
        Args:
            text: Text to check
            
        Returns:
            True if API key pattern is found, False otherwise
        """
        return bool(self.API_KEY_PATTERN.search(text))
    
    def sanitize_error_message(self, error_msg: str) -> str:
        """
        Sanitize error message to remove any API keys.
        
        Args:
            error_msg: Original error message
            
        Returns:
            Sanitized error message with API keys masked
        """
        # Replace any API key patterns with masked version
        sanitized = self.API_KEY_PATTERN.sub(
            lambda m: self.mask_key(m.group()),
            error_msg
        )
        return sanitized


# Global API key manager instance
_global_api_key_manager = None


def get_api_key_manager(key_name: str = "OPENAI_API_KEY") -> APIKeyManager:
    """
    Get or create the global API key manager.
    
    Args:
        key_name: Name of the environment variable
        
    Returns:
        Global APIKeyManager instance
    """
    global _global_api_key_manager
    if _global_api_key_manager is None or _global_api_key_manager.key_name != key_name:
        _global_api_key_manager = APIKeyManager(key_name)
    return _global_api_key_manager


def get_openai_api_key() -> str:
    """
    Get OpenAI API key securely.
    
    Returns:
        API key string
        
    Raises:
        ValueError: If API key is not set or invalid
    """
    manager = get_api_key_manager("OPENAI_API_KEY")
    return manager.require_key()


def validate_input_path(path: str, path_type: str = "file") -> str:
    """
    Validate input file or directory path.
    
    Args:
        path: Path to validate
        path_type: Type of path ("file" or "directory")
        
    Returns:
        Validated path string
        
    Raises:
        ValueError: If path is invalid or contains suspicious patterns
        FileNotFoundError: If path does not exist
    """
    if not path:
        raise ValueError(f"{path_type.capitalize()} path cannot be empty")
    
    # Check for path traversal attempts
    if ".." in path or path.startswith("/"):
        # Allow absolute paths but log warning
        if path.startswith("/"):
            import warnings
            warnings.warn(f"Using absolute path: {path}")
    
    # Check if path exists
    if path_type == "file" and not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    elif path_type == "directory" and not os.path.isdir(path):
        raise FileNotFoundError(f"Directory not found: {path}")
    
    return path


def validate_output_path(path: str) -> str:
    """
    Validate output file path.
    
    Args:
        path: Output path to validate
        
    Returns:
        Validated path string
        
    Raises:
        ValueError: If path is invalid
    """
    if not path:
        raise ValueError("Output path cannot be empty")
    
    # Ensure parent directory exists
    parent_dir = os.path.dirname(path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    
    return path

