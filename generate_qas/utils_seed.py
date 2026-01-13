"""
Seed management utilities for reproducibility.

This module provides centralized seed management to ensure reproducible results
across different parts of the codebase.
"""

import random
import numpy as np
import os


class SeedManager:
    """
    Centralized seed manager for reproducible random number generation.
    
    This class ensures that all random number generators (Python's random,
    NumPy's random) are initialized consistently across the codebase.
    """
    
    def __init__(self, seed: int = None):
        """
        Initialize the seed manager.
        
        Args:
            seed: Random seed value. If None, uses environment variable RANDOM_SEED
                  or defaults to 42.
        """
        if seed is None:
            seed = int(os.getenv('RANDOM_SEED', 42))
        
        self.seed = seed
        self._set_seed(seed)
    
    def _set_seed(self, seed: int):
        """Set seed for all random number generators."""
        random.seed(seed)
        np.random.seed(seed)
        # Set seed for other libraries if needed
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
    
    def reset(self, seed: int = None):
        """
        Reset seed to a new value.
        
        Args:
            seed: New seed value. If None, uses the original seed.
        """
        if seed is None:
            seed = self.seed
        self.seed = seed
        self._set_seed(seed)
    
    def get_seed(self) -> int:
        """Get the current seed value."""
        return self.seed
    
    def create_file_seed(self, file_path: str) -> int:
        """
        Create a deterministic seed based on file path.
        
        This ensures that the same file always gets the same seed,
        while different files get different seeds.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Deterministic seed value
        """
        file_hash = hash(os.path.basename(file_path)) % (2**32 - 1)
        return (self.seed + file_hash) % (2**32 - 1)


# Global seed manager instance
_global_seed_manager = None


def get_seed_manager(seed: int = None) -> SeedManager:
    """
    Get or create the global seed manager.
    
    Args:
        seed: Initial seed value (only used on first call)
        
    Returns:
        Global SeedManager instance
    """
    global _global_seed_manager
    if _global_seed_manager is None:
        _global_seed_manager = SeedManager(seed)
    return _global_seed_manager


def set_global_seed(seed: int = None):
    """
    Set the global random seed.
    
    This is a convenience function that initializes or resets the global seed manager.
    
    Args:
        seed: Random seed value. If None, uses environment variable RANDOM_SEED
              or defaults to 42.
    """
    manager = get_seed_manager(seed)
    manager.reset(seed)


def get_global_seed() -> int:
    """Get the current global seed value."""
    return get_seed_manager().get_seed()

