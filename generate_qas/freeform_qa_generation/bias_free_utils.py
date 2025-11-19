"""
Bias-free utility functions for QA generation

This module provides utility functions to ensure bias-free QA generation,
following the project's requirements for unbiased VLM QA benchmarks.
"""

import numpy as np
import random
from typing import List, Any, Dict


def bias_free_shuffle(data: List[Any], seed: int = None) -> List[Any]:
    """
    Perform bias-free shuffling of data while preserving data structure
    
    Args:
        data: List of data to shuffle
        seed: Random seed for reproducibility (optional)
    
    Returns:
        Shuffled list
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Create a copy to avoid modifying original data
    shuffled_data = data.copy()
    
    # Use numpy's shuffle for consistency
    np.random.shuffle(shuffled_data)
    
    return shuffled_data


def remove_choice_bias(choices: List[str], correct_answer: str) -> tuple:
    """
    Remove choice-bias by shuffling choices and updating correct answer index
    
    Args:
        choices: List of answer choices
        correct_answer: The correct answer string
    
    Returns:
        Tuple of (shuffled_choices, new_correct_index)
    """
    if correct_answer not in choices:
        raise ValueError(f"Correct answer '{correct_answer}' not found in choices")
    
    # Find original index
    original_index = choices.index(correct_answer)
    
    # Create indexed choices for shuffling
    indexed_choices = list(enumerate(choices))
    
    # Shuffle the indexed choices
    shuffled_indexed = bias_free_shuffle(indexed_choices)
    
    # Extract shuffled choices and find new correct index
    shuffled_choices = [choice for _, choice in shuffled_indexed]
    new_correct_index = next(i for i, (orig_idx, _) in enumerate(shuffled_indexed) 
                           if orig_idx == original_index)
    
    return shuffled_choices, new_correct_index


def remove_coordinate_position_bias(coordinates: List[tuple], 
                                  associated_data: List[Any] = None) -> tuple:
    """
    Remove coordinate position bias by randomizing coordinate order
    
    Args:
        coordinates: List of coordinate tuples
        associated_data: Optional associated data to shuffle with coordinates
    
    Returns:
        Tuple of (shuffled_coordinates, shuffled_associated_data)
    """
    if associated_data and len(coordinates) != len(associated_data):
        raise ValueError("Coordinates and associated data must have same length")
    
    # Create indexed data for consistent shuffling
    if associated_data:
        combined_data = list(zip(coordinates, associated_data))
        shuffled_combined = bias_free_shuffle(combined_data)
        shuffled_coords, shuffled_data = zip(*shuffled_combined)
        return list(shuffled_coords), list(shuffled_data)
    else:
        return bias_free_shuffle(coordinates), None


def ensure_balanced_categories(qa_pairs: List[Dict[str, Any]], 
                             category_key: str = 'category') -> List[Dict[str, Any]]:
    """
    Ensure balanced representation of categories in QA pairs
    
    Args:
        qa_pairs: List of QA pair dictionaries
        category_key: Key used for category information
    
    Returns:
        Balanced list of QA pairs
    """
    if not qa_pairs:
        return qa_pairs
    
    # Group by category
    category_groups = {}
    for qa in qa_pairs:
        category = qa.get(category_key, 'unknown')
        if category not in category_groups:
            category_groups[category] = []
        category_groups[category].append(qa)
    
    # Find minimum category size
    min_size = min(len(group) for group in category_groups.values())
    
    # Balance categories
    balanced_pairs = []
    for category, group in category_groups.items():
        # Shuffle the group first
        shuffled_group = bias_free_shuffle(group)
        # Take up to min_size items
        balanced_pairs.extend(shuffled_group[:min_size])
    
    # Final shuffle to remove any ordering bias
    return bias_free_shuffle(balanced_pairs)


def validate_qa_pair_bias_free(qa_pair: Dict[str, Any]) -> bool:
    """
    Validate that a QA pair is bias-free
    
    Args:
        qa_pair: QA pair dictionary to validate
    
    Returns:
        True if bias-free, False otherwise
    """
    # Check for common bias indicators
    question = qa_pair.get('question', '').lower()
    answer = qa_pair.get('answer', '').lower()
    
    # Check for position-based bias words
    position_bias_words = ['first', 'second', 'third', 'left', 'right', 'top', 'bottom']
    if any(word in question for word in position_bias_words):
        return False
    
    # Check for leading answers (answers that start with obvious indicators)
    if answer.startswith(('a)', 'b)', 'c)', 'd)', '1)', '2)', '3)', '4)')):
        return False
    
    # Check for coordinate bias (specific coordinates mentioned)
    import re
    coord_pattern = r'\(\s*\d+\s*,\s*\d+\s*\)'
    if re.search(coord_pattern, question) or re.search(coord_pattern, answer):
        return False
    
    return True


def apply_bias_prevention_measures(qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply comprehensive bias prevention measures to QA pairs
    
    Args:
        qa_pairs: List of QA pair dictionaries
    
    Returns:
        Bias-free QA pairs
    """
    # Filter out biased QA pairs
    clean_pairs = [qa for qa in qa_pairs if validate_qa_pair_bias_free(qa)]
    
    # Balance categories
    balanced_pairs = ensure_balanced_categories(clean_pairs)
    
    # Final shuffle
    final_pairs = bias_free_shuffle(balanced_pairs)
    
    return final_pairs