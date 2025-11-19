import json
import numpy as np
import sys
import os
import random
import time

def tqdm_safe_print(*args, **kwargs):
    """Print wrapper safe for use with tqdm progress bars."""
    try:
        from tqdm import tqdm
        if tqdm.monitor:
            with tqdm.external_write_mode():
                print(*args, **kwargs)
                sys.stdout.flush()
            return
    except (ImportError, Exception):
        pass

    print(*args, **kwargs)
    sys.stdout.flush()


def write_json_line(file_path, data):
    """Write a single JSON object to a file, one object per line."""
    def numpy_encoder(obj):
        """JSON encoder for numpy types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    def convert_to_json_serializable(data):
        """Recursively convert data to JSON serializable format."""
        if isinstance(data, dict):
            return {k: convert_to_json_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [convert_to_json_serializable(i) for i in data]
        elif isinstance(data, tuple):
            return tuple(convert_to_json_serializable(i) for i in data)
        return data
        
    serializable_data = convert_to_json_serializable(data)
    try:
        with open(file_path, 'a') as f:
            json_str = json.dumps(serializable_data, default=numpy_encoder)
            f.write(json_str + '\n')
    except IOError as e:
        tqdm_safe_print(f"Error writing to {file_path}: {e}")
    except TypeError as e:
        tqdm_safe_print(f"Error serializing data to JSON: {e}")
        tqdm_safe_print(f"Problematic data: {data}")


def select_choices_with_diversity(candidates, scores, target_count=4, ensure_max=True, ensure_min=False, min_score_gap=0.03):
    """
    Select diverse candidates based on scores.
    
    Algorithm:
    1. Ensure max (and optionally min) scores are included
    2. Greedily add candidates with sufficient score gap from already selected ones
    3. Fill remaining slots from remaining pool if needed
    
    Args:
        candidates: List of candidate items (e.g., coordinates)
        scores: List of scores corresponding to candidates
        target_count: Number of candidates to select
        ensure_max: Whether to ensure maximum score is included
        ensure_min: Whether to ensure minimum score is included
        min_score_gap: Minimum score difference between selected candidates
    
    Returns:
        Tuple of (selected_candidates, selected_scores)
    """
    if not candidates or not isinstance(candidates, list) or not isinstance(scores, list) or len(candidates) != len(scores):
        return [], []
        
    if len(candidates) == 0:
        return [], []

    if len(candidates) < target_count:
        return list(candidates), list(scores)

    float_scores = [float(s) for s in scores]
    combined = sorted(zip(candidates, float_scores), key=lambda x: x[1])
    
    selected_candidates = []
    selected_scores = []
    
    def is_selected(cand, sel_cands):
        """Check if candidate is already selected (handles numpy arrays)."""
        for sc in sel_cands:
            if isinstance(cand, np.ndarray) and isinstance(sc, np.ndarray):
                if np.array_equal(cand, sc):
                    return True
            elif cand == sc:
                return True
        return False

    # Step 1: Add best score if required
    if ensure_max and combined:
        max_score = combined[-1][1]
        max_candidates = [item for item in combined if abs(item[1] - max_score) < 1e-10]
        
        if len(max_candidates) > 1:
            import random
            best_candidate, best_score = random.choice(max_candidates)
        else:
            best_candidate, best_score = combined[-1]
            
        if not is_selected(best_candidate, selected_candidates):
            selected_candidates.append(best_candidate)
            selected_scores.append(best_score)

    # Step 2: Add worst score if required
    if ensure_min and combined:
        min_score = combined[0][1]
        min_candidates = [item for item in combined if abs(item[1] - min_score) < 1e-10]
        
        if len(min_candidates) > 1:
            import random
            worst_candidate, worst_score = random.choice(min_candidates)
        else:
            worst_candidate, worst_score = combined[0]
            
        if not is_selected(worst_candidate, selected_candidates):
            selected_candidates.insert(0, worst_candidate)
            selected_scores.insert(0, worst_score)
            
    remaining_combined = [cs for cs in combined if not is_selected(cs[0], selected_candidates)]

    # Step 3: Greedily add candidates with sufficient score gap
    for cand, score in reversed(remaining_combined):
        if len(selected_candidates) >= target_count:
            break
        if is_selected(cand, selected_candidates):
            continue

        is_far_enough = True
        if selected_scores:
            for sel_score in selected_scores:
                if abs(score - sel_score) < min_score_gap:
                    is_far_enough = False
                    break
        
        if is_far_enough:
            selected_candidates.append(cand)
            selected_scores.append(score)

    # Step 4: Fill remaining slots if not enough candidates selected
    if len(selected_candidates) < target_count:
        current_remaining_after_gap = [cs for cs in remaining_combined if not is_selected(cs[0], selected_candidates)]
        current_remaining_after_gap.sort(key=lambda x: x[1]) 
        
        needed = target_count - len(selected_candidates)
        to_add_from_remaining = []
        temp_remaining_pool = list(current_remaining_after_gap)

        # Alternate between highest and lowest to maintain diversity
        for _ in range(needed):
            if not temp_remaining_pool:
                break
            if len(selected_candidates) % 2 == 0 or len(temp_remaining_pool) == 1:
                to_add_from_remaining.append(temp_remaining_pool.pop(-1))
            else:
                to_add_from_remaining.append(temp_remaining_pool.pop(0))
        
        for cand, score in to_add_from_remaining:
             if len(selected_candidates) >= target_count:
                 break
             if not is_selected(cand, selected_candidates):
                selected_candidates.append(cand)
                selected_scores.append(score)

    # Step 5: Final fallback - take any remaining unique candidates
    if len(selected_candidates) < target_count:
        for cand, score in combined:
            if len(selected_candidates) >= target_count:
                break
            if not is_selected(cand, selected_candidates):
                selected_candidates.append(cand)
                selected_scores.append(score)

    final_candidates = selected_candidates[:target_count]
    final_scores = selected_scores[:target_count]
    
    return final_candidates, final_scores

def select_choices_prioritizing_correct_gap(candidates, scores, target_count=4, ensure_max=True, min_correct_gap=0.05):
    """
    Select choices prioritizing gap between correct answer and distractors.
    
    This function ensures sufficient gap between the max score (correct answer) and other choices,
    while being less strict about gaps between distractors themselves. This is useful when
    the primary concern is ensuring the correct answer is clearly distinguishable.
    
    Args:
        candidates: List of candidate items
        scores: List of scores corresponding to candidates
        target_count: Number of candidates to select
        ensure_max: Whether to ensure maximum score is included
        min_correct_gap: Minimum score difference between correct answer and distractors
    
    Returns:
        Tuple of (selected_candidates, selected_scores)
    """
    if not candidates or not isinstance(candidates, list) or not isinstance(scores, list) or len(candidates) != len(scores):
        return [], []
        
    if len(candidates) == 0:
        return [], []

    if len(candidates) < target_count:
        return list(candidates), list(scores)

    float_scores = [float(s) for s in scores]
    combined = sorted(zip(candidates, float_scores), key=lambda x: x[1])
    
    selected_candidates = []
    selected_scores = []
    
    def is_selected(cand, sel_cands):
        """Check if candidate is already selected."""
        for sc in sel_cands:
            if isinstance(cand, np.ndarray) and isinstance(sc, np.ndarray):
                if np.array_equal(cand, sc):
                    return True
            elif cand == sc:
                return True
        return False

    # Step 1: Add correct answer (max score)
    if ensure_max and combined:
        max_score = combined[-1][1]
        max_candidates = [item for item in combined if abs(item[1] - max_score) < 1e-10]
        
        if len(max_candidates) > 1:
            import random
            best_candidate, best_score = random.choice(max_candidates)
        else:
            best_candidate, best_score = combined[-1]
            
        selected_candidates.append(best_candidate)
        selected_scores.append(best_score)

    remaining_combined = [cs for cs in combined if not is_selected(cs[0], selected_candidates)]
    
    # Step 2: Separate candidates by gap from correct answer
    correct_score = selected_scores[0] if selected_scores else float('-inf')
    suitable_candidates = []
    fallback_candidates = []
    
    for cand, score in remaining_combined:
        gap_from_correct = abs(correct_score - score)
        if gap_from_correct >= min_correct_gap:
            suitable_candidates.append((cand, score))
        else:
            fallback_candidates.append((cand, score))
    
    # Step 3: Select from suitable candidates first
    suitable_candidates.sort(key=lambda x: x[1])
    
    for cand, score in suitable_candidates:
        if len(selected_candidates) >= target_count:
            break
        if not is_selected(cand, selected_candidates):
            selected_candidates.append(cand)
            selected_scores.append(score)
    
    # Step 4: Fill remaining slots from fallback candidates if needed
    if len(selected_candidates) < target_count:
        fallback_candidates.sort(key=lambda x: x[1])
        for cand, score in fallback_candidates:
            if len(selected_candidates) >= target_count:
                break
            if not is_selected(cand, selected_candidates):
                selected_candidates.append(cand)
                selected_scores.append(score)
    
    return selected_candidates[:target_count], selected_scores[:target_count]

def add_short_instruction(question_text):
    explanation = "\nPlease provide only short answer without explanation"
    if explanation not in question_text:
        return question_text + explanation
    return question_text 

_bias_free_random = None

def _init_bias_free_random():
    """Initialize bias-free random generator."""
    global _bias_free_random
    if _bias_free_random is None:
        import os
        try:
            choice_random_seed = int.from_bytes(os.urandom(4), 'big') % 2147483647
        except:
            choice_random_seed = (int(time.time() * 1000000) + os.getpid()) % 2147483647
            
        _bias_free_random = random.Random(choice_random_seed)
        print(f" Bias-free shuffle initialized: seed {choice_random_seed}")
    return _bias_free_random

def bias_free_shuffle(choices_list, use_fresh_seed=True):
    """
    Shuffle choices using independent random generator to prevent position bias.
    
    Uses OS-provided randomness (os.urandom) when available to ensure true randomness
    and prevent any systematic bias in choice ordering that could affect question difficulty.
    
    Args:
        choices_list: List of choices to shuffle
        use_fresh_seed: If True, generate new seed each time; if False, use shared generator
    
    Returns:
        Shuffled list of choices
    """
    if not choices_list:
        return choices_list
    
    if use_fresh_seed:
        import os
        try:
            fresh_seed = int.from_bytes(os.urandom(4), 'big') % 2147483647
        except:
            fresh_seed = (int(time.time() * 1000000) + os.getpid() + id(choices_list)) % 2147483647
        
        temp_rng = random.Random(fresh_seed)
        shuffled_choices = choices_list.copy()
        temp_rng.shuffle(shuffled_choices)
        return shuffled_choices
    else:
        rng = _init_bias_free_random()
        shuffled_choices = choices_list.copy()
        rng.shuffle(shuffled_choices)
        return shuffled_choices
