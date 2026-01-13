# Geo3DVQA Function Documentation (English)

## Overview

This document provides detailed descriptions of each function executed from `run.py`, including their purposes, implementations, and processing details.

---

## 1. run.py

### 1.1 `main()`
**Purpose**: Main execution function that parses command-line arguments and executes the dataset generation pipeline.

**Processing**:
- Parses command-line arguments (`--svf_path`, `--geonrw_path`, `--mode`, `--qa_multiplier`, etc.)
- Sets global seed for reproducibility
- Generates datasets for different versions (standard, medium, large, xl, small)
- Executes parallel processing using multiprocessing
- Saves question-answer data in JSONL format

**Notes**:
- Multiprocessing may increase memory usage
- Memory management is important when processing large numbers of files

### 1.2 `process_single_file(process_args, svf_plots, qa_multiplier=1)`
**Purpose**: Processes a single SVF file and generates questions and answers.

**Processing**:
- Loads SVF file, height map, segmentation map, and RGB image
- Checks and resizes images for consistency
- Validates valid pixels (skips if less than 100 pixels)
- Checks SVF value variation (skips if range is less than 0.001)
- Instantiates `ConstructSVFQuestionRGB` or `ConstructSVFQuestionRegionBased`
- Generates questions (calls `chooseQuestionsToAsk` method)
- Generates plot images (optional)

**Notes**:
- Memory is explicitly freed after processing for efficiency
- Comprehensive error handling skips invalid files

### 1.3 `smart_area_sampling(all_files, sampling_ratio=0.3, min_files_per_area=2)`
**Purpose**: Performs area-based smart sampling to reduce the number of files to process.

**Processing**:
- Groups files by area
- Samples files from each area based on `sampling_ratio`
- Ensures at least `min_files_per_area` files from each area

**Notes**:
- Sampling maintains dataset diversity
- Balance between areas is maintained

### 1.4 `flush_batch_buffer(answer_file, question_file, detailed_file, conversation_file=None)`
**Purpose**: Asynchronously writes batch buffer contents to files.

**Processing**:
- Uses `ThreadPoolExecutor` for asynchronous writing
- Executes when buffer size reaches `max_buffer_size` (default 500)
- Batch processing for memory efficiency

**Notes**:
- Asynchronous processing reduces I/O wait time
- Error handling is implemented

---

## 2. svf_questions_rgb_estimated.py

### 2.1 `class ConstructSVFQuestionRGB`
**Purpose**: Class that generates questions based on SVF maps estimated from RGB images.

**Key Methods**:

#### 2.1.1 `__init__(estimated_svf_map, ...)`
**Purpose**: Initializes the class, receiving SVF map, height map, segmentation map, and RGB image, and prepares for question generation.

**Processing**:
- Stores various maps and images
- Initializes question templates (13 categories, multiple templates per category)
- Sets category weights (weighting based on model performance)
- Initializes cache system
- Initializes error handler

#### 2.1.2 `chooseQuestionsToAsk(number_question=50)`
**Purpose**: Generates a specified number of questions.

**Processing**:
- Selects question types based on category weights
- Calls methods corresponding to each question type
- Generates questions, answers, and canonical questions
- Balances categories (when `balanced_categories` is enabled)


#### 2.1.3 `sunExposure()`
**Purpose**: Generates questions about solar exposure.

**Processing**:
- Extracts candidate points from SVF map
- Calculates SVF values for each candidate point
- Selects point with highest value as correct answer
- Generates choices (4-choice including highest value)
- Selects and applies question template

#### 2.1.4 `skyVisibility()`
**Purpose**: Generates questions about sky visibility.

**Processing**:
- Uses SVF map and segmentation map
- Combines building density and SVF values for scoring
- Selects location with highest score as correct answer

#### 2.1.5 `urbanDensity()`
**Purpose**: Generates questions about urban density.

**Processing**:
- Extracts building areas from segmentation map
- Performs grid-based or region-based analysis
- Selects region with highest building density as correct answer

#### 2.1.6 `visibilityRange()`
**Purpose**: Generates questions about visibility range.

**Processing**:
- Combines SVF map and height map
- Calculates visibility spread
- Selects location with highest value as correct answer

#### 2.1.7 `opennessAssessment()`
**Purpose**: Generates questions about spatial openness.

**Processing**:
- Combines SVF values and building density for scoring
- Selects location with highest openness as correct answer

---

## 3. svf_questions_region_based.py

### 3.1 `class ConstructSVFQuestionRegionBased`
**Purpose**: Class that generates region-based questions. Inherits from `ConstructSVFQuestionRGB`.

**Key Methods**:

#### 3.1.1 `svfRegionAnalysis()`
**Purpose**: Generates questions about SVF variability analysis.

**Processing**:
- Generates 3 regions
- Calculates SVF mean and standard deviation for each region
- Selects region with highest/lowest standard deviation as correct answer
- Uses relative coordinates (%) for region specification

#### 3.1.2 `assign_balanced_labels(regions, target_region)`
**Purpose**: Label distribution function to statistically evenly distribute correct answer positions.

**Processing**:
- Randomization through multiple rounds of shuffling
- Label distribution through weighted selection
- Prevents correct answers from being biased toward specific positions (A, B, C, D)

---

## 4. utils_sampling.py

### 4.1 `filter_by_distance(points, min_dist=30)`
**Purpose**: Filter function that excludes choices that are too close to each other.

**Processing**:
- Calculates Euclidean distance for each candidate point
- Excludes points with distance less than `min_dist`

**Notes**:
- Default `min_dist=30` pixels is an appropriate value
- Ensures spatial diversity of choices

### 4.2 `filter_by_metric_gap(points, metrics, min_gap=0.08, target_count=4)`
**Purpose**: Filter function that selects choices with sufficient differences in metric values.

**Processing**:
- Sorts metric values in descending order
- Selects points with difference of at least `min_gap` from existing choices
- Generates `target_count` choices

**Notes**:
- `min_gap=0.08` is an appropriate value considering SVF value range (0-1)

### 4.3 `select_by_quartiles(points, metrics, target_count=4, min_points_per_quartile=1)`
**Purpose**: Function that selects choices based on quartiles.

**Processing**:
- Calculates quartiles of metric values
- Selects at least `min_points_per_quartile` points from each quartile
- Applies distance filter

**Notes**:
- Quartile-based selection ensures even distribution of values

### 4.4 `select_by_quartiles_with_top(points, metrics, target_count=4, min_points_per_quartile=1)`
**Purpose**: Function that selects choices based on quartiles (always includes highest value).

**Processing**:
- Ensures highest value first
- Selects remaining points by quartiles
- Applies distance filter (prioritizes keeping highest value)

---

## 5. utils_seed.py

### 5.1 `class SeedManager`
**Purpose**: Seed management class for reproducibility.

**Key Methods**:

#### 5.1.1 `__init__(seed: int = None)`
**Purpose**: Initializes the seed manager.

**Processing**:
- Sets seed value (from environment variable `RANDOM_SEED` or default value 42)
- Sets seeds for Python's `random`, NumPy's `random`, and PyTorch's `random`

#### 5.1.2 `create_file_seed(file_path: str) -> int`
**Purpose**: Generates a deterministic seed based on file path.

**Processing**:
- Calculates hash value of filename
- Combines with global seed to generate seed

**Notes**:
- Same file always generates same seed, ensuring reproducibility
- Hash collisions are theoretically possible but unlikely

---

## 6. visualize_qa_results.py

### 6.1 `class QAVisualizer`
**Purpose**: Class that visualizes QA results.

**Key Methods**:

#### 6.1.1 `visualize_qa_result_multimodal(qa_result, save_path=None)`
**Purpose**: Visualizes QA results in multimodal format (RGB, SVF, segmentation).

**Processing**:
- Loads RGB image, SVF image, and segmentation image
- Adds overlays according to question category (points, regions, grids, etc.)
- Visually distinguishes correct and incorrect answers
- Saves images

#### 6.1.2 `extract_region_from_text(text, img_shape)`
**Purpose**: Extracts region coordinates from question text.

**Processing**:
- Parses region coordinates using regular expressions
- Converts percentage coordinates to pixel coordinates

**Notes**:
- Parsing depends on question text format; may not work correctly if format changes

---

## Summary

This codebase is a comprehensive system for generating question-answer datasets from 3D geospatial data. It includes appropriate error handling, reproducibility assurance, and efficient processing. The system supports multiple question categories, various sampling strategies, and multimodal visualization capabilities.

