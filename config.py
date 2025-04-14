"""
Configuration settings for the Music-to-Story Agent.
"""

# Model configuration
MODEL_NAME = "t5-base"  # Base model to use
MAX_INPUT_LENGTH = 512  # Maximum input sequence length
MAX_TARGET_LENGTH = 1024  # Maximum output sequence length

# Training configuration
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1

# Generation configuration
GEN_MAX_LENGTH = 512
NUM_BEAMS = 4
NO_REPEAT_NGRAM_SIZE = 3

# Data configuration
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# File paths
DEFAULT_MODEL_SAVE_PATH = "saved_models/t5_music_story_model"
DEFAULT_DATASET_PATH = "data/music_story_dataset.json"
SYNTHETIC_DATASET_PATH = "data/synthetic_music_story_dataset.json"