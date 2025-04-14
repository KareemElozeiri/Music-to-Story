"""
Functions for preparing and processing music-story datasets.
"""

import os
import json
import logging
import numpy as np
from tqdm import tqdm
from faker import Faker
from sklearn.model_selection import train_test_split

from ..feature_extraction.music_features import MusicFeatureExtractor
from ..config import RANDOM_SEED, VALIDATION_SPLIT, SYNTHETIC_DATASET_PATH

logger = logging.getLogger(__name__)


def prepare_dataset(dataset_path, music_dir=None, feature_extractor=None):
    """
    Prepare dataset for fine-tuning.
    
    Args:
        dataset_path (str): Path to JSON file or directory containing music files
        music_dir (str, optional): Base directory for music files
        feature_extractor (MusicFeatureExtractor, optional): Feature extractor instance
        
    Returns:
        list: Processed data items
    """
    if feature_extractor is None:
        feature_extractor = MusicFeatureExtractor()
    
    if dataset_path.endswith('.json'):
        # Load existing dataset
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
            
        processed_data = []
        
        for item in tqdm(dataset, desc="Processing dataset"):
            music_path = os.path.join(music_dir, item['music_path']) if music_dir else item['music_path']
            
            # Extract features if the file exists
            if os.path.exists(music_path):
                features = feature_extractor.extract_features(music_path)
                if features:
                    music_description = feature_extractor.create_text_description(features)
                    processed_data.append({
                        'music_path': item['music_path'],
                        'music_description': music_description,
                        'story': item['story'],
                        'features': features
                    })
            else:
                # If we can't access the file but have a description, use that
                if 'music_description' in item:
                    processed_data.append({
                        'music_path': item['music_path'],
                        'music_description': item['music_description'],
                        'story': item['story']
                    })
                    
        return processed_data
    
    elif os.path.isdir(dataset_path):
        processed_data = []
        
        for root, _, files in os.walk(dataset_path):
            for file in tqdm(files, desc="Extracting features"):
                if file.endswith(('.mp3', '.wav', '.ogg', '.flac')):
                    music_path = os.path.join(root, file)
                    features = feature_extractor.extract_features(music_path)
                    
                    if features:
                        music_description = feature_extractor.create_text_description(features)
                        processed_data.append({
                            'music_path': os.path.relpath(music_path, start=music_dir) if music_dir else music_path,
                            'music_description': music_description,
                            'features': features
                        })
        
        return processed_data
    
    else:
        raise ValueError("dataset_path must be a JSON file or directory")


def prepare_real_dataset(music_dir, story_file, output_file, split=VALIDATION_SPLIT):
    """
    Prepare a real dataset by extracting features from music files and pairing with stories.
    
    Args:
        music_dir (str): Directory containing music files
        story_file (str): JSON file containing stories with music file references
        output_file (str): Path to save the processed dataset
        split (float): Train/validation split ratio
        
    Returns:
        tuple: (train_data, val_data)
    """
    # Initialize feature extractor
    feature_extractor = MusicFeatureExtractor()
    
    # Load stories
    with open(story_file, 'r') as f:
        stories = json.load(f)
    
    # Process music files and match with stories
    dataset = []
    
    for item in tqdm(stories, desc="Processing music files"):
        music_path = os.path.join(music_dir, item['music_path'])
        
        if os.path.exists(music_path):
            features = feature_extractor.extract_features(music_path)
            
            if features:
                music_description = feature_extractor.create_text_description(features)
                
                dataset.append({
                    'music_path': item['music_path'],
                    'music_description': music_description,
                    'story': item['story'],
                    'features': features
                })
    
    # Split into train and validation sets
    train_data, val_data = train_test_split(dataset, test_size=split, random_state=RANDOM_SEED)
    
    # Save processed dataset
    with open(output_file, 'w') as f:
        json.dump({
            'train': train_data,
            'validation': val_data
        }, f, indent=2)
    
    logger.info(f"Processed {len(dataset)} music-story pairs")
    logger.info(f"Train set: {len(train_data)} examples")
    logger.info(f"Validation set: {len(val_data)} examples")
    
    return train_data, val_data


def create_synthetic_dataset(size=100, save_path=SYNTHETIC_DATASET_PATH):
    """
    Create an artificial dataset for initial training.
    
    Args:
        size (int): Number of examples to generate
        save_path (str): Path to save the dataset
        
    Returns:
        list: Generated dataset
    """
    feature_extractor = MusicFeatureExtractor()
    fake = Faker()
    
    # Possible values for synthetic data
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    tonalities = ['major', 'minor']
    tempo_categories = ['very_slow', 'slow', 'moderate', 'fast', 'very_fast']
    dynamics_categories = ['quiet', 'moderate', 'loud']
    mood_categories = ['major_happy', 'major_calm', 'minor_sad', 'minor_intense', 'dissonant']
    
    # Generate synthetic dataset
    dataset = []
    
    for i in range(size):
        # Generate random musical features
        key = np.random.choice(keys)
        tonality = np.random.choice(tonalities)
        tempo = np.random.randint(40, 200)
        
        if tempo < 60:
            tempo_category = "very_slow"
        elif tempo < 80:
            tempo_category = "slow"
        elif tempo < 120:
            tempo_category = "moderate"
        elif tempo < 160:
            tempo_category = "fast"
        else:
            tempo_category = "very_fast"
            
        dynamics = np.random.uniform(0.05, 0.5)
        if dynamics < 0.1:
            dynamics_category = "quiet"
        elif dynamics > 0.3:
            dynamics_category = "loud"
        else:
            dynamics_category = "moderate"
            
        # Determine mood based on tonality and tempo
        if tonality == "major":
            if tempo_category in ["fast", "very_fast"]:
                mood_category = "major_happy"
            else:
                mood_category = "major_calm"
        else:
            if dynamics_category == "loud" or tempo_category in ["fast", "very_fast"]:
                mood_category = "minor_intense"
            else:
                mood_category = "minor_sad"
        
        # Randomly set some to dissonant
        if np.random.random() < 0.1:
            mood_category = "dissonant"
            
        features = {
            "tempo": tempo,
            "tempo_category": tempo_category,
            "key": key,
            "tonality": tonality,
            "dynamics": dynamics,
            "dynamics_category": dynamics_category,
            "mood_category": mood_category
        }
        
        # Generate text description
        music_description = feature_extractor.create_text_description(features)
        
        # Generate a synthetic story based on the mood
        story = _generate_synthetic_story(mood_category, fake)
        
        # Add to dataset
        dataset.append({
            "music_path": f"synthetic_{i}.mp3",  # Placeholder path
            "music_description": music_description,
            "story": story,
            "features": features
        })
    
    # Save dataset
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(dataset, f, indent=2)
        
    logger.info(f"Created synthetic dataset with {size} examples and saved to {save_path}")
    return dataset


def _generate_synthetic_story(mood_category, fake):
    """Generate a synthetic story based on mood."""
    # Set the theme based on mood
    if mood_category == "major_happy":
        themes = ["celebration", "achievement", "reunion", "adventure", "discovery"]
    elif mood_category == "major_calm":
        themes = ["reflection", "nostalgia", "serenity", "healing", "comfort"]
    elif mood_category == "minor_sad":
        themes = ["loss", "longing", "regret", "goodbye", "melancholy"]
    elif mood_category == "minor_intense":
        themes = ["challenge", "conflict", "suspense", "revelation", "transformation"] 
    else:  # dissonant
        themes = ["confusion", "disorientation", "surrealism", "chaos", "uncertainty"]
        
    theme = np.random.choice(themes)
    
    # Generate a story with paragraphs
    paragraphs = np.random.randint(3, 7)
    story = ""
    
    for p in range(paragraphs):
        sentences = np.random.randint(3, 8)
        paragraph = ""
        
        for s in range(sentences):
            paragraph += fake.sentence() + " "
            
        story += paragraph.strip() + "\n\n"
    
    return story.strip()