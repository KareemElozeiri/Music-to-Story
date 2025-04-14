"""
Tools for extracting and representing musical features from audio files.
"""

import numpy as np
import librosa
import random
import logging

logger = logging.getLogger(__name__)


class MusicFeatureExtractor:
    """Extract relevant features from music files and convert them to text descriptions."""
    
    def __init__(self):
        """Initialize the feature extractor with descriptive mappings."""
        self.mood_map = {
            "major_happy": ["joyful", "triumphant", "celebratory", "optimistic"],
            "major_calm": ["peaceful", "serene", "gentle", "nostalgic"],
            "minor_sad": ["melancholic", "sorrowful", "longing", "regretful"],
            "minor_intense": ["mysterious", "suspenseful", "dramatic", "intense"],
            "dissonant": ["chaotic", "unsettling", "confusing", "surreal"]
        }
        
        self.tempo_descriptors = {
            "very_slow": ["slow-paced", "methodical", "reflective", "deliberate"],
            "slow": ["relaxed", "contemplative", "steady", "unhurried"],
            "moderate": ["balanced", "flowing", "steady", "regular"],
            "fast": ["energetic", "lively", "dynamic", "active"],
            "very_fast": ["frantic", "urgent", "racing", "exhilarating"]
        }
        
        self.dynamics_descriptors = {
            "quiet": ["subtle", "intimate", "delicate", "whispered"],
            "moderate": ["balanced", "controlled", "steady", "moderate"],
            "loud": ["powerful", "bold", "commanding", "intense"]
        }

    def extract_features(self, audio_file):
        """
        Extract musical features from an audio file.
        
        Args:
            audio_file (str): Path to the audio file
            
        Returns:
            dict: Dictionary of extracted features or None if extraction failed
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_file)
            
            # Extract tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # Extract key/tonality
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            key_idx = np.argmax(chroma_mean)
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key = keys[key_idx]
            
            # Check if major or minor
            minor_indices = [(key_idx + 3) % 12, (key_idx + 7) % 12]
            major_indices = [(key_idx + 4) % 12, (key_idx + 7) % 12]
            
            minor_strength = sum(chroma_mean[i] for i in minor_indices)
            major_strength = sum(chroma_mean[i] for i in major_indices)
            
            tonality = "major" if major_strength > minor_strength else "minor"
            
            # Extract dynamics
            rms = librosa.feature.rms(y=y)[0]
            dynamics = np.mean(rms)
            
            # Extract spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0])
            spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
            
            # Extract MFCC features
            mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
            
            # Categorize features
            tempo_category = self._categorize_tempo(tempo)
            dynamics_category = self._categorize_dynamics(dynamics)
            mood_category = self._determine_mood(tonality, tempo_category, dynamics_category, spectral_bandwidth)
                
            # Create feature dictionary
            features = {
                "tempo": tempo,
                "tempo_category": tempo_category,
                "key": key,
                "tonality": tonality,
                "dynamics": dynamics,
                "dynamics_category": dynamics_category,
                "spectral_centroid": spectral_centroid,
                "spectral_bandwidth": spectral_bandwidth,
                "mfccs": mfccs.tolist(),
                "mood_category": mood_category
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features from {audio_file}: {e}")
            return None
    
    def _categorize_tempo(self, tempo):
        """Categorize tempo value into descriptive category."""
        if tempo < 60:
            return "very_slow"
        elif tempo < 80:
            return "slow"
        elif tempo < 120:
            return "moderate"
        elif tempo < 160:
            return "fast"
        else:
            return "very_fast"
    
    def _categorize_dynamics(self, dynamics):
        """Categorize dynamics (volume) value into descriptive category."""
        if dynamics < 0.1:
            return "quiet"
        elif dynamics > 0.3:
            return "loud"
        else:
            return "moderate"
    
    def _determine_mood(self, tonality, tempo_category, dynamics_category, spectral_bandwidth):
        """Determine overall mood category based on musical features."""
        # Check for dissonance first
        if spectral_bandwidth > 2000:
            return "dissonant"
            
        if tonality == "major":
            if tempo_category in ["fast", "very_fast"]:
                return "major_happy"
            else:
                return "major_calm"
        else:
            if dynamics_category == "loud" or tempo_category in ["fast", "very_fast"]:
                return "minor_intense"
            else:
                return "minor_sad"
    
    def create_text_description(self, features):
        """
        Create a textual description of the musical features.
        
        Args:
            features (dict): Dictionary of musical features
            
        Returns:
            str: Natural language description of the music
        """
        tempo_desc = random.choice(self.tempo_descriptors[features["tempo_category"]])
        dynamics_desc = random.choice(self.dynamics_descriptors[features["dynamics_category"]])
        mood_desc = random.choice(self.mood_map[features["mood_category"]])
        
        description = f"Music in {features['key']} {features['tonality']} with a {tempo_desc} tempo around {int(features['tempo'])} BPM. "
        description += f"The piece has a {dynamics_desc} dynamic quality and creates a {mood_desc} atmosphere. "
        
        return description