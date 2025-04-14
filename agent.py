import librosa
import numpy as np
import random
from transformers import pipeline
import sounddevice as sd
import soundfile as sf
from meta_data import MetaData

class MusicStoryAgent:
    def __init__(self):
        # TODO: Load the story generation model
        self.story_generator = None
        
        self.mood_map = MetaData.mood_map        
        self.tempo_map = MetaData.tempo_map
        self.dynamics_map = MetaData.dynamic_map
        self.instrument_map = MetaData.instrument_map
        self.story_templates = MetaData.story_templates
        self.character_types = MetaData.character_types
        self.settings = MetaData.settings

    def analyze_audio(self, audio_file):
        """Analyze the audio file to extract musical features"""
        # Load the audio file
        y, sr = librosa.load(audio_file)
        
        # Extract various features
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0].mean()
        
        # Chromagram for key detection
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        key_index = np.argmax(chroma_mean)
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key = keys[key_index]
        
        # Check if major or minor (simplistic approach)
        minor_indices = [(key_index + 3) % 12, (key_index + 7) % 12]
        major_indices = [(key_index + 4) % 12, (key_index + 7) % 12]
        
        minor_strength = sum(chroma_mean[i] for i in minor_indices)
        major_strength = sum(chroma_mean[i] for i in major_indices)
        
        is_major = major_strength > minor_strength
        tonality = "major" if is_major else "minor"
        
        # Dynamics (volume)
        rms = librosa.feature.rms(y=y)[0]
        dynamics = rms.mean()
        
        # Instrumentation detection (simplified)
        # Looking at frequency bands that might indicate different instrument groups
        spec = np.abs(librosa.stft(y))
        
        # Extract low, mid, and high frequencies
        low_freq = np.mean(spec[:int(spec.shape[0]*0.2), :])
        mid_freq = np.mean(spec[int(spec.shape[0]*0.2):int(spec.shape[0]*0.7), :])
        high_freq = np.mean(spec[int(spec.shape[0]*0.7):, :])
        
        # Simple instrument classification
        instruments = []
        if low_freq > 0.5:
            instruments.append("percussion")
        if mid_freq > 0.5:
            instruments.append("strings")
            instruments.append("woodwinds")
        if high_freq > 0.5:
            instruments.append("brass")
        if len(instruments) == 0:
            instruments.append(random.choice(list(self.instrument_map.keys())))
        
        # Map tempo to categories
        tempo_category = "moderate"
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
        
        # Map dynamics to categories
        dynamics_category = "moderate"
        if dynamics < 0.1:
            dynamics_category = "quiet"
        elif dynamics > 0.3:
            dynamics_category = "loud"
        
        # Determine mood
        mood_category = "major_happy"
        if is_major:
            if tempo_category in ["fast", "very_fast"]:
                mood_category = "major_happy"
            else:
                mood_category = "major_calm"
        else:
            if dynamics_category == "loud" or tempo_category in ["fast", "very_fast"]:
                mood_category = "minor_intense"
            else:
                mood_category = "minor_sad"
        
        # Special case for dissonance
        if spectral_bandwidth > 2000:
            mood_category = "dissonant"
        
        return {
            "tempo": tempo_category,
            "key": key,
            "tonality": tonality,
            "dynamics": dynamics_category,
            "instruments": instruments,
            "mood": mood_category
        }
    
    def generate_story_seed(self, music_features):
        """Generate a story seed based on the musical features"""
        # Select random descriptors from each category
        mood = random.choice(self.mood_map[music_features["mood"]])
        tempo = random.choice(self.tempo_map[music_features["tempo"]])
        dynamics = random.choice(self.dynamics_map[music_features["dynamics"]])
        
        # Choose random characters and settings
        character = random.choice(self.character_types)
        setting = random.choice(self.settings)
        
        # Add instrument influence
        instrument_mood = []
        for instrument in music_features["instruments"]:
            if instrument in self.instrument_map:
                instrument_mood.append(random.choice(self.instrument_map[instrument]))
        
        # Template for story seed
        template = random.choice(self.story_templates)
        story_seed = template.format(
            mood=mood,
            tempo=tempo,
            dynamics=dynamics,
            character=character,
            Character=character.capitalize(),
            setting=setting
        )
        
        # Add instrument influence
        if instrument_mood:
            story_seed += f" There was a {random.choice(instrument_mood)} quality to the atmosphere."
        
        return story_seed
    
    def generate_full_story(self, story_seed, length=500):
        """Generate a full story based on the seed"""
        # Generate the story using the pre-trained model
        story = self.story_generator(
            story_seed, 
            max_length=length, 
            num_return_sequences=1,
            temperature=0.8
        )[0]['generated_text']
        
        return story
    
    def create_story_from_music(self, audio_file, story_length=500):
        """Main function to create a story from a music file"""
        print(f"Analyzing music: {audio_file}")
        
        # Analyze the audio
        music_features = self.analyze_audio(audio_file)
        
        print("Music features extracted:")
        print(f"- Tempo: {music_features['tempo']}")
        print(f"- Key: {music_features['key']} {music_features['tonality']}")
        print(f"- Dynamics: {music_features['dynamics']}")
        print(f"- Detected instruments: {', '.join(music_features['instruments'])}")
        print(f"- Overall mood: {music_features['mood']}")
        
        # Generate story seed
        story_seed = self.generate_story_seed(music_features)
        print("\nStory seed generated:")
        print(story_seed)
        
        # Generate full story
        full_story = self.generate_full_story(story_seed, length=story_length)
        print("\nFull story generated. Enjoy!")
        
        return {
            "music_features": music_features,
            "story_seed": story_seed,
            "full_story": full_story
        }

    def play_music_while_reading(self, audio_file, story):
        """Optional feature to play the music while displaying the story"""
        data, fs = sf.read(audio_file)        
        sd.play(data, fs)
        print(story)
        sd.wait()

if __name__ == "__main__":
    agent = MusicStoryAgent()
    
    audio_file = "path_to_your_music_file.mp3"
    
    '''
    # Mocked music features for testing
    music_features = {
        "tempo": "moderate",
        "key": "G",
        "tonality": "major",
        "dynamics": "moderate",
        "instruments": ["strings", "piano"],
        "mood": "major_calm"
    }
    story_seed = agent.generate_story_seed(music_features)
    full_story = agent.generate_full_story(story_seed, length=500)
    print("\nGenerated Story:")
    print(full_story)
    '''    
    # result = agent.create_story_from_music(audio_file)
    # print("\nGenerated Story:")
    # print(result["full_story"])