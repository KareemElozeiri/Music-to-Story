import os
import sys
from sklearn.model_selection import train_test_split
from model.model import T5MusicStoryGenerator
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))





if __name__ == "__main__":
    # Initialize the model
    t5_generator = T5MusicStoryGenerator(model_name="t5-base")
    
    # Create synthetic dataset for initial training
    print("Creating synthetic dataset...")
    synthetic_dataset = t5_generator.create_artificial_dataset(size=200)
    
    # Split into train and validation sets
    train_data, val_data = train_test_split(synthetic_dataset, test_size=0.2, random_state=42)
    print(f"Train set: {len(train_data)} examples")
    print(f"Validation set: {len(val_data)} examples")
    
    # Fine-tune the model
    print("\nStarting fine-tuning...")
    train_losses, val_losses = t5_generator.fine_tune(
        train_data, 
        val_data, 
        epochs=3, 
        batch_size=4,
        learning_rate=5e-5,
        save_path="t5_music_story_model"
    )
    
    # Generate a story from an example music file
    print("\nGenerating a sample story...")
    # sample_music = "path/to/sample_music.mp3"
    # result = t5_generator.generate_story(music_file=sample_music)
    
    # For demonstration, use synthetic features
    sample_features = {
        "tempo": 120,
        "tempo_category": "moderate",
        "key": "G",
        "tonality": "major",
        "dynamics": 0.2,
        "dynamics_category": "moderate",
        "mood_category": "major_calm"
    }
    result = t5_generator.generate_story(music_features=sample_features)
    
    print("\nMusic Description:")
    print(result["music_description"])
    print("\nGenerated Story:")
    print(result["story"])