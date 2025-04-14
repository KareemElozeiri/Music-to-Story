import os
import torch
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, get_linear_schedule_with_warmup

from data.dataset import MusicStoryDataset
from utils.visualization import plot_training_loss
from config import (
    MODEL_NAME, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH, BATCH_SIZE, 
    LEARNING_RATE, NUM_EPOCHS, WARMUP_RATIO, GEN_MAX_LENGTH, 
    NUM_BEAMS, NO_REPEAT_NGRAM_SIZE, DEFAULT_MODEL_SAVE_PATH
)

logger = logging.getLogger(__name__)


class T5MusicStoryGenerator:
    """Class for fine-tuning T5 model on music-to-story generation."""
    
    def __init__(self, model_name=MODEL_NAME, device=None):
        """
        Initialize the model.
        
        Args:
            model_name (str): Name of the pretrained T5 model
            device (str, optional): Device to use (cuda or cpu)
        """
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
    
    def fine_tune(self, train_data, val_data=None, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, 
                  learning_rate=LEARNING_RATE, save_path=DEFAULT_MODEL_SAVE_PATH):
        """
        Fine-tune the T5 model on the music-story dataset.
        
        Args:
            train_data (list): Training data
            val_data (list, optional): Validation data
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            learning_rate (float): Learning rate
            save_path (str): Path to save the model
            
        Returns:
            tuple: (train_losses, val_losses)
        """
        # Create datasets
        train_dataset = MusicStoryDataset(
            train_data, 
            self.tokenizer, 
            max_input_length=MAX_INPUT_LENGTH, 
            max_target_length=MAX_TARGET_LENGTH
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_data:
            val_dataset = MusicStoryDataset(
                val_data, 
                self.tokenizer,
                max_input_length=MAX_INPUT_LENGTH, 
                max_target_length=MAX_TARGET_LENGTH
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        else:
            val_loader = None
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(WARMUP_RATIO * total_steps),
            num_training_steps=total_steps
        )
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            epoch_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                epoch_loss += loss.item()
                
                # Backward pass and optimization
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            logger.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")
            
            # Validation
            if val_loader:
                self.model.eval()
                val_loss = 0
                
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc="Validation"):
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        
                        val_loss += outputs.loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                logger.info(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}")
        
        # Save the model
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Plot training/validation loss
        if val_losses:
            plot_training_loss(train_losses, val_losses, save_path)
            
        return train_losses, val_losses
    
    def generate_story(self, music_description, max_length=GEN_MAX_LENGTH, num_beams=NUM_BEAMS):
        """
        Generate a story based on a music description.
        
        Args:
            music_description (str): Textual description of music
            max_length (int): Maximum length of generated story
            num_beams (int): Number of beams for beam search
            
        Returns:
            str: Generated story
        """
        # Prepare input for model
        input_text = f"Generate a story for this music: {music_description}"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        
        # Generate story
        self.model.eval()
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                num_beams=num_beams,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                early_stopping=True
            )
        
        # Decode the generated story
        story = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return story
    
    @classmethod
    def from_pretrained(cls, model_path, device=None):
        """
        Load a fine-tuned model from disk.
        
        Args:
            model_path (str): Path to the saved model
            device (str, optional): Device to use
            
        Returns:
            T5MusicStoryGenerator: Loaded model instance
        """
        instance = cls(model_name=None, device=device)
        instance.tokenizer = T5Tokenizer.from_pretrained(model_path)
        instance.model = T5ForConditionalGeneration.from_pretrained(model_path).to(instance.device)
        return instance