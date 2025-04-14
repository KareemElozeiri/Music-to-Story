"""
Dataset classes for handling music-story data pairs.
"""

import torch
from torch.utils.data import Dataset


class MusicStoryDataset(Dataset):
    """Dataset for T5 fine-tuning with music features as input and stories as output."""
    
    def __init__(self, data, tokenizer, max_input_length=512, max_target_length=1024):
        """
        Initialize the dataset.
        
        Args:
            data (list): List of dictionaries containing music descriptions and stories
            tokenizer: T5 tokenizer
            max_input_length (int): Maximum input sequence length
            max_target_length (int): Maximum target sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Create the input prompt
        input_text = f"Generate a story for this music: {item['music_description']}"
        
        # Tokenize inputs and targets
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        target_encoding = self.tokenizer(
            item['story'],
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Important: Replace padding token id with -100 for loss calculation
        target_ids = target_encoding.input_ids
        target_ids[target_ids == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_encoding.input_ids.squeeze(),
            "attention_mask": input_encoding.attention_mask.squeeze(),
            "labels": target_ids.squeeze()
        }