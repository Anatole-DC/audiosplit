#!/usr/bin/env python3
"""
MIDI Instrument Separation Model

This module provides tools for training a deep learning model that can separate
MIDI files into individual instrument tracks.

The approach:
1. Convert MIDI files to piano roll representations
2. Train a model to predict individual instrument tracks from mixed input
3. Convert model outputs back to separate MIDI files
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pretty_midi
from pathlib import Path
import pickle
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class MIDIDataProcessor:
    """Processes MIDI files for instrument separation training"""
    
    def __init__(self, fs=4, n_semitones=128):
        """
        Initialize the MIDI processor
        
        Args:
            fs: Sampling frequency (frames per second)
            n_semitones: Number of MIDI note values (0-127)
        """
        self.fs = fs
        self.n_semitones = n_semitones
        self.instrument_classes = self._get_instrument_classes()
    
    def _get_instrument_classes(self):
        """Get common instrument classes for separation"""
        return {
            'piano': [0, 1, 2, 3, 4, 5, 6, 7],  # Piano family
            'guitar': [24, 25, 26, 27, 28, 29, 30, 31],  # Guitar family
            'bass': [32, 33, 34, 35, 36, 37, 38, 39],  # Bass family
            'strings': [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51],  # Strings
            'brass': [56, 57, 58, 59, 60, 61, 62, 63],  # Brass family
            'woodwind': [64, 65, 66, 67, 68, 69, 70, 71],  # Woodwind family
            'synth': [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],  # Synth
            'drums': [128]  # Drums (special case)
        }
    
    def midi_to_piano_roll(self, midi_file_path, max_length=None):
        """
        Convert MIDI file to piano roll representation
        
        Args:
            midi_file_path: Path to MIDI file
            max_length: Maximum length in seconds (None for full length)
            
        Returns:
            Dictionary with piano rolls for each instrument class
        """
        try:
            midi = pretty_midi.PrettyMIDI(str(midi_file_path))
            
            if max_length is None:
                max_length = midi.get_end_time()
            
            # Create time grid
            times = np.arange(0, max_length, 1.0/self.fs)
            
            # Initialize piano rolls for each instrument class
            piano_rolls = {}
            for class_name in self.instrument_classes.keys():
                piano_rolls[class_name] = np.zeros((len(times), self.n_semitones))
            
            # Process each instrument
            for instrument in midi.instruments:
                if instrument.is_drum:
                    class_name = 'drums'
                else:
                    # Find instrument class
                    class_name = self._classify_instrument(instrument.program)
                
                if class_name in piano_rolls:
                    # Get piano roll for this instrument
                    instrument_roll = instrument.get_piano_roll(fs=self.fs, times=times)
                    
                    # Add to the appropriate class (transpose to match our format)
                    if instrument_roll.shape[1] == len(times):
                        piano_rolls[class_name] += instrument_roll.T
            
            # Normalize
            for class_name in piano_rolls:
                piano_rolls[class_name] = np.clip(piano_rolls[class_name], 0, 1)
            
            return piano_rolls
            
        except Exception as e:
            print(f"Error processing {midi_file_path}: {e}")
            return None
    
    def _classify_instrument(self, program):
        """Classify MIDI program number into instrument class"""
        for class_name, programs in self.instrument_classes.items():
            if program in programs:
                return class_name
        return 'synth'  # Default class
    
    def piano_roll_to_midi(self, piano_rolls, output_dir, filename_prefix="separated"):
        """
        Convert piano roll back to MIDI files
        
        Args:
            piano_rolls: Dictionary of piano rolls by instrument class
            output_dir: Directory to save MIDI files
            filename_prefix: Prefix for output filenames
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        created_files = []
        
        for class_name, piano_roll in piano_rolls.items():
            if np.sum(piano_roll) > 0:  # Only create file if there are notes
                midi = pretty_midi.PrettyMIDI()
                
                # Create instrument
                if class_name == 'drums':
                    instrument = pretty_midi.Instrument(program=0, is_drum=True)
                else:
                    # Use representative program for each class
                    program = self.instrument_classes[class_name][0]
                    instrument = pretty_midi.Instrument(program=program)
                
                # Convert piano roll to notes
                notes = self._piano_roll_to_notes(piano_roll)
                instrument.notes.extend(notes)
                
                midi.instruments.append(instrument)
                
                # Save MIDI file
                output_file = output_dir / f"{filename_prefix}_{class_name}.mid"
                midi.write(str(output_file))
                created_files.append(output_file)
        
        return created_files
    
    def _piano_roll_to_notes(self, piano_roll, threshold=0.5):
        """Convert piano roll matrix to pretty_midi Note objects"""
        notes = []
        
        # Find note onsets and offsets
        for pitch in range(piano_roll.shape[1]):
            # Get the piano roll for this pitch
            pitch_roll = piano_roll[:, pitch] > threshold
            
            # Find note boundaries
            note_changes = np.diff(np.concatenate(([False], pitch_roll, [False])).astype(int))
            note_onsets = np.where(note_changes == 1)[0]
            note_offsets = np.where(note_changes == -1)[0]
            
            # Create notes
            for onset, offset in zip(note_onsets, note_offsets):
                start_time = onset / self.fs
                end_time = offset / self.fs
                
                if end_time > start_time:  # Valid note
                    note = pretty_midi.Note(
                        velocity=80,
                        pitch=pitch,
                        start=start_time,
                        end=end_time
                    )
                    notes.append(note)
        
        return notes
    
    def create_training_data(self, midi_files, max_length=16.0, augment=True):
        """
        Create training data from MIDI files
        
        Args:
            midi_files: List of MIDI file paths
            max_length: Maximum sequence length in seconds
            augment: Whether to apply data augmentation
            
        Returns:
            Tuple of (inputs, targets) for training
        """
        inputs = []
        targets = []
        
        print("Creating training data...")
        
        for midi_file in tqdm(midi_files):
            piano_rolls = self.midi_to_piano_roll(midi_file, max_length)
            
            if piano_rolls is None:
                continue
            
            # Create mixed input (sum of all instruments)
            mixed_roll = np.zeros_like(list(piano_rolls.values())[0])
            for class_roll in piano_rolls.values():
                mixed_roll += class_roll
            mixed_roll = np.clip(mixed_roll, 0, 1)
            
            # Skip if empty
            if np.sum(mixed_roll) == 0:
                continue
            
            # Stack individual instrument rolls as targets
            target_stack = np.stack([piano_rolls[class_name] 
                                   for class_name in self.instrument_classes.keys()], axis=-1)
            
            inputs.append(mixed_roll)
            targets.append(target_stack)
            
            # Data augmentation: pitch shifting
            if augment:
                for shift in [-2, -1, 1, 2]:  # Shift by semitones
                    shifted_input, shifted_target = self._pitch_shift(mixed_roll, target_stack, shift)
                    if shifted_input is not None:
                        inputs.append(shifted_input)
                        targets.append(shifted_target)
        
        return np.array(inputs), np.array(targets)
    
    def _pitch_shift(self, input_roll, target_stack, shift):
        """Apply pitch shifting augmentation"""
        if abs(shift) >= self.n_semitones:
            return None, None
        
        shifted_input = np.zeros_like(input_roll)
        shifted_target = np.zeros_like(target_stack)
        
        if shift > 0:
            shifted_input[:, shift:] = input_roll[:, :-shift]
            shifted_target[:, shift:, :] = target_stack[:, :-shift, :]
        elif shift < 0:
            shifted_input[:, :shift] = input_roll[:, -shift:]
            shifted_target[:, :shift, :] = target_stack[:, -shift:, :]
        else:
            shifted_input = input_roll
            shifted_target = target_stack
        
        return shifted_input, shifted_target

class MIDISeparationDataset(Dataset):
    """PyTorch Dataset for MIDI separation training"""
    
    def __init__(self, inputs, targets, sequence_length=64):
        """
        Initialize dataset
        
        Args:
            inputs: Mixed piano roll inputs
            targets: Separated instrument targets
            sequence_length: Length of sequences for training
        """
        self.inputs = inputs
        self.targets = targets
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_roll = self.inputs[idx]
        target_roll = self.targets[idx]
        
        # Extract random sequence if longer than sequence_length
        if input_roll.shape[0] > self.sequence_length:
            start_idx = np.random.randint(0, input_roll.shape[0] - self.sequence_length)
            input_seq = input_roll[start_idx:start_idx + self.sequence_length]
            target_seq = target_roll[start_idx:start_idx + self.sequence_length]
        else:
            # Pad if shorter
            input_seq = np.pad(input_roll, 
                             ((0, max(0, self.sequence_length - input_roll.shape[0])), (0, 0)), 
                             mode='constant')
            target_seq = np.pad(target_roll,
                              ((0, max(0, self.sequence_length - target_roll.shape[0])), (0, 0), (0, 0)),
                              mode='constant')
        
        return torch.FloatTensor(input_seq), torch.FloatTensor(target_seq)

class InstrumentSeparationModel(nn.Module):
    """Neural network for MIDI instrument separation"""
    
    def __init__(self, input_size=128, hidden_size=256, num_instruments=8, num_layers=3):
        """
        Initialize the separation model
        
        Args:
            input_size: Size of input (number of MIDI notes)
            hidden_size: Hidden layer size
            num_instruments: Number of instrument classes to separate
            num_layers: Number of LSTM layers
        """
        super(InstrumentSeparationModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_instruments = num_instruments
        
        # Encoder
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, 
                              batch_first=True, bidirectional=True)
        
        # Decoder for each instrument
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size, input_size),
                nn.Sigmoid()
            ) for _ in range(num_instruments)
        ])
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=8, batch_first=True)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_size)
            
        Returns:
            Separated instrument outputs
        """
        batch_size, seq_length, _ = x.shape
        
        # Encode
        encoded, _ = self.encoder(x)
        
        # Apply attention
        attended, _ = self.attention(encoded, encoded, encoded)
        
        # Decode for each instrument
        outputs = []
        for decoder in self.decoders:
            instrument_output = decoder(attended)
            outputs.append(instrument_output)
        
        # Stack outputs
        separated = torch.stack(outputs, dim=-1)  # (batch, seq, notes, instruments)
        
        return separated

class MIDISeparationTrainer:
    """Training pipeline for MIDI separation model"""
    
    def __init__(self, model, device='cpu'):
        """
        Initialize trainer
        
        Args:
            model: The separation model
            device: Training device ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.device = device
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001):
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            lr: Learning rate
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_inputs, batch_targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_inputs)
                loss = criterion(outputs, batch_targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_inputs, batch_targets in val_loader:
                    batch_inputs = batch_inputs.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    outputs = self.model(batch_inputs)
                    loss = criterion(outputs, batch_targets)
                    val_loss += loss.item()
            
            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
    
    def save_model(self, filepath):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', {'train_loss': [], 'val_loss': []})
        print(f"Model loaded from {filepath}")
    
    def plot_training_history(self):
        """Plot training history"""
        if not self.history['train_loss']:
            print("No training history available")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.show()

def separate_midi_file(model, processor, input_file, output_dir, device='cpu'):
    """
    Use trained model to separate a MIDI file
    
    Args:
        model: Trained separation model
        processor: MIDIDataProcessor instance
        input_file: Path to input MIDI file
        output_dir: Directory to save separated files
        device: Device to run inference on
    """
    model.eval()
    
    # Convert MIDI to piano roll
    piano_rolls = processor.midi_to_piano_roll(input_file)
    if piano_rolls is None:
        print(f"Error processing {input_file}")
        return []
    
    # Create mixed input
    mixed_roll = np.zeros_like(list(piano_rolls.values())[0])
    for class_roll in piano_rolls.values():
        mixed_roll += class_roll
    mixed_roll = np.clip(mixed_roll, 0, 1)
    
    # Convert to tensor and add batch dimension
    input_tensor = torch.FloatTensor(mixed_roll).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        separated = model(input_tensor)
    
    # Convert back to numpy
    separated_np = separated.squeeze(0).cpu().numpy()
    
    # Create piano rolls for each instrument
    separated_rolls = {}
    for i, class_name in enumerate(processor.instrument_classes.keys()):
        separated_rolls[class_name] = separated_np[:, :, i]
    
    # Convert to MIDI files
    output_files = processor.piano_roll_to_midi(
        separated_rolls, 
        output_dir, 
        Path(input_file).stem
    )
    
    return output_files

if __name__ == "__main__":
    print("🎵 MIDI Instrument Separation Model")
    print("=" * 40)
    print("This module provides tools for training a model to separate")
    print("MIDI files into individual instrument tracks.")
    print()
    print("Usage examples:")
    print("1. Create training data from Lakh MIDI dataset")
    print("2. Train the separation model")
    print("3. Use the model to separate new MIDI files")
