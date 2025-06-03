#!/usr/bin/env python3
"""
Lakh MIDI Dataset Explorer
A comprehensive toolkit for exploring and analyzing the Lakh MIDI Dataset (LMD)

Dataset: Colin Raffel's Lakh MIDI Dataset v0.1
URL: https://colinraffel.com/projects/lmd/
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import pickle
from pathlib import Path
import pretty_midi
import librosa
from tqdm import tqdm
import warnings
import hashlib
import tarfile
import requests
from urllib.parse import urlparse
import time

warnings.filterwarnings('ignore')

class LakhMIDIExplorer:
    """
    A comprehensive explorer for the Lakh MIDI Dataset
    """
    
    def __init__(self, dataset_path="./lmd_data", cache_dir="./cache"):
        """
        Initialize the MIDI explorer
        
        Args:
            dataset_path: Path to the extracted Lakh MIDI dataset
            cache_dir: Directory to store cached analysis results
        """
        self.dataset_path = Path(dataset_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.dataset_path.mkdir(exist_ok=True)
        
        # Dataset components
        self.lmd_full_path = self.dataset_path / "lmd_full"
        self.lmd_matched_path = self.dataset_path / "lmd_matched"
        self.lmd_aligned_path = self.dataset_path / "lmd_aligned"
        
        # Metadata files
        self.md5_to_paths_file = self.dataset_path / "md5_to_paths.json"
        self.match_scores_file = self.dataset_path / "match_scores.json"
        
        # Analysis results
        self.analysis_results = {}
        self.midi_stats = {}
        
        # Download URLs
        self.download_urls = {
            'lmd_full': 'http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz',
            'lmd_matched': 'http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz',
            'lmd_aligned': 'http://hog.ee.columbia.edu/craffel/lmd/lmd_aligned.tar.gz',
            'md5_to_paths': 'http://hog.ee.columbia.edu/craffel/lmd/md5_to_paths.json',
            'match_scores': 'http://hog.ee.columbia.edu/craffel/lmd/match_scores.json',
            'clean_midi': 'http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz'
        }
        
    def download_dataset(self, components=['md5_to_paths', 'match_scores'], force_download=False):
        """
        Download specified components of the Lakh MIDI Dataset
        
        Args:
            components: List of components to download
            force_download: Force re-download even if files exist
        """
        print("🎵 Downloading Lakh MIDI Dataset components...")
        
        for component in components:
            if component not in self.download_urls:
                print(f"❌ Unknown component: {component}")
                continue
                
            url = self.download_urls[component]
            filename = Path(urlparse(url).path).name
            filepath = self.dataset_path / filename
            
            if filepath.exists() and not force_download:
                print(f"✅ {component} already exists: {filepath}")
                continue
                
            print(f"📥 Downloading {component}...")
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                with open(filepath, 'wb') as f, tqdm(
                    desc=filename,
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        size = f.write(chunk)
                        pbar.update(size)
                        
                print(f"✅ Downloaded: {filepath}")
                
                # Extract if it's a tar.gz file
                if filepath.suffix == '.gz' and filepath.stem.endswith('.tar'):
                    print(f"📦 Extracting {filepath}...")
                    with tarfile.open(filepath, 'r:gz') as tar:
                        tar.extractall(self.dataset_path)
                    print(f"✅ Extracted to: {self.dataset_path}")
                    
            except Exception as e:
                print(f"❌ Error downloading {component}: {e}")
    
    def load_metadata(self):
        """Load metadata files (MD5 to paths mapping and match scores)"""
        print("📂 Loading metadata...")
        
        # Load MD5 to paths mapping
        if self.md5_to_paths_file.exists():
            with open(self.md5_to_paths_file, 'r') as f:
                self.md5_to_paths = json.load(f)
            print(f"✅ Loaded {len(self.md5_to_paths)} MD5 to path mappings")
        else:
            print("⚠️  MD5 to paths file not found")
            self.md5_to_paths = {}
            
        # Load match scores
        if self.match_scores_file.exists():
            with open(self.match_scores_file, 'r') as f:
                self.match_scores = json.load(f)
            print(f"✅ Loaded {len(self.match_scores)} match scores")
        else:
            print("⚠️  Match scores file not found")
            self.match_scores = {}
    
    def scan_midi_files(self, subset='all', max_files=None):
        """
        Scan MIDI files in the dataset
        
        Args:
            subset: 'all', 'full', 'matched', or 'aligned'
            max_files: Maximum number of files to scan (for quick testing)
        """
        print(f"🔍 Scanning MIDI files ({subset})...")
        
        if subset == 'all' or subset == 'full':
            midi_paths = list(self.lmd_full_path.glob("**/*.mid*")) if self.lmd_full_path.exists() else []
        elif subset == 'matched':
            midi_paths = list(self.lmd_matched_path.glob("**/*.mid*")) if self.lmd_matched_path.exists() else []
        elif subset == 'aligned':
            midi_paths = list(self.lmd_aligned_path.glob("**/*.mid*")) if self.lmd_aligned_path.exists() else []
        else:
            midi_paths = []
            
        if max_files:
            midi_paths = midi_paths[:max_files]
            
        print(f"📁 Found {len(midi_paths)} MIDI files")
        return midi_paths
    
    def analyze_midi_file(self, midi_path):
        """
        Analyze a single MIDI file
        
        Args:
            midi_path: Path to MIDI file
            
        Returns:
            Dictionary with analysis results
        """
        try:
            midi = pretty_midi.PrettyMIDI(str(midi_path))
            
            analysis = {
                'filename': midi_path.name,
                'duration': midi.get_end_time(),
                'num_instruments': len(midi.instruments),
                'num_tracks': len([inst for inst in midi.instruments if not inst.is_drum]),
                'num_drum_tracks': len([inst for inst in midi.instruments if inst.is_drum]),
                'tempo_changes': len(midi.get_tempo_changes()[0]),
                'key_signatures': len(midi.key_signature_changes),
                'time_signatures': len(midi.time_signature_changes),
                'total_notes': sum(len(inst.notes) for inst in midi.instruments),
                'programs': [inst.program for inst in midi.instruments if not inst.is_drum],
                'instruments': [pretty_midi.program_to_instrument_name(inst.program) 
                              for inst in midi.instruments if not inst.is_drum],
                'has_lyrics': len(midi.lyrics) > 0,
                'has_text': len(midi.text_events) > 0,
                'pitch_range': self._get_pitch_range(midi),
                'tempo_stats': self._get_tempo_stats(midi),
                'valid': True
            }
            
            return analysis
            
        except Exception as e:
            return {
                'filename': midi_path.name,
                'error': str(e),
                'valid': False
            }
    
    def _get_pitch_range(self, midi):
        """Get pitch range statistics from MIDI"""
        all_pitches = []
        for instrument in midi.instruments:
            if not instrument.is_drum:
                all_pitches.extend([note.pitch for note in instrument.notes])
        
        if all_pitches:
            return {
                'min_pitch': min(all_pitches),
                'max_pitch': max(all_pitches),
                'pitch_range': max(all_pitches) - min(all_pitches),
                'mean_pitch': np.mean(all_pitches)
            }
        return {'min_pitch': 0, 'max_pitch': 0, 'pitch_range': 0, 'mean_pitch': 0}
    
    def _get_tempo_stats(self, midi):
        """Get tempo statistics from MIDI"""
        tempo_times, tempos = midi.get_tempo_changes()
        if len(tempos) > 0:
            return {
                'initial_tempo': tempos[0],
                'mean_tempo': np.mean(tempos),
                'tempo_std': np.std(tempos),
                'tempo_changes': len(tempos)
            }
        return {'initial_tempo': 120, 'mean_tempo': 120, 'tempo_std': 0, 'tempo_changes': 0}
    
    def batch_analyze(self, subset='matched', max_files=1000, save_cache=True):
        """
        Perform batch analysis of MIDI files
        
        Args:
            subset: Which subset to analyze
            max_files: Maximum number of files to analyze
            save_cache: Whether to save results to cache
        """
        cache_file = self.cache_dir / f"analysis_{subset}_{max_files}.pkl"
        
        if cache_file.exists():
            print(f"📦 Loading cached analysis from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.analysis_results = pickle.load(f)
            return self.analysis_results
        
        midi_paths = self.scan_midi_files(subset, max_files)
        
        if not midi_paths:
            print("❌ No MIDI files found!")
            return {}
        
        print(f"🎼 Analyzing {len(midi_paths)} MIDI files...")
        
        results = []
        valid_files = 0
        
        for midi_path in tqdm(midi_paths, desc="Analyzing MIDI files"):
            analysis = self.analyze_midi_file(midi_path)
            results.append(analysis)
            if analysis.get('valid', False):
                valid_files += 1
        
        self.analysis_results = {
            'files': results,
            'summary': {
                'total_files': len(results),
                'valid_files': valid_files,
                'invalid_files': len(results) - valid_files,
                'subset': subset
            }
        }
        
        if save_cache:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.analysis_results, f)
            print(f"💾 Saved analysis cache to {cache_file}")
        
        print(f"✅ Analysis complete: {valid_files}/{len(results)} files valid")
        return self.analysis_results
    
    def generate_statistics(self):
        """Generate comprehensive statistics from the analysis"""
        if not self.analysis_results:
            print("❌ No analysis results available. Run batch_analyze() first.")
            return
        
        valid_files = [f for f in self.analysis_results['files'] if f.get('valid', False)]
        
        if not valid_files:
            print("❌ No valid MIDI files to analyze.")
            return
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(valid_files)
        
        # Basic statistics
        stats = {
            'duration': {
                'mean': df['duration'].mean(),
                'median': df['duration'].median(),
                'std': df['duration'].std(),
                'min': df['duration'].min(),
                'max': df['duration'].max()
            },
            'instruments': {
                'mean_per_file': df['num_instruments'].mean(),
                'median_per_file': df['num_instruments'].median(),
                'max_per_file': df['num_instruments'].max()
            },
            'notes': {
                'mean_per_file': df['total_notes'].mean(),
                'median_per_file': df['total_notes'].median(),
                'total_notes': df['total_notes'].sum()
            },
            'tempo': {
                'mean_initial': df['tempo_stats'].apply(lambda x: x['initial_tempo']).mean(),
                'mean_avg': df['tempo_stats'].apply(lambda x: x['mean_tempo']).mean()
            }
        }
        
        # Most common instruments
        all_instruments = []
        for instruments in df['instruments']:
            all_instruments.extend(instruments)
        
        stats['common_instruments'] = Counter(all_instruments).most_common(10)
        
        # Most common programs
        all_programs = []
        for programs in df['programs']:
            all_programs.extend(programs)
        
        stats['common_programs'] = Counter(all_programs).most_common(10)
        
        self.midi_stats = stats
        return stats
    
    def create_visualizations(self, save_path=None):
        """Create comprehensive visualizations of the dataset"""
        if not self.analysis_results or not self.midi_stats:
            print("❌ No analysis results available. Run batch_analyze() and generate_statistics() first.")
            return
        
        valid_files = [f for f in self.analysis_results['files'] if f.get('valid', False)]
        df = pd.DataFrame(valid_files)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Lakh MIDI Dataset Analysis', fontsize=16, fontweight='bold')
        
        # 1. Duration distribution
        axes[0, 0].hist(df['duration'], bins=50, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Duration Distribution')
        axes[0, 0].set_xlabel('Duration (seconds)')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Number of instruments
        axes[0, 1].hist(df['num_instruments'], bins=range(0, df['num_instruments'].max()+2), 
                       alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('Number of Instruments per File')
        axes[0, 1].set_xlabel('Number of Instruments')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Total notes distribution
        axes[0, 2].hist(df['total_notes'], bins=50, alpha=0.7, color='salmon')
        axes[0, 2].set_title('Total Notes Distribution')
        axes[0, 2].set_xlabel('Total Notes')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_xscale('log')
        
        # 4. Tempo distribution
        initial_tempos = df['tempo_stats'].apply(lambda x: x['initial_tempo'])
        axes[1, 0].hist(initial_tempos, bins=50, alpha=0.7, color='gold')
        axes[1, 0].set_title('Initial Tempo Distribution')
        axes[1, 0].set_xlabel('Tempo (BPM)')
        axes[1, 0].set_ylabel('Frequency')
        
        # 5. Pitch range distribution
        pitch_ranges = df['pitch_range'].apply(lambda x: x['pitch_range'])
        axes[1, 1].hist(pitch_ranges, bins=50, alpha=0.7, color='mediumpurple')
        axes[1, 1].set_title('Pitch Range Distribution')
        axes[1, 1].set_xlabel('Pitch Range (semitones)')
        axes[1, 1].set_ylabel('Frequency')
        
        # 6. Most common instruments
        instruments = [item[0] for item in self.midi_stats['common_instruments'][:10]]
        counts = [item[1] for item in self.midi_stats['common_instruments'][:10]]
        
        axes[1, 2].barh(range(len(instruments)), counts, color='lightcoral')
        axes[1, 2].set_yticks(range(len(instruments)))
        axes[1, 2].set_yticklabels(instruments, fontsize=8)
        axes[1, 2].set_title('Most Common Instruments')
        axes[1, 2].set_xlabel('Frequency')
        
        # 7. Track types
        track_data = {
            'Regular Tracks': df['num_tracks'].sum(),
            'Drum Tracks': df['num_drum_tracks'].sum()
        }
        axes[2, 0].pie(track_data.values(), labels=track_data.keys(), autopct='%1.1f%%')
        axes[2, 0].set_title('Track Types Distribution')
        
        # 8. Files with lyrics/text
        text_data = {
            'With Lyrics': df['has_lyrics'].sum(),
            'Without Lyrics': len(df) - df['has_lyrics'].sum(),
            'With Text': df['has_text'].sum(),
            'Without Text': len(df) - df['has_text'].sum()
        }
        
        categories = ['Lyrics', 'Text Events']
        with_counts = [text_data['With Lyrics'], text_data['With Text']]
        without_counts = [text_data['Without Lyrics'], text_data['Without Text']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        axes[2, 1].bar(x - width/2, with_counts, width, label='With', color='lightblue')
        axes[2, 1].bar(x + width/2, without_counts, width, label='Without', color='lightgray')
        axes[2, 1].set_xlabel('Content Type')
        axes[2, 1].set_ylabel('Number of Files')
        axes[2, 1].set_title('Text Content in MIDI Files')
        axes[2, 1].set_xticks(x)
        axes[2, 1].set_xticklabels(categories)
        axes[2, 1].legend()
        
        # 9. Duration vs Number of Notes scatter plot
        axes[2, 2].scatter(df['duration'], df['total_notes'], alpha=0.6, color='darkblue')
        axes[2, 2].set_xlabel('Duration (seconds)')
        axes[2, 2].set_ylabel('Total Notes')
        axes[2, 2].set_title('Duration vs Total Notes')
        axes[2, 2].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualizations saved to {save_path}")
        
        plt.show()
    
    def print_summary_report(self):
        """Print a comprehensive summary report"""
        if not self.analysis_results or not self.midi_stats:
            print("❌ No analysis results available.")
            return
        
        print("\n" + "="*60)
        print("🎵 LAKH MIDI DATASET EXPLORATION REPORT")
        print("="*60)
        
        # Dataset overview
        summary = self.analysis_results['summary']
        print(f"\n📊 DATASET OVERVIEW")
        print(f"   Total files analyzed: {summary['total_files']:,}")
        print(f"   Valid MIDI files: {summary['valid_files']:,}")
        print(f"   Invalid files: {summary['invalid_files']:,}")
        print(f"   Success rate: {summary['valid_files']/summary['total_files']*100:.1f}%")
        
        # Duration statistics
        duration_stats = self.midi_stats['duration']
        print(f"\n⏱️  DURATION STATISTICS")
        print(f"   Mean duration: {duration_stats['mean']:.1f} seconds ({duration_stats['mean']/60:.1f} minutes)")
        print(f"   Median duration: {duration_stats['median']:.1f} seconds")
        print(f"   Shortest file: {duration_stats['min']:.1f} seconds")
        print(f"   Longest file: {duration_stats['max']:.1f} seconds ({duration_stats['max']/60:.1f} minutes)")
        
        # Instrument statistics
        inst_stats = self.midi_stats['instruments']
        print(f"\n🎼 INSTRUMENT STATISTICS")
        print(f"   Mean instruments per file: {inst_stats['mean_per_file']:.1f}")
        print(f"   Median instruments per file: {inst_stats['median_per_file']:.1f}")
        print(f"   Max instruments in a file: {inst_stats['max_per_file']}")
        
        # Note statistics
        note_stats = self.midi_stats['notes']
        print(f"\n🎵 NOTE STATISTICS")
        print(f"   Total notes in dataset: {note_stats['total_notes']:,}")
        print(f"   Mean notes per file: {note_stats['mean_per_file']:.0f}")
        print(f"   Median notes per file: {note_stats['median_per_file']:.0f}")
        
        # Most common instruments
        print(f"\n🎸 TOP 10 INSTRUMENTS")
        for i, (instrument, count) in enumerate(self.midi_stats['common_instruments'][:10], 1):
            print(f"   {i:2d}. {instrument}: {count:,} occurrences")
        
        # Tempo statistics
        tempo_stats = self.midi_stats['tempo']
        print(f"\n🥁 TEMPO STATISTICS")
        print(f"   Mean initial tempo: {tempo_stats['mean_initial']:.1f} BPM")
        print(f"   Mean average tempo: {tempo_stats['mean_avg']:.1f} BPM")
        
        print("\n" + "="*60)
        print("📈 Analysis complete! Use create_visualizations() to see charts.")
        print("="*60)

    def export_results(self, filename="lmd_analysis_results.json"):
        """Export analysis results to JSON file"""
        if not self.analysis_results:
            print("❌ No analysis results to export.")
            return
        
        export_data = {
            'analysis_results': self.analysis_results,
            'statistics': self.midi_stats,
            'metadata': {
                'dataset_path': str(self.dataset_path),
                'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_md5_mappings': len(self.md5_to_paths) if hasattr(self, 'md5_to_paths') else 0,
                'total_match_scores': len(self.match_scores) if hasattr(self, 'match_scores') else 0
            }
        }
        
        output_path = self.cache_dir / filename
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"📤 Results exported to {output_path}")
        return output_path

if __name__ == "__main__":
    # Example usage
    explorer = LakhMIDIExplorer()
    
    # Download metadata (small files)
    explorer.download_dataset(['md5_to_paths', 'match_scores'])
    
    # Load metadata
    explorer.load_metadata()
    
    print("\n🎵 Lakh MIDI Dataset Explorer initialized!")
    print("Use explorer.batch_analyze() to start analyzing MIDI files.")
