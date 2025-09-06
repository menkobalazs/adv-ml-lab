import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def explore_dataset_structure(base_dir):
    """
    Explore the dataset structure and print summary statistics
    
    Args:
        base_dir (str): Base directory containing genre subdirectories
        
    Returns:
        dict: Summary statistics about the dataset
    """
    print(f"Exploring dataset in {base_dir}...")
    
    # Initialize counters
    genres = []
    file_counts = {}
    total_files = 0
    file_durations = []
    
    # Explore each genre directory
    for genre in os.listdir(base_dir):
        genre_path = os.path.join(base_dir, genre)
        
        # Skip if not a directory
        if not os.path.isdir(genre_path):
            continue
            
        genres.append(genre)
        
        # Count files in this genre
        webm_files = [f for f in os.listdir(genre_path) if f.endswith('.webm')]
        file_counts[genre] = len(webm_files)
        total_files += len(webm_files)
        
        # Sample a few files to get duration estimates
        if webm_files:
            sample_files = webm_files[:min(5, len(webm_files))]
            for sample_file in sample_files:
                file_path = os.path.join(genre_path, sample_file)
                try:
                    y, sr = librosa.load(file_path, sr=None, duration=800)  # Load up to 2 minutes
                    duration = librosa.get_duration(y=y, sr=sr)
                    file_durations.append(duration)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    # Compile dataset summary
    dataset_summary = {
        "genres": genres,
        "file_counts": file_counts,
        "total_files": total_files,
        "avg_duration": np.mean(file_durations) if file_durations else None,
        "min_duration": np.min(file_durations) if file_durations else None,
        "max_duration": np.max(file_durations) if file_durations else None
    }
    
    # Print summary
    print("\nDataset Summary:")
    print(f"Found {len(genres)} genres: {', '.join(genres)}")
    print(f"Total files: {total_files}")
    
    print("\nFiles per genre:")
    for genre, count in file_counts.items():
        print(f"  - {genre}: {count} files")
    
    if file_durations:
        print(f"\nAudio Duration Statistics:")
        print(f"  - Average: {dataset_summary['avg_duration']:.2f} seconds")
        print(f"  - Minimum: {dataset_summary['min_duration']:.2f} seconds")
        print(f"  - Maximum: {dataset_summary['max_duration']:.2f} seconds")
    
    return dataset_summary

def extract_features_from_webm(file_path, duration=30, sr=22050, hop_length=512, segment_length=None):
    """
    Extract audio features from a .webm file
    
    Args:
        file_path (str): Path to .webm file
        duration (int): Maximum duration to analyze in seconds
        sr (int): Sample rate
        hop_length (int): Hop length for feature extraction
        segment_length (int): Optional fixed segment length in frames
        
    Returns:
        dict: Dictionary of sequential features
    """
    try:
        # Load audio file (first n seconds)
        y, sr = librosa.load(file_path, sr=sr, duration=duration)
        
        # Calculate the number of frames
        n_frames = len(y) // hop_length + 1
        
        # If segment_length is specified, we'll take segments from the middle
        if segment_length and n_frames > segment_length:
            # Take segment from the middle
            start_frame = (n_frames - segment_length) // 2
            end_frame = start_frame + segment_length
            start_sample = start_frame * hop_length
            end_sample = min(end_frame * hop_length, len(y))
            y = y[start_sample:end_sample]
        
        # Extract MFCCs (13 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
        
        # Extract Chroma features (12 pitches)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        
        # Extract Spectral Contrast (7 bands)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
        
        # Extract Onset Strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        
        # Extract Tempogram
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
        
        # Package all features as a dictionary of time series (each will be of shape [time_steps, n_features])
        features = {
            'mfcc': mfccs.T,             # Shape: [time_steps, 13]
            'chroma': chroma.T,          # Shape: [time_steps, 12]
            'contrast': contrast.T,       # Shape: [time_steps, 7]
            'onset': onset_env.reshape(-1, 1),  # Shape: [time_steps, 1]
            'tempogram': tempogram.T      # Shape: [time_steps, tempo_bins]
        }
        
        return features
        
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

def process_dataset(base_dir, output_dir, duration=30, max_files_per_genre=None):
    """
    Process the entire dataset and save extracted features
    
    Args:
        base_dir (str): Base directory containing genre subdirectories
        output_dir (str): Directory to save processed features
        duration (int): Duration in seconds to analyze for each file
        max_files_per_genre (int): Maximum number of files to process per genre
        
    Returns:
        pd.DataFrame: Metadata about processed files
    """
    
    # Initialize metadata list
    metadata = []
    
    # Process each genre directory
    for genre in os.listdir(base_dir):
        genre_path = os.path.join(base_dir, genre)
        
        # Skip if not a directory
        if not os.path.isdir(genre_path):
            continue
            
        print(f"\nProcessing genre: {genre}")
        
        # Get all webm files in this genre
        webm_files = [f for f in os.listdir(genre_path) if f.endswith('.webm')]
        
        # Limit files if needed
        if max_files_per_genre and len(webm_files) > max_files_per_genre:
            webm_files = webm_files[:max_files_per_genre]
        
        # Process each file
        for i, webm_file in enumerate(tqdm(webm_files, desc=f"Extracting features for {genre}")):
            file_path = os.path.join(genre_path, webm_file)
            
            # Extract features
            features = extract_features_from_webm(file_path, duration=duration)
            
            if features is not None:
                # Create unique ID for this file
                file_id = f"{genre}_{i:04d}"
                
                # Save features to disk
                feature_path = os.path.join(output_dir, f"{file_id}.npz")
                np.savez(feature_path, 
                         mfcc=features['mfcc'],
                         chroma=features['chroma'],
                         contrast=features['contrast'],
                         onset=features['onset'],
                         tempogram=features['tempogram'])
                
                # Add metadata
                metadata.append({
                    'file_id': file_id,
                    'original_file': webm_file,
                    'genre': genre,
                    'feature_path': feature_path,
                    'mfcc_shape': features['mfcc'].shape,
                    'chroma_shape': features['chroma'].shape,
                    'contrast_shape': features['contrast'].shape,
                    'onset_shape': features['onset'].shape,
                    'tempogram_shape': features['tempogram'].shape
                })
    
    # Create metadata DataFrame
    metadata_df = pd.DataFrame(metadata)
    
    # Save metadata
    metadata_df.to_csv(os.path.join(output_dir, 'others/metadata.csv'), index=False)
    
    print(f"\nProcessing complete. Features saved to {output_dir}others/")
    print(f"Processed {len(metadata_df)} files across {metadata_df['genre'].nunique()} genres")
    
    return metadata_df

def visualize_features(metadata_df, feature_dir, num_samples=3):
    """
    Visualize extracted features for a few samples from each genre
    
    Args:
        metadata_df (pd.DataFrame): Metadata about processed files
        feature_dir (str): Directory containing feature files
        num_samples (int): Number of samples to visualize per genre
    """
    # Get unique genres
    genres = metadata_df['genre'].unique()
    
    # Create a figure
    plt.figure(figsize=(15, 5 * len(genres) * num_samples))
    
    # Plot for each genre
    sample_idx = 0
    for genre in genres:
        # Get files for this genre
        genre_files = metadata_df[metadata_df['genre'] == genre].sample(num_samples)
        
        # Visualize each sample
        for _, row in genre_files.iterrows():
            # Load features
            features = np.load(row['feature_path'])
            
            # MFCC visualization
            plt.subplot(len(genres)*num_samples, 2, sample_idx*2 + 1)
            plt.title(f"{genre} - MFCCs")
            librosa.display.specshow(features['mfcc'].T, x_axis='time')
            plt.colorbar(format='%+2.0f')
            
            # Chroma visualization
            plt.subplot(len(genres)*num_samples, 2, sample_idx*2 + 2)
            plt.title(f"{genre} - Chroma")
            librosa.display.specshow(features['chroma'].T, y_axis='chroma', x_axis='time')
            plt.colorbar()
            
            sample_idx += 1
    
    plt.tight_layout()
    plt.savefig(os.path.join(feature_dir, 'others/feature_visualization.png'))
    plt.show()


if __name__ == "__main__":
    print('--- Start program ---')
    base_dir = "genres/" 
    output_dir = "processed_features/"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir+"others/", exist_ok=True)

    print('--- Step 1: Explore dataset structure ---')
    dataset_summary = explore_dataset_structure(base_dir)
    
    print('--- Step 2: Process the dataset and extract features ---')
    metadata_df = process_dataset(base_dir, output_dir=output_dir, duration=30)
    
    print('--- Step 3: Visualize some features ---')
    visualize_features(metadata_df, output_dir, num_samples=2)
    
    print("Dataset preprocessing complete!")
    print('--- End program ---\n\n')
