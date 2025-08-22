"""
Sleep-EDF-2013 Dataset Preparation Script

This script downloads and preprocesses the Sleep-EDF-2013 dataset for sleep stage classification.
The dataset contains whole-night polysomnographic sleep recordings.
"""

import numpy as np
import os
import urllib.request
import gzip
import shutil
from scipy.io import loadmat
import mne


def download_sleep_edf_2013(data_dir="./datasets/sleep-edf-2013/raw/"):
    """
    Download Sleep-EDF-2013 dataset from PhysioNet
    
    Args:
        data_dir: Directory to store downloaded data
    """
    os.makedirs(data_dir, exist_ok=True)
    
    base_url = "https://physionet.org/files/sleep-edfx/1.0.0/"
    
    # List of files to download (subset for example)
    files_to_download = [
        "sleep-cassette/SC4001E0-PSG.edf",
        "sleep-cassette/SC4001E0-PSG.edf.txt",
        "sleep-cassette/SC4002E0-PSG.edf", 
        "sleep-cassette/SC4002E0-PSG.edf.txt",
        # Add more files as needed
    ]
    
    print("Downloading Sleep-EDF-2013 dataset...")
    
    for file_path in files_to_download:
        url = base_url + file_path
        local_path = os.path.join(data_dir, os.path.basename(file_path))
        
        if not os.path.exists(local_path):
            print(f"Downloading {file_path}...")
            try:
                urllib.request.urlretrieve(url, local_path)
                print(f"Downloaded: {local_path}")
            except Exception as e:
                print(f"Error downloading {file_path}: {e}")
        else:
            print(f"Already exists: {local_path}")


def read_edf_file(edf_path):
    """
    Read EDF file and extract EEG data
    
    Args:
        edf_path: Path to EDF file
        
    Returns:
        data: EEG data array
        sfreq: Sampling frequency
        ch_names: Channel names
    """
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        
        # Select EEG channels (modify based on your needs)
        eeg_channels = ['EEG Fpz-Cz', 'EEG Pz-Oz']  # Common channels
        available_channels = [ch for ch in eeg_channels if ch in raw.ch_names]
        
        if not available_channels:
            print(f"No EEG channels found in {edf_path}")
            return None, None, None
        
        raw.pick_channels(available_channels)
        data = raw.get_data()
        sfreq = raw.info['sfreq']
        
        return data, sfreq, available_channels
        
    except Exception as e:
        print(f"Error reading {edf_path}: {e}")
        return None, None, None


def read_annotation_file(annotation_path):
    """
    Read sleep stage annotations
    
    Args:
        annotation_path: Path to annotation file
        
    Returns:
        annotations: List of sleep stage annotations
    """
    annotations = []
    
    try:
        with open(annotation_path, 'r') as f:
            for line in f:
                if 'Sleep stage' in line:
                    # Extract sleep stage information
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        stage = parts[-1]  # Last part should be the stage
                        annotations.append(stage)
        
        return annotations
        
    except Exception as e:
        print(f"Error reading annotation file {annotation_path}: {e}")
        return []


def convert_sleep_stages(annotations):
    """
    Convert sleep stage annotations to numerical labels
    
    Sleep stages mapping:
    - W: 0 (Wake)
    - 1: 1 (NREM Stage 1)  
    - 2: 2 (NREM Stage 2)
    - 3: 3 (NREM Stage 3)
    - 4: 3 (NREM Stage 4 -> merged with Stage 3)
    - R: 4 (REM Sleep)
    
    Args:
        annotations: List of sleep stage strings
        
    Returns:
        numerical_labels: Array of numerical labels
    """
    stage_mapping = {
        'W': 0, 'Wake': 0, '0': 0,
        '1': 1, 'S1': 1, 'N1': 1,
        '2': 2, 'S2': 2, 'N2': 2,
        '3': 3, 'S3': 3, 'N3': 3,
        '4': 3, 'S4': 3,  # Merge Stage 4 with Stage 3
        'R': 4, 'REM': 4, '5': 4
    }
    
    numerical_labels = []
    for annotation in annotations:
        # Clean annotation string
        clean_annotation = annotation.strip().upper()
        
        if clean_annotation in stage_mapping:
            numerical_labels.append(stage_mapping[clean_annotation])
        else:
            # Default to wake if unknown
            numerical_labels.append(0)
            print(f"Unknown sleep stage: {annotation}, defaulting to Wake")
    
    return np.array(numerical_labels)


def preprocess_sleep_edf_2013(raw_dir, output_dir, segment_length=30):
    """
    Preprocess Sleep-EDF-2013 dataset
    
    Args:
        raw_dir: Directory containing raw EDF files
        output_dir: Directory to save processed data
        segment_length: Segment length in seconds
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_data = []
    all_labels = []
    subjects = []
    
    # Find all EDF files
    edf_files = [f for f in os.listdir(raw_dir) if f.endswith('.edf')]
    
    print(f"Found {len(edf_files)} EDF files")
    
    for edf_file in edf_files:
        print(f"Processing {edf_file}...")
        
        edf_path = os.path.join(raw_dir, edf_file)
        annotation_path = edf_path + ".txt"  # Annotation file
        
        if not os.path.exists(annotation_path):
            print(f"Annotation file not found: {annotation_path}")
            continue
        
        # Read EEG data
        data, sfreq, ch_names = read_edf_file(edf_path)
        if data is None:
            continue
        
        # Read annotations
        annotations = read_annotation_file(annotation_path)
        if not annotations:
            continue
        
        # Convert annotations to numerical labels
        labels = convert_sleep_stages(annotations)
        
        # Segment the data
        segment_samples = int(segment_length * sfreq)
        n_segments = min(len(labels), data.shape[1] // segment_samples)
        
        for i in range(n_segments):
            start_sample = i * segment_samples
            end_sample = start_sample + segment_samples
            
            segment_data = data[:, start_sample:end_sample]
            segment_label = labels[i] if i < len(labels) else 0
            
            # Quality check - skip if segment has artifacts
            if np.max(np.abs(segment_data)) > 500:  # Simple artifact detection
                continue
            
            all_data.append(segment_data.T)  # Transpose to (time, channels)
            all_labels.append(segment_label)
            subjects.append(edf_file)
    
    print(f"Total segments: {len(all_data)}")
    
    # Convert to numpy arrays
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    
    print(f"Data shape: {all_data.shape}")
    print(f"Labels shape: {all_labels.shape}")
    
    # Create k-fold splits by subjects
    from sklearn.model_selection import StratifiedKFold
    
    unique_subjects = list(set(subjects))
    subject_labels = [all_labels[subjects.index(sub)] for sub in unique_subjects]
    
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    fold_data = []
    fold_labels = []
    fold_lengths = []
    
    for train_idx, test_idx in kfold.split(unique_subjects, subject_labels):
        test_subjects = [unique_subjects[i] for i in test_idx]
        
        # Get all segments from test subjects
        test_indices = [i for i, sub in enumerate(subjects) if sub in test_subjects]
        
        fold_data.append(all_data[test_indices])
        fold_labels.append(to_categorical(all_labels[test_indices], 5))
        fold_lengths.append(len(test_indices))
    
    # Save processed data
    output_path = os.path.join(output_dir, f"sleep_edf_processed_{segment_length}s.npz")
    np.savez_compressed(output_path,
                       Fold_data=fold_data,
                       Fold_label=fold_labels,
                       Fold_len=fold_lengths)
    
    print(f"Processed data saved to: {output_path}")
    
    # Print statistics
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    print("\nClass distribution:")
    for label, count in zip(unique_labels, counts):
        if label < len(class_names):
            print(f"  {class_names[int(label)]}: {count} ({count/len(all_labels)*100:.1f}%)")


def to_categorical(y, num_classes):
    """Convert class vector to one-hot encoded matrix"""
    categorical = np.zeros((len(y), num_classes))
    for i, label in enumerate(y):
        if 0 <= label < num_classes:
            categorical[i, int(label)] = 1
    return categorical


if __name__ == "__main__":
    # Configuration
    raw_data_dir = "./datasets/sleep-edf-2013/raw/"
    output_dir = "./datasets/sleep-edf-2013/npz/"
    
    # Download dataset (uncomment if needed)
    # download_sleep_edf_2013(raw_data_dir)
    
    # Process dataset
    print("Starting Sleep-EDF-2013 preprocessing...")
    
    # Create different segment lengths
    for seg_len in [30, 5]:
        print(f"\nProcessing {seg_len}s segments...")
        preprocess_sleep_edf_2013(raw_data_dir, output_dir, seg_len)
    
    print("Sleep-EDF-2013 preprocessing completed!")
