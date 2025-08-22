"""
Sleep-EDF-2018 Dataset Preparation Script

This script downloads and preprocesses the Sleep-EDF-2018 dataset for sleep stage classification.
"""

import numpy as np
import os
import urllib.request


def download_sleep_edf_2018(data_dir="./datasets/sleep-edf-2018/raw/"):
    """
    Download Sleep-EDF-2018 dataset from PhysioNet
    
    Args:
        data_dir: Directory to store downloaded data
    """
    os.makedirs(data_dir, exist_ok=True)
    
    base_url = "https://physionet.org/files/sleep-edfx/1.0.0/"
    
    # List of files to download (subset for example)
    files_to_download = [
        "sleep-telemetry/ST7011J0-PSG.edf",
        "sleep-telemetry/ST7011J0-PSG.edf.txt",
        "sleep-telemetry/ST7022J0-PSG.edf",
        "sleep-telemetry/ST7022J0-PSG.edf.txt",
        # Add more files as needed
    ]
    
    print("Downloading Sleep-EDF-2018 dataset...")
    
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


def create_synthetic_sleep_edf_2018_data(output_dir, n_subjects=20, n_epochs_per_subject=1000):
    """
    Create synthetic Sleep-EDF-2018 data for demonstration purposes
    
    Args:
        output_dir: Directory to save synthetic data
        n_subjects: Number of synthetic subjects
        n_epochs_per_subject: Number of epochs per subject
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating synthetic Sleep-EDF-2018 data for demonstration...")
    
    # Sampling rate and channels
    sfreq = 100  # Hz
    n_channels = 4  # EEG channels
    epoch_length = 30  # seconds
    samples_per_epoch = sfreq * epoch_length
    
    all_data = []
    all_labels = []
    all_subjects = []
    
    # Sleep stage probabilities (realistic distribution)
    stage_probs = [0.15, 0.05, 0.45, 0.20, 0.15]  # W, N1, N2, N3, REM
    
    for subject_id in range(n_subjects):
        print(f"Creating synthetic subject {subject_id + 1}/{n_subjects}")
        
        subject_data = []
        subject_labels = []
        
        for epoch in range(n_epochs_per_subject):
            # Generate synthetic EEG data
            # Different frequency characteristics for different sleep stages
            sleep_stage = np.random.choice(5, p=stage_probs)
            
            epoch_data = generate_synthetic_eeg_epoch(sleep_stage, samples_per_epoch, 
                                                     n_channels, sfreq)
            
            subject_data.append(epoch_data)
            subject_labels.append(sleep_stage)
            all_subjects.append(subject_id)
        
        all_data.extend(subject_data)
        all_labels.extend(subject_labels)
    
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    
    print(f"Generated data shape: {all_data.shape}")
    print(f"Generated labels shape: {all_labels.shape}")
    
    # Create k-fold splits by subjects
    fold_data = []
    fold_labels = []
    fold_lengths = []
    
    subjects_per_fold = n_subjects // 10
    
    for fold in range(10):
        start_subject = fold * subjects_per_fold
        end_subject = min((fold + 1) * subjects_per_fold, n_subjects)
        
        # Get indices for this fold's subjects
        fold_indices = [i for i, subj in enumerate(all_subjects) 
                       if start_subject <= subj < end_subject]
        
        if fold_indices:
            fold_data.append(all_data[fold_indices])
            fold_labels.append(to_categorical(all_labels[fold_indices], 5))
            fold_lengths.append(len(fold_indices))
        else:
            # Empty fold
            fold_data.append(np.empty((0, samples_per_epoch, n_channels)))
            fold_labels.append(np.empty((0, 5)))
            fold_lengths.append(0)
    
    # Save for 30s segments
    output_path_30s = os.path.join(output_dir, "sleep_edf_processed_30s.npz")
    np.savez_compressed(output_path_30s,
                       Fold_data=fold_data,
                       Fold_label=fold_labels,
                       Fold_len=fold_lengths)
    
    # Create 5s segments
    samples_per_5s = sfreq * 5
    fold_data_5s = []
    fold_labels_5s = []
    fold_lengths_5s = []
    
    for fold_idx in range(10):
        fold_30s_data = fold_data[fold_idx]
        fold_30s_labels = fold_labels[fold_idx]
        
        if len(fold_30s_data) == 0:
            fold_data_5s.append(np.empty((0, samples_per_5s, n_channels)))
            fold_labels_5s.append(np.empty((0, 5)))
            fold_lengths_5s.append(0)
            continue
        
        # Split each 30s epoch into 6 x 5s segments
        fold_5s_segments = []
        fold_5s_segment_labels = []
        
        for epoch_idx in range(len(fold_30s_data)):
            epoch_30s = fold_30s_data[epoch_idx]
            epoch_label = fold_30s_labels[epoch_idx]
            
            # Create 6 segments of 5s each
            for seg_idx in range(6):
                start_sample = seg_idx * samples_per_5s
                end_sample = start_sample + samples_per_5s
                
                segment_5s = epoch_30s[start_sample:end_sample]
                fold_5s_segments.append(segment_5s)
                fold_5s_segment_labels.append(epoch_label)
        
        fold_data_5s.append(np.array(fold_5s_segments))
        fold_labels_5s.append(np.array(fold_5s_segment_labels))
        fold_lengths_5s.append(len(fold_5s_segments))
    
    # Save for 5s segments
    output_path_5s = os.path.join(output_dir, "sleep_edf_processed_5s.npz")
    np.savez_compressed(output_path_5s,
                       Fold_data=fold_data_5s,
                       Fold_label=fold_labels_5s,
                       Fold_len=fold_lengths_5s)
    
    print(f"30s data saved to: {output_path_30s}")
    print(f"5s data saved to: {output_path_5s}")
    
    # Print statistics
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    print("\nClass distribution:")
    for label, count in zip(unique_labels, counts):
        if label < len(class_names):
            print(f"  {class_names[int(label)]}: {count} ({count/len(all_labels)*100:.1f}%)")


def generate_synthetic_eeg_epoch(sleep_stage, n_samples, n_channels, sfreq):
    """
    Generate synthetic EEG epoch with stage-specific characteristics
    
    Args:
        sleep_stage: Sleep stage (0-4)
        n_samples: Number of samples
        n_channels: Number of channels
        sfreq: Sampling frequency
        
    Returns:
        synthetic_epoch: Synthetic EEG epoch
    """
    # Time vector
    t = np.linspace(0, n_samples / sfreq, n_samples)
    
    # Stage-specific frequency characteristics
    if sleep_stage == 0:  # Wake
        # Higher frequency, lower amplitude
        base_freq = 10  # Alpha rhythm
        amplitude = 30
        noise_level = 15
    elif sleep_stage == 1:  # N1
        # Mixed frequencies
        base_freq = 7
        amplitude = 40
        noise_level = 20
    elif sleep_stage == 2:  # N2
        # Sleep spindles and K-complexes
        base_freq = 12  # Sleep spindles
        amplitude = 50
        noise_level = 10
    elif sleep_stage == 3:  # N3
        # Delta waves (slow wave sleep)
        base_freq = 2
        amplitude = 80
        noise_level = 5
    else:  # REM (stage 4)
        # Similar to wake but with different patterns
        base_freq = 8
        amplitude = 35
        noise_level = 18
    
    epoch_data = np.zeros((n_samples, n_channels))
    
    for ch in range(n_channels):
        # Base signal with stage-specific characteristics
        signal = amplitude * np.sin(2 * np.pi * base_freq * t)
        
        # Add harmonics
        signal += (amplitude * 0.3) * np.sin(2 * np.pi * base_freq * 2 * t)
        signal += (amplitude * 0.1) * np.sin(2 * np.pi * base_freq * 3 * t)
        
        # Add noise
        noise = np.random.normal(0, noise_level, n_samples)
        signal += noise
        
        # Add slow drift
        drift = 5 * np.sin(2 * np.pi * 0.1 * t)
        signal += drift
        
        # Channel-specific variations
        channel_factor = 0.8 + 0.4 * np.random.random()
        signal *= channel_factor
        
        epoch_data[:, ch] = signal
    
    return epoch_data


def to_categorical(y, num_classes):
    """Convert class vector to one-hot encoded matrix"""
    categorical = np.zeros((len(y), num_classes))
    for i, label in enumerate(y):
        if 0 <= label < num_classes:
            categorical[i, int(label)] = 1
    return categorical


if __name__ == "__main__":
    # Configuration
    output_dir = "./datasets/sleepedf-2018/npz/"
    
    # Create synthetic data for demonstration
    print("Creating synthetic Sleep-EDF-2018 dataset...")
    create_synthetic_sleep_edf_2018_data(output_dir, n_subjects=20, n_epochs_per_subject=800)
    
    print("Sleep-EDF-2018 data creation completed!")
    print("\nNote: This is synthetic data for demonstration purposes.")
    print("For real data, download the actual Sleep-EDF-2018 dataset from PhysioNet.")
