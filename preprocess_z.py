import numpy as np
import os


def preprocess_sleep_data(data_path, output_path, segment_length=30, overlap=0, 
                         sampling_rate=100, filter_artifacts=True):
    """
    Preprocess sleep EEG data for training
    
    Args:
        data_path: Path to raw sleep data
        output_path: Path to save processed data
        segment_length: Length of each segment in seconds
        overlap: Overlap between segments (0-1)
        sampling_rate: Sampling rate of the data
        filter_artifacts: Whether to filter artifacts
    """
    
    print(f"Preprocessing sleep data from: {data_path}")
    print(f"Segment length: {segment_length}s")
    print(f"Overlap: {overlap}")
    print(f"Sampling rate: {sampling_rate} Hz")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load raw data
    try:
        raw_data = np.load(data_path, allow_pickle=True)
        print(f"Raw data loaded successfully!")
        print(f"Data keys: {list(raw_data.keys())}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Extract data and labels
    if 'data' in raw_data and 'labels' in raw_data:
        eeg_data = raw_data['data']
        sleep_labels = raw_data['labels']
    else:
        print("Expected 'data' and 'labels' keys in the input file")
        return
    
    print(f"EEG data shape: {eeg_data.shape}")
    print(f"Labels shape: {sleep_labels.shape}")
    
    # Segment the data
    segment_samples = segment_length * sampling_rate
    step_size = int(segment_samples * (1 - overlap))
    
    segments = []
    segment_labels = []
    
    for i in range(0, len(eeg_data) - segment_samples + 1, step_size):
        segment = eeg_data[i:i + segment_samples]
        label = sleep_labels[i + segment_samples - 1]  # Use last label in segment
        
        if filter_artifacts and is_artifact(segment):
            continue
            
        segments.append(segment)
        segment_labels.append(label)
    
    segments = np.array(segments)
    segment_labels = np.array(segment_labels)
    
    print(f"Created {len(segments)} segments")
    print(f"Segments shape: {segments.shape}")
    
    # Create k-fold splits
    from sklearn.model_selection import StratifiedKFold
    
    k_folds = 10
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_data = []
    fold_labels = []
    fold_lengths = []
    
    for train_idx, test_idx in skf.split(segments, segment_labels):
        fold_data.append(segments[test_idx])
        fold_labels.append(to_categorical(segment_labels[test_idx], num_classes=5))
        fold_lengths.append(len(test_idx))
    
    # Save processed data
    np.savez_compressed(output_path,
                       Fold_data=fold_data,
                       Fold_label=fold_labels,
                       Fold_len=fold_lengths)
    
    print(f"Processed data saved to: {output_path}")
    
    # Print class distribution
    unique_labels, counts = np.unique(segment_labels, return_counts=True)
    print(f"\nClass distribution:")
    class_names = ['W', 'N1', 'N2', 'N3', 'REM']
    for label, count in zip(unique_labels, counts):
        if label < len(class_names):
            print(f"  {class_names[int(label)]}: {count} ({count/len(segment_labels)*100:.1f}%)")


def is_artifact(segment, threshold=500):
    """
    Simple artifact detection based on amplitude threshold
    
    Args:
        segment: EEG segment
        threshold: Amplitude threshold for artifact detection
        
    Returns:
        Boolean indicating if segment contains artifacts
    """
    max_amplitude = np.max(np.abs(segment))
    return max_amplitude > threshold


def to_categorical(y, num_classes):
    """
    Convert class vector to one-hot encoded matrix
    
    Args:
        y: Class vector
        num_classes: Number of classes
        
    Returns:
        One-hot encoded matrix
    """
    categorical = np.zeros((len(y), num_classes))
    for i, label in enumerate(y):
        if 0 <= label < num_classes:
            categorical[i, int(label)] = 1
    return categorical


def normalize_data(data, method='z-score'):
    """
    Normalize EEG data
    
    Args:
        data: Input data
        method: Normalization method ('z-score', 'min-max', 'robust')
        
    Returns:
        Normalized data
    """
    if method == 'z-score':
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        normalized = (data - mean) / (std + 1e-8)
    elif method == 'min-max':
        min_val = np.min(data, axis=0, keepdims=True)
        max_val = np.max(data, axis=0, keepdims=True)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
    elif method == 'robust':
        median = np.median(data, axis=0, keepdims=True)
        mad = np.median(np.abs(data - median), axis=0, keepdims=True)
        normalized = (data - median) / (mad + 1e-8)
    else:
        print(f"Unknown normalization method: {method}")
        return data
    
    return normalized


def create_different_segments(input_path, output_dir, segment_lengths=[1, 5, 10, 15, 30]):
    """
    Create multiple preprocessed datasets with different segment lengths
    
    Args:
        input_path: Path to raw data
        output_dir: Directory to save processed data
        segment_lengths: List of segment lengths in seconds
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for seg_len in segment_lengths:
        output_path = os.path.join(output_dir, f"sleep_edf_processed_{seg_len}s.npz")
        print(f"\nProcessing {seg_len}s segments...")
        preprocess_sleep_data(input_path, output_path, segment_length=seg_len)


if __name__ == "__main__":
    # Example usage
    
    # Paths (modify according to your data location)
    raw_data_path = "./datasets/raw/sleep_edf_raw.npz"
    output_dir = "./datasets/processed/"
    
    # Create segments of different lengths
    segment_lengths = [1, 5, 30]  # seconds
    
    print("Starting preprocessing...")
    create_different_segments(raw_data_path, output_dir, segment_lengths)
    print("Preprocessing completed!")