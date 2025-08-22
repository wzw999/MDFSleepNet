import numpy as np
from sklearn.model_selection import KFold


class kFoldGenerator:
    """
    K-Fold Cross Validation Data Generator for Sleep Stage Classification
    """
    
    def __init__(self, fold_data, fold_labels, fold_num, subject_num, random_state=42):
        """
        Initialize the k-fold generator
        
        Args:
            fold_data: List of data arrays for each fold
            fold_labels: List of label arrays for each fold
            fold_num: Number of folds
            subject_num: Number of subjects
            random_state: Random state for reproducibility
        """
        self.fold_data = fold_data
        self.fold_labels = fold_labels
        self.fold_num = fold_num
        self.subject_num = subject_num
        self.random_state = random_state
        
        # Convert labels to categorical if needed
        self.processed_labels = []
        for labels in fold_labels:
            if len(labels.shape) == 1:
                # Convert to one-hot encoding
                num_classes = len(np.unique(labels))
                categorical_labels = np.eye(num_classes)[labels.astype(int)]
                self.processed_labels.append(categorical_labels)
            else:
                self.processed_labels.append(labels)
    
    def getFold(self, fold_idx):
        """
        Get training and validation data for a specific fold
        
        Args:
            fold_idx: Index of the validation fold
            
        Returns:
            train_data, train_labels, val_data, val_labels
        """
        # Use the specified fold as validation set
        val_data = self.fold_data[fold_idx]
        val_labels = self.processed_labels[fold_idx]
        
        # Combine all other folds as training set
        train_data_list = []
        train_labels_list = []
        
        for i in range(self.fold_num):
            if i != fold_idx:
                train_data_list.append(self.fold_data[i])
                train_labels_list.append(self.processed_labels[i])
        
        # Concatenate training data
        train_data = np.concatenate(train_data_list, axis=0)
        train_labels = np.concatenate(train_labels_list, axis=0)
        
        return train_data, train_labels, val_data, val_labels
    
    def getAllFolds(self):
        """
        Get all folds data
        
        Returns:
            Generator yielding (fold_idx, train_data, train_labels, val_data, val_labels)
        """
        for fold_idx in range(self.fold_num):
            train_data, train_labels, val_data, val_labels = self.getFold(fold_idx)
            yield fold_idx, train_data, train_labels, val_data, val_labels
    
    def getDataStats(self):
        """
        Get statistics about the dataset
        
        Returns:
            Dictionary containing dataset statistics
        """
        total_samples = sum([len(data) for data in self.fold_data])
        
        # Calculate class distribution
        all_labels = np.concatenate([np.argmax(labels, axis=1) if len(labels.shape) > 1 
                                   else labels for labels in self.processed_labels])
        unique_classes, class_counts = np.unique(all_labels, return_counts=True)
        
        stats = {
            'total_samples': total_samples,
            'num_folds': self.fold_num,
            'num_subjects': self.subject_num,
            'samples_per_fold': [len(data) for data in self.fold_data],
            'class_distribution': dict(zip(unique_classes, class_counts)),
            'data_shape': self.fold_data[0].shape[1:] if len(self.fold_data) > 0 else None
        }
        
        return stats
    
    def printStats(self):
        """
        Print dataset statistics
        """
        stats = self.getDataStats()
        
        print("="*50)
        print("DATASET STATISTICS")
        print("="*50)
        print(f"Total samples: {stats['total_samples']}")
        print(f"Number of folds: {stats['num_folds']}")
        print(f"Number of subjects: {stats['num_subjects']}")
        print(f"Data shape: {stats['data_shape']}")
        
        print(f"\nSamples per fold:")
        for i, count in enumerate(stats['samples_per_fold']):
            print(f"  Fold {i}: {count} samples")
        
        print(f"\nClass distribution:")
        class_names = ['W', 'N1', 'N2', 'N3', 'REM']
        total = sum(stats['class_distribution'].values())
        for class_idx, count in stats['class_distribution'].items():
            class_name = class_names[int(class_idx)] if int(class_idx) < len(class_names) else f"Class_{int(class_idx)}"
            percentage = (count / total) * 100
            print(f"  {class_name}: {count} samples ({percentage:.2f}%)")
        print("="*50)


class SequentialDataGenerator:
    """
    Sequential data generator for time series data
    """
    
    def __init__(self, data, labels, sequence_length, overlap=0):
        """
        Initialize sequential data generator
        
        Args:
            data: Input data array
            labels: Label array
            sequence_length: Length of each sequence
            overlap: Overlap between sequences (0-1)
        """
        self.data = data
        self.labels = labels
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.step_size = int(sequence_length * (1 - overlap))
    
    def generate_sequences(self):
        """
        Generate sequences from the data
        
        Returns:
            sequences, sequence_labels
        """
        sequences = []
        sequence_labels = []
        
        for i in range(0, len(self.data) - self.sequence_length + 1, self.step_size):
            seq = self.data[i:i + self.sequence_length]
            label = self.labels[i + self.sequence_length - 1]  # Use the last label in sequence
            
            sequences.append(seq)
            sequence_labels.append(label)
        
        return np.array(sequences), np.array(sequence_labels)


def balance_dataset(data, labels, method='undersample', random_state=42):
    """
    Balance the dataset using different methods
    
    Args:
        data: Input data
        labels: Labels (one-hot encoded or integer)
        method: 'undersample' or 'oversample'
        random_state: Random state for reproducibility
        
    Returns:
        balanced_data, balanced_labels
    """
    np.random.seed(random_state)
    
    # Convert one-hot to integer labels if needed
    if len(labels.shape) > 1:
        int_labels = np.argmax(labels, axis=1)
        original_one_hot = True
    else:
        int_labels = labels
        original_one_hot = False
    
    unique_classes, class_counts = np.unique(int_labels, return_counts=True)
    
    if method == 'undersample':
        # Undersample to the minimum class count
        min_count = min(class_counts)
        
        balanced_indices = []
        for class_label in unique_classes:
            class_indices = np.where(int_labels == class_label)[0]
            selected_indices = np.random.choice(class_indices, min_count, replace=False)
            balanced_indices.extend(selected_indices)
        
        balanced_indices = np.array(balanced_indices)
        np.random.shuffle(balanced_indices)
        
        balanced_data = data[balanced_indices]
        balanced_labels = labels[balanced_indices]
        
    elif method == 'oversample':
        # Oversample to the maximum class count
        max_count = max(class_counts)
        
        balanced_data_list = []
        balanced_labels_list = []
        
        for class_label in unique_classes:
            class_indices = np.where(int_labels == class_label)[0]
            class_data = data[class_indices]
            class_labels = labels[class_indices]
            
            if len(class_indices) < max_count:
                # Oversample this class
                additional_indices = np.random.choice(len(class_indices), 
                                                    max_count - len(class_indices), 
                                                    replace=True)
                class_data = np.concatenate([class_data, class_data[additional_indices]])
                class_labels = np.concatenate([class_labels, class_labels[additional_indices]])
            
            balanced_data_list.append(class_data)
            balanced_labels_list.append(class_labels)
        
        balanced_data = np.concatenate(balanced_data_list)
        balanced_labels = np.concatenate(balanced_labels_list)
        
        # Shuffle the balanced dataset
        shuffle_indices = np.random.permutation(len(balanced_data))
        balanced_data = balanced_data[shuffle_indices]
        balanced_labels = balanced_labels[shuffle_indices]
    
    return balanced_data, balanced_labels
