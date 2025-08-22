import numpy as np
# sys.path.append('../input')
import os
from Utils import *
from DataGenerator import *


# Dataset paths - uncomment the dataset you want to use

# ISRUC-S3 Dataset
# Path_30 = "./datasets/ISRUC/ISRUC_S3/ISRUC_S3_all.npz"
# Path_5 = "./datasets/ISRUC/ISRUC_S3/ISRUC_S3_5s_ham.npz"
# output_path = "./MFE/output_s3/test30_5/"

# ISRUC-S1 Dataset
# Path_30 = "./datasets/ISRUC/ISRUC_S1/ISRUC_S1_all.npz"
# Path_5 = "./datasets/ISRUC/ISRUC_S1/ISRUC_S1_5s_ham.npz"
# output_path = "./MFE/output_s1/test30_5/"

# Sleep-EDF-2013 Dataset
# Path_30 = "./datasets/sleepedf-2013/npz/sleep_edf_processed.npz"
# Path_5 = "./datasets/sleepedf-2013/npz/sleep_edf_processed_5s.npz"
# output_path = "./MFE/output_sleepedf2013/test30_5/"

# Sleep-EDF-2018 Dataset (default)
Path_30 = "./datasets/sleepedf-2018/npz/sleep_edf_processed_30s.npz"
Path_5 = "./datasets/sleepedf-2018/npz/sleep_edf_processed_5s.npz"
output_path = "./MFE/output_sleepedf2018/test30_5/"


# Load data
try:
    ReadList_30 = np.load(Path_30, allow_pickle=True)
    Fold_Num_30 = ReadList_30['Fold_len']    # Num of samples of each fold
    Fold_Data_30 = ReadList_30['Fold_data']   # Data of each fold
    Fold_Label_30 = ReadList_30['Fold_label']  # Labels of each fold

    ReadList_5 = np.load(Path_5, allow_pickle=True)
    Fold_Num_5 = ReadList_5['Fold_len']    # Num of samples of each fold
    Fold_Data_5 = ReadList_5['Fold_data']   # Data of each fold
    Fold_Label_5 = ReadList_5['Fold_label']  # Labels of each fold
    
    print("Data loaded successfully!")
    print(f"30s data shape: {Fold_Data_30[0].shape if len(Fold_Data_30) > 0 else 'Empty'}")
    print(f"5s data shape: {Fold_Data_5[0].shape if len(Fold_Data_5) > 0 else 'Empty'}")

except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please make sure the dataset files exist in the specified paths.")
    exit(1)


# Configuration
freq = 100
channels = 4
subject_num = len(Fold_Num_30)
fold = 10
seg_30 = 30
seg_5 = 5

cfg = {
    'bs': 32,           # Batch size
    'epochs': 100,      # Number of epochs
    'lr': 0.001,        # Learning rate
    'patience': 15,     # Early stopping patience
    'monitor': 'val_loss',  # Metric to monitor for early stopping
}

# Create data generators
DataGenerator_30 = kFoldGenerator(Fold_Data_30, Fold_Label_30, fold, subject_num)
DataGenerator_5 = kFoldGenerator(Fold_Data_5, Fold_Label_5, fold, subject_num)

# Print dataset statistics
print("\n30s Dataset Statistics:")
DataGenerator_30.printStats()

print("\n5s Dataset Statistics:")
DataGenerator_5.printStats()

# Clean up memory
del ReadList_30, ReadList_5, Fold_Data_30, Fold_Data_5, Fold_Label_30, Fold_Label_5

# Create output directory
if not os.path.exists(output_path):
    os.makedirs(output_path)


# Import TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.layers import (Conv1D, Dense, Dropout, Flatten, MaxPooling1D, 
                                       Activation, BatchNormalization, Add, Reshape, 
                                       TimeDistributed, Input, GlobalAveragePooling1D)
    from tensorflow.keras.models import Model
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}")

except ImportError as e:
    print(f"Error importing TensorFlow: {e}")
    print("Please install TensorFlow: pip install tensorflow")
    exit(1)

try:
    from sklearn import metrics
    print("Scikit-learn imported successfully")
except ImportError as e:
    print(f"Error importing scikit-learn: {e}")
    print("Please install scikit-learn: pip install scikit-learn")
    exit(1)

import random as python_random
import gc

# Set seeds for reproducibility
os.environ['TF_DETERMINISTIC_OPS'] = '1'
np.random.seed(42)
python_random.seed(42)
tf.random.set_seed(42)

# GPU configuration
tf.config.set_soft_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPUs available. Using CPU for computations.")


def CNN_block(inputs, filters=32, kernel_size=5, pool_size=3, weight_decay=0.001):
    """CNN block with Conv1D, BatchNorm, ReLU, and MaxPooling"""
    x = Conv1D(filters, kernel_size, 1, padding='same', 
              kernel_regularizer=l2(weight_decay))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size, 2, padding='same')(x)
    return x


def ResNet_block(inputs, filters=32, kernel_size_1=5, pool_size_1=3, 
                kernel_size_2=5, pool_size_2=3, weight_decay=0.001):
    """ResNet block with two CNN blocks and skip connection"""
    # Main path
    x = CNN_block(inputs, filters, kernel_size_1, pool_size_1, weight_decay)
    x = CNN_block(x, filters, kernel_size_2, pool_size_2, weight_decay)
    
    # Skip connection
    shortcut = Conv1D(filters, 1, 2, padding='same')(inputs)
    shortcut = Conv1D(filters, 1, 2, padding='same')(shortcut)
    
    return Add()([x, shortcut])


def create_model(input_shape, channels, time_seconds, freq=100, num_classes=5):
    """
    Create ResNet-based model for sleep stage classification
    
    Args:
        input_shape: Shape of input data (time_points, channels)
        channels: Number of EEG channels
        time_seconds: Duration in seconds
        freq: Sampling frequency
        num_classes: Number of sleep stages
        
    Returns:
        Compiled Keras model
    """
    # Single channel feature extractor
    inputs_channel = Input(shape=(time_seconds * freq, 1))
    
    # Multi-scale ResNet blocks
    x = ResNet_block(inputs_channel, 16)
    x = Dropout(0.2)(x)
    x = ResNet_block(x, 32)
    x = Dropout(0.2)(x)
    x = ResNet_block(x, 64)
    x = Dropout(0.2)(x)
    x = ResNet_block(x, 128)
    x = Dropout(0.2)(x)
    
    # Global average pooling for feature extraction
    outputs = GlobalAveragePooling1D()(x)
    
    # Feature extractor model
    feature_extractor = Model(inputs=inputs_channel, outputs=outputs)
    
    # Multi-channel model
    inputs = Input(shape=input_shape)
    input_reshaped = Reshape((channels, time_seconds * freq, 1))(inputs)
    
    # Extract features from each channel
    features_all = TimeDistributed(feature_extractor)(input_reshaped)
    
    # Flatten and add dropout
    flattened_features = Flatten()(features_all)
    flattened_features = Dropout(0.5)(flattened_features)
    
    # Classification head
    dense_features = Dense(128, activation='relu')(flattened_features)
    dense_features = Dropout(0.3)(dense_features)
    outputs = Dense(num_classes, activation='softmax', name='predictions')(dense_features)
    
    # Create final model
    model = Model(inputs, outputs)
    
    return model


def train_model(model, train_data, train_labels, val_data, val_labels, 
               fold_idx, model_name, save_path):
    """Train a model with callbacks"""
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor=cfg['monitor'],
        patience=cfg['patience'],
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(save_path, f"{model_name}_fold_{fold_idx}_best.h5"),
        monitor=cfg['monitor'],
        save_best_only=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor=cfg['monitor'],
        factor=0.5,
        patience=cfg['patience']//2,
        min_lr=1e-7,
        verbose=1
    )
    
    callbacks = [early_stopping, model_checkpoint, reduce_lr]
    
    # Train model
    history = model.fit(
        train_data, train_labels,
        batch_size=cfg['bs'],
        epochs=cfg['epochs'],
        validation_data=(val_data, val_labels),
        callbacks=callbacks,
        verbose=1
    )
    
    return history


# Cross-validation results storage
all_scores_30s = []
all_scores_5s = []
all_predictions_30s = []
all_true_labels_30s = []
all_predictions_5s = []
all_true_labels_5s = []

print(f"\nStarting {fold}-fold cross-validation...")

for i in range(fold):
    print(f"\n{'='*60}")
    print(f"FOLD {i+1}/{fold}")
    print(f"{'='*60}")
    
    # Get fold data
    train_data_30s, train_labels_30s, val_data_30s, val_labels_30s = DataGenerator_30.getFold(i)
    train_data_5s, train_labels_5s, val_data_5s, val_labels_5s = DataGenerator_5.getFold(i)
    
    # Reshape data
    train_data_30s = train_data_30s.reshape(-1, seg_30 * freq, channels)
    val_data_30s = val_data_30s.reshape(-1, seg_30 * freq, channels)
    train_data_5s = train_data_5s.reshape(-1, seg_5 * freq, channels)
    val_data_5s = val_data_5s.reshape(-1, seg_5 * freq, channels)
    
    print(f"30s - Train: {train_data_30s.shape}, Val: {val_data_30s.shape}")
    print(f"5s - Train: {train_data_5s.shape}, Val: {val_data_5s.shape}")
    
    # Create and train 30s model
    print(f"\nTraining 30s model...")
    model_30s = create_model(train_data_30s.shape[1:], channels, seg_30, freq)
    model_30s.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg['lr']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history_30s = train_model(model_30s, train_data_30s, train_labels_30s, 
                             val_data_30s, val_labels_30s, i, "model_30s", output_path)
    
    # Evaluate 30s model
    pred_probs_30s = model_30s.predict(val_data_30s, batch_size=cfg['bs'], verbose=0)
    pred_labels_30s = np.argmax(pred_probs_30s, axis=1)
    true_labels_30s = np.argmax(val_labels_30s, axis=1)
    
    acc_30s = metrics.accuracy_score(true_labels_30s, pred_labels_30s)
    all_scores_30s.append(acc_30s)
    all_predictions_30s.extend(pred_labels_30s)
    all_true_labels_30s.extend(true_labels_30s)
    
    print(f"30s Model - Fold {i+1} Accuracy: {acc_30s:.4f}")
    PrintScore(true_labels_30s, pred_labels_30s, fold=i+1, 
              savePath=output_path, model_name="model_30s")
    ConfusionMatrix(true_labels_30s, pred_labels_30s, fold=i+1, 
                   savePath=output_path, model_name="model_30s")
    
    # Create and train 5s model
    print(f"\nTraining 5s model...")
    model_5s = create_model(train_data_5s.shape[1:], channels, seg_5, freq)
    model_5s.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg['lr']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history_5s = train_model(model_5s, train_data_5s, train_labels_5s, 
                            val_data_5s, val_labels_5s, i, "model_5s", output_path)
    
    # Evaluate 5s model
    pred_probs_5s = model_5s.predict(val_data_5s, batch_size=cfg['bs'], verbose=0)
    pred_labels_5s = np.argmax(pred_probs_5s, axis=1)
    true_labels_5s = np.argmax(val_labels_5s, axis=1)
    
    acc_5s = metrics.accuracy_score(true_labels_5s, pred_labels_5s)
    all_scores_5s.append(acc_5s)
    all_predictions_5s.extend(pred_labels_5s)
    all_true_labels_5s.extend(true_labels_5s)
    
    print(f"5s Model - Fold {i+1} Accuracy: {acc_5s:.4f}")
    PrintScore(true_labels_5s, pred_labels_5s, fold=i+1, 
              savePath=output_path, model_name="model_5s")
    ConfusionMatrix(true_labels_5s, pred_labels_5s, fold=i+1, 
                   savePath=output_path, model_name="model_5s")
    
    # Plot training history
    plot_training_history(history_30s, fold=i+1, savePath=output_path, model_name="model_30s")
    plot_training_history(history_5s, fold=i+1, savePath=output_path, model_name="model_5s")
    
    # Clean up memory
    del model_30s, model_5s
    del train_data_30s, train_labels_30s, val_data_30s, val_labels_30s
    del train_data_5s, train_labels_5s, val_data_5s, val_labels_5s
    tf.keras.backend.clear_session()
    gc.collect()
    
    print(f"Fold {i+1} completed!")


# Final results
print(f"\n{'='*80}")
print("FINAL CROSS-VALIDATION RESULTS")
print(f"{'='*80}")

print(f"\n30s Model Results:")
print(f"Fold accuracies: {[f'{acc:.4f}' for acc in all_scores_30s]}")
print(f"Mean accuracy: {np.mean(all_scores_30s):.4f} ± {np.std(all_scores_30s):.4f}")

print(f"\n5s Model Results:")
print(f"Fold accuracies: {[f'{acc:.4f}' for acc in all_scores_5s]}")
print(f"Mean accuracy: {np.mean(all_scores_5s):.4f} ± {np.std(all_scores_5s):.4f}")

# Overall evaluation
print(f"\nOverall 30s Model Evaluation:")
calculate_overall_metrics(np.array(all_true_labels_30s), np.array(all_predictions_30s))
PrintScore(np.array(all_true_labels_30s), np.array(all_predictions_30s), 
          savePath=output_path, model_name="model_30s_overall")
ConfusionMatrix(np.array(all_true_labels_30s), np.array(all_predictions_30s), 
               savePath=output_path, model_name="model_30s_overall")

print(f"\nOverall 5s Model Evaluation:")
calculate_overall_metrics(np.array(all_true_labels_5s), np.array(all_predictions_5s))
PrintScore(np.array(all_true_labels_5s), np.array(all_predictions_5s), 
          savePath=output_path, model_name="model_5s_overall")
ConfusionMatrix(np.array(all_true_labels_5s), np.array(all_predictions_5s), 
               savePath=output_path, model_name="model_5s_overall")

# Save final results
final_results = {
    'scores_30s': all_scores_30s,
    'scores_5s': all_scores_5s,
    'mean_30s': np.mean(all_scores_30s),
    'std_30s': np.std(all_scores_30s),
    'mean_5s': np.mean(all_scores_5s),
    'std_5s': np.std(all_scores_5s),
    'predictions_30s': all_predictions_30s,
    'true_labels_30s': all_true_labels_30s,
    'predictions_5s': all_predictions_5s,
    'true_labels_5s': all_true_labels_5s
}

np.savez(os.path.join(output_path, 'final_results.npz'), **final_results)

print(f"\n{'='*80}")
print("TRAINING COMPLETED SUCCESSFULLY!")
print(f"Results saved to: {output_path}")
print(f"{'='*80}")
