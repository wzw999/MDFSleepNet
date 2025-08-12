import numpy as np
# sys.path.append('../input')


import os


from Utils import *
from DataGenerator import *


# Path_30 = "./datasets/ISRUC/ISRUC_S3/ISRUC_S3_all.npz"
# Path_1 = "./datasets/ISRUC/ISRUC_S3/ISRUC_S3_5s_ham.npz"
# output_path = "./MFE/output_s3/test30_5_c/"
# output_path_30s = "./MFE/output_s3/testwithoutdomain/"
# output_path_1s = "./MFE/output_s3/test5sham/"
# Path_30 = "./datasets/ISRUC/ISRUC_S1/ISRUC_S1_all.npz"
# Path_1 = "./datasets/ISRUC/ISRUC_S1/ISRUC_S1_5s_ham.npz"
# output_path = "./MFE/output_s1/test30_5_c/"
# output_path_30s = "./MFE/output_s1/testbz32wodm/"
# output_path_1s = "./MFE/output_s1/test5sham/"


# Path_30 = "./datasets/sleepedf-2013/npz/sleep_edf_processed.npz"
# Path_1 = "./datasets/sleepedf-2013/npz/sleep_edf_processed_5s.npz"
# output_path = "./MFE/output_sleepedf2013/test30_5/"
# output_path_30s = "./MFE/output_sleepedf2013/test30s/"
# output_path_1s = "./MFE/output_sleepedf2013/test5s/"

Path_30 = "./datasets/sleepedf-2018/npz/sleep_edf_processed_30s.npz"
Path_1 = "./datasets/sleepedf-2018/npz/sleep_edf_processed_5s.npz"
output_path = "./MFE/output_sleepedf2018/test30_5/"
output_path_30s = "./MFE/output_sleepedf2018/test30s/"
output_path_1s = "./MFE/output_sleepedf2018/test5s/"


ReadList_1 = np.load(Path_1, allow_pickle=True)
Fold_Num_1   = ReadList_1['Fold_len']    # Num of samples of each fold
Fold_Data_1  = ReadList_1['Fold_data']   # Data of each fold
Fold_Label_1 = ReadList_1['Fold_label']  # Labels of each fold
ReadList_30 = np.load(Path_30, allow_pickle=True)
Fold_Num_30   = ReadList_30['Fold_len']    # Num of samples of each fold
Fold_Data_30  = ReadList_30['Fold_data']   # Data of each fold
Fold_Label_30 = ReadList_30['Fold_label']  # Labels of each fold


freq = 100
channels = 4
subject_num = len(Fold_Num_30)
fold = 10
seg_30 = 30
seg_3 = 3
seg_1 = 5
seg_5 = 5
seg_10 = 10
seg_15 = 15


cfg = {
    'bs': 32,  # Batch size for fusion model
    'epochs': 20,  # Epochs for fusion model
    'bs_1s': 32,  # Batch size for 1s model
    'bs_30s': 32,  # Batch size for 30s model
    'epochs_1s': 10,  # Epochs for 1s model
    'epochs_30s': 10,  # Epochs for 30s model
}


DataGenerator_30 = kFoldGenerator(Fold_Data_30, Fold_Label_30, fold, subject_num)

DataGenerator_1 = kFoldGenerator(Fold_Data_1, Fold_Label_1, fold, subject_num)  


del ReadList_30, ReadList_1, Fold_Data_30, Fold_Data_1, Fold_Label_30, Fold_Label_1


if not os.path.exists(output_path):
    os.makedirs(output_path)



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D, Activation,\
BatchNormalization, Add, Reshape, TimeDistributed, Input, GlobalAveragePooling1D, GRU, Concatenate, Softmax, Permute, Lambda, LayerNormalization, Layer, RepeatVector, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


from sklearn import metrics

import random as python_random
os.environ['TF_DETERMINISTIC_OPS'] = '1'
np.random.seed(32)
python_random.seed(32)
tf.random.set_seed(32)
print("keras version:", keras.__version__)
 
# print(device_lib.list_local_devices())

tf.config.set_soft_device_placement(True)

# 获取所有可用的 GPU 设备
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Exclude GPU:2 and set GPU:1 as the visible device
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[1], True)
        print("Using GPU:1 for computations")
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs available. Using CPU for computations.")

# 验证当前使用的设备
print("Current device:", tf.config.get_visible_devices('GPU'))




def CNN(inputs, fs=32, kernel_size=5, pool_size=3, weight=0.001):
    x = Conv1D(fs, kernel_size,1, padding='same', kernel_regularizer=l2(weight))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size, 2, padding='same')(x)
    return x


def ResNet(inputs, fs=32, ks_1=5, ps_1=3, ks_2=5, ps_2=3, weight=0.001):
    x = CNN(inputs, fs, ks_1, ps_1,weight)
    x = CNN(x, fs, ks_2, ps_2,weight)
    shortcut_x = Conv1D(fs,1,2,padding='same')(inputs)
    shortcut_x = Conv1D(fs,1,2,padding='same')(shortcut_x)
    return Add()([x, shortcut_x])




def create_model(input_shape, channels, time_second, freq=100):# input_shape: (3000, 10)
    inputs_channel = Input(shape=(time_second*freq, 1))
    x = ResNet(inputs_channel, 16)
    x = Dropout(0.2)(x)
    x = ResNet(x, 32)
    x = Dropout(0.2)(x)
    x = ResNet(x, 64)
    x = Dropout(0.2)(x)
    x = ResNet(x, 128)
    x = Dropout(0.2)(x)
    
    outputs = GlobalAveragePooling1D()(x)
    
    fea_part = Model(inputs=inputs_channel, outputs=outputs) # (, 3000, 1) -> (, 128)
    # fea_part.summary()
    
    # extract the features from each channel
    inputs = Input(shape=input_shape) # (3000, 10)
    input_re = Reshape((channels, time_second * freq, 1))(inputs) # (10, 3000, 1)
#     fea_all = tf.stack([fea_part(input_re[:,i,:,:]) for i in range(channels)], axis=1)
    fea_all = TimeDistributed(fea_part)(input_re) 
#     fea_all = tf.keras.layers.Attention(use_scale=True)([fea_all, fea_all])
    
    fla_fea = Flatten()(fea_all)
    fla_fea = Dropout(0.5)(fla_fea)

#     merged = GlobalAveragePooling1D()(fea_all)
    merged = Dense(128)(fla_fea)
    label_out = Dense(5, activation='softmax', name='Label')(merged)

    ce_model = Model(inputs, label_out)  
    
    # return cl_model, ce_model, pre_model
    return ce_model



import gc
# k-fold cross validation
all_scores = []
all_scores_3s, all_scores_30s, all_scores_1s = [], [], []
all_scores_5s = []

first_decay_steps = 10
lr_decayed_fn = (
  tf.keras.optimizers.schedules.CosineDecayRestarts(
      0.001,
      first_decay_steps))


tf.config.experimental_run_functions_eagerly(True)




# Function to aggregate predictions back to 30-second predictions by summing probabilities
def aggregate_predictions(predictions, seg):
    """
    Aggregate predictions back to 30-second predictions by summing probabilities.
    :param predictions: np.ndarray, shape (total_segments, num_classes)
    :return: aggregated_predictions
    """
    aggregated_predictions = []
    start_idx = 0
    num_segments_per_epoch = 30 // seg # Number of smaller segments in a 30-second epoch 10
    fold_lengths = predictions.shape[0]  # Total number of segments in the fold


    for _ in range(fold_lengths // num_segments_per_epoch):
        # Extract predictions for the current 30-second epoch


        start = start_idx
        end = start_idx + num_segments_per_epoch
        segment_predictions = predictions[start:end, :]  # Shape: (num_segments_per_epoch, num_classes)

        # Sum probabilities for each class
        summed_probabilities = np.sum(segment_predictions, axis=0)  /  num_segments_per_epoch

        # Find the label with the highest summed probability
        # most_probable_label = np.argmax(summed_probabilities)
        # aggregated_predictions.append(most_probable_label)
        aggregated_predictions.append(summed_probabilities)

        start_idx += num_segments_per_epoch

    return np.array(aggregated_predictions)



  

all_scores_fusion = []

for i in range(fold):  # 20-fold
    print('Fold #', i)

    # Train 1s model
    train_data_1s, train_targets_1s, val_data_1s, val_targets_1s = DataGenerator_1.getFold(i)
    train_data_1s, val_data_1s = train_data_1s.reshape(-1, seg_1 * freq, channels), val_data_1s.reshape(-1, seg_1 * freq, channels)

    opt_1s = tf.keras.optimizers.Adam(learning_rate=lr_decayed_fn, amsgrad=True)
    ce_model_1s = create_model(input_shape=train_data_1s.shape[1:], freq=freq, channels=channels, time_second=seg_1)

    ce_model_1s.compile(
        optimizer=opt_1s,
        loss={'Label': "categorical_crossentropy"},
        metrics={'Label': "accuracy"},
    )



    # Train 30s model
    train_data_30s, train_targets_30s, val_data_30s, val_targets_30s = DataGenerator_30.getFold(i)
    train_data_30s, val_data_30s = train_data_30s.reshape(-1, seg_30 * freq, channels), val_data_30s.reshape(-1, seg_30 * freq, channels)

    opt_30s = tf.keras.optimizers.Adam(learning_rate=lr_decayed_fn, amsgrad=True)
    ce_model_30s = create_model(input_shape=train_data_30s.shape[1:], freq=freq, channels=channels, time_second=seg_30)

    ce_model_30s.compile(
        optimizer=opt_30s,
        loss={'Label': "categorical_crossentropy"},
        metrics={'Label': "accuracy"},
    )



    # Load weights for evaluation
    ce_model_1s.load_weights(output_path_1s + str(i) + 'ResNet_Best' + '.h5')
    ce_model_30s.load_weights(output_path_30s + str(i) + 'ResNet_Best' + '.h5')
    # Predict with 1s model
    predicts_1s = ce_model_1s.predict(val_data_1s, batch_size=cfg['bs_1s'], verbose=1)
    predicts_30s = ce_model_30s.predict(val_data_30s, batch_size=cfg['bs_30s'], verbose=1)
   
    aggregate_1s = aggregate_predictions(predicts_1s, seg_1)
    aggregate_30s = aggregate_predictions(predicts_30s, seg_30)
   
    # Combine predictions by summing probabilities
    combined_aggregates = np.sum([aggregate_1s, aggregate_30s], axis=0)
    print(f"Fold {i} - Combined Aggregates Shape: {combined_aggregates.shape}")


    AllTrue_temp = np.argmax(val_targets_30s, axis=1)

   
    
    AllPred_fusion = np.argmax(combined_aggregates, axis=1)
    AllTrue_fusion = np.argmax(val_targets_30s, axis=1)
    acc_fusion = metrics.accuracy_score(AllTrue_fusion, AllPred_fusion)
    all_scores_fusion.append(acc_fusion)
    PrintScore(AllTrue_fusion, AllPred_fusion, fold=i, savePath=output_path, model_name="fusion_model")
    print(f"Fold {i} - Fusion Model: Accuracy = {metrics.accuracy_score(AllTrue_fusion, AllPred_fusion)}")
    ConfusionMatrix(AllTrue_fusion, AllPred_fusion, fold=i, classes=['W', 'N1', 'N2', 'N3', 'REM'], savePath=output_path, model_name="fusion_model")
    
    # Collect results for overall evaluation
    if i == 0:
        AllPred = AllPred_fusion
        AllTrue = AllTrue_fusion
    else:
        AllPred = np.concatenate((AllPred, AllPred_fusion))
        AllTrue = np.concatenate((AllTrue, AllTrue_fusion))
    
    # 5. 清理内存
    del ce_model_1s, ce_model_30s
    del train_data_1s, train_targets_1s, val_data_1s, val_targets_1s
    del train_data_30s, train_targets_30s, val_data_30s, val_targets_30s
    tf.keras.backend.clear_session()
    gc.collect()


print(128 * '_')
print('End of training MFE.')
print(128 * '#')

# print acc of each fold
print(128*'=')
print("Combined model acc:", all_scores_fusion)
print("Average combined acc:", np.mean(all_scores_fusion))

# Print confusion matrix and save

PrintScore(AllTrue, AllPred, savePath=output_path)
ConfusionMatrix(AllTrue, AllPred, classes=['W','N1','N2','N3','REM'], savePath=output_path)

print('End of evaluating MFE.')
print('###train without contrastive learning###')
print(128 * '#')



