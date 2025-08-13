import numpy as np
# sys.path.append('../input')

import os

# mfe256 256 128 111 e30 fn0.001

from Utils import *
from DataGenerator import *


# Path = "./datasets/sleepedf-2013/npz/sleep_edf_processed.npz"
# Path = "./datasets/sleepedf-2018/npz/sleep_edf_processed_15s.npz"
Path = "./datasets/sleepedf-2018/npz/sleep_edf_processed_30s.npz"
# output_path = "./MFE/output_sleepedf2013/test30s53/"
output_path = "./MFE/output_sleepedf2018/test30s53/"


# Path = "./datasets/ISRUC/ISRUC_S3/ISRUC_S3_all.npz"
# Path = "./datasets/ISRUC/ISRUC_S1/ISRUC_S1_5s_ham.npz"
# output_path = "./MFE/output_s3/test30s53/"
# output_path = "./MFE/output_s3/test5s53/"


ReadList = np.load(Path, allow_pickle=True)
Fold_Num   = ReadList['Fold_len']    # Num of samples of each fold
Fold_Data  = ReadList['Fold_data']   # Data of each fold
Fold_Label = ReadList['Fold_label']  # Labels of each fold


freq = 100
channels = 4
subject_num = len(Fold_Num)
fold = 10
seg = 30


cfg = {
    'bs': 32,
    'epochs': 50
}



print("Read data successfully")
print('Number of samples: ',np.sum(Fold_Num))
print('Number of subjects:', len(Fold_Num))
print('Fold_Data length:', len(Fold_Data))
print('Fold_Label length:', len(Fold_Label))
print('Fold configuration: {} folds with {} subjects total'.format(fold, subject_num))


DataGenerator = kFoldGenerator(Fold_Data, Fold_Label, fold, subject_num)
del ReadList, Fold_Label, Fold_Data

# Debug: Check if the DataGenerator was initialized correctly
print('DataGenerator initialized with:')
print('k (number of folds):', DataGenerator.k)
print('n (number of subjects):', DataGenerator.n) 
print('x_list length:', len(DataGenerator.x_list))
print('y_list length:', len(DataGenerator.y_list))

if not os.path.exists(output_path):
    os.makedirs(output_path)


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D, Activation,\
BatchNormalization, Add, Reshape, TimeDistributed, Input, GlobalAveragePooling1D
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

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Exclude GPU:2 and set GPU:1 as the visible device
        tf.config.experimental.set_visible_devices(gpus[3], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[3], True)
        print("Using GPU:0 for computations")
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs available. Using CPU for computations.")

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


def create_model(input_shape, channels=channels, time_second=30, freq=100):# input_shape: (3000, 10)
    inputs_channel = Input(shape=(time_second*freq, 1))
    # inputs_channel = Input(shape=(151, 1)) # 151, 1
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
    # input_re = Reshape((channels, 151, 1))(inputs) # (10, 3000, 1)
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


first_decay_steps = 10
lr_decayed_fn = (
  tf.keras.optimizers.schedules.CosineDecayRestarts(
      0.001,
      first_decay_steps))


tf.config.experimental_run_functions_eagerly(True)

def aggregate_predictions(predictions, seg):
    """
    Aggregate predictions back to 30-second predictions by summing probabilities.
    :param predictions: np.ndarray, shape (total_segments, num_classes)
    :return: aggregated_predictions
    """
    aggregated_predictions = []
    start_idx = 0
    num_segments_per_epoch = 30 // seg  # Number of smaller segments in a 30-second epoch 10
    fold_lengths = predictions.shape[0]  # Total number of segments in the fold


    for _ in range(fold_lengths // num_segments_per_epoch):
        start = start_idx
        end = start_idx + num_segments_per_epoch
        segment_predictions = predictions[start:end, :]  # Shape: (num_segments_per_epoch, num_classes)

        # Sum probabilities for each class
        summed_probabilities = np.sum(segment_predictions, axis=0)  /  num_segments_per_epoch

        aggregated_predictions.append(summed_probabilities)

        start_idx += num_segments_per_epoch

    return np.array(aggregated_predictions)

best_val_acc = []
all_scores = []

for i in range(0, fold):  # 20-fold
    print('Fold #', i)

    # Ensure time_second and freq are passed correctly
    time_second = seg  # Define time_second explicitly

    train_data, train_targets, val_data, val_targets = DataGenerator.getFold(i)  # train_data [7665, 10, 3000] val_data[924, 10, 3000] train_targets[7665, 5] val_targets[924, 5]
    train_data, val_data = train_data.reshape(-1, time_second * freq, channels), val_data.reshape(-1, time_second * freq, channels)  # train_data [7665, 3000, 10] val_data[924, 3000, 10]

    opt = tf.keras.optimizers.Adam(learning_rate=lr_decayed_fn, amsgrad=True)
    ce_model = create_model(input_shape=train_data.shape[1:], freq=freq, channels=channels, time_second=time_second)
    verbose = 1

    ce_model.compile(
        optimizer=opt,
        loss={'Label': "categorical_crossentropy"},  # Ensure keys match model outputs
        metrics={'Label': "accuracy"}
    )

    if not os.path.exists(output_path+str(i)+'ResNet_Best'+'.h5'):

        history = ce_model.fit(
            train_data, train_targets,
            batch_size=cfg['bs'], epochs=cfg['epochs'],
            validation_data=(val_data, val_targets),
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    output_path + str(i) + 'ResNet_Best' + '.h5',
                    monitor='val_accuracy',  # Monitor validation accuracy
                    verbose=1,
                    save_best_only=True,
                    mode='auto')],
            verbose=verbose
        )

        # Save training information
        fit_loss = np.array(history.history['loss'])
        fit_acc = np.array(history.history['accuracy'])
        fit_val_loss = np.array(history.history['val_loss'])
        fit_val_acc = np.array(history.history['val_accuracy'])
        print('Best val acc:', max(history.history['val_accuracy']))
        best_val_acc.append(max(history.history['val_accuracy']))

        saveFile = open(output_path + "Result_MFE.txt", 'a+')
        print('Fold #'+str(i), file=saveFile)
        print(history.history, file=saveFile)
        saveFile.close()    


    # get and save the learned feature
    ce_model.load_weights(output_path+str(i)+'ResNet_Best'+'.h5', skip_mismatch=True, by_name=True)


    # Predict ------------------------------------------------------------
    predicts = ce_model.predict(val_data, batch_size=cfg['bs'])
    print('Predict shape:', predicts.shape)  # (924, 5)

    aggregated_preds = aggregate_predictions(predicts, seg)
    print('Aggregated predictions shape:', aggregated_preds.shape)  # (924,)
    AllPred_temp = np.argmax(aggregated_preds, axis=1)

    aggregated_true = aggregate_predictions(val_targets, seg)
    AllTrue_temp = np.argmax(aggregated_true, axis=1)




    # Predict ------------------------------------------------------------
    acc = metrics.accuracy_score(AllTrue_temp, AllPred_temp)
    print("Predict accuracy:", acc)
    all_scores.append(acc)

    if i == 0:
        AllPred = AllPred_temp
        AllTrue = AllTrue_temp
    else:
        AllPred = np.concatenate((AllPred, AllPred_temp))
        AllTrue = np.concatenate((AllTrue, AllTrue_temp))

    # VariationCurve(fit_acc, fit_val_acc, f'Acc_{i}', output_path, figsize=(9, 6))
    # VariationCurve(fit_loss, fit_val_loss, f'Loss_{i}', output_path, figsize=(9, 6))

    # Print score to console
    print(128*'=')
    PrintScore(AllTrue_temp, AllPred_temp, fold=i, savePath=output_path)
    ConfusionMatrix(AllTrue_temp, AllPred_temp, fold=i, classes=['W', 'N1', 'N2', 'N3', 'REM'], savePath=output_path)

    # Fold finish
    keras.backend.clear_session()
    del train_data, train_targets, val_data, val_targets
    gc.collect()
    print('Fold #', i, 'finished')



print(128 * '_')
print('End of training MFE.')
print(128 * '#')

# print acc of each fold
if len(best_val_acc) != 0:
    print(128*'=')
    print("best val acc: ",best_val_acc)
    print("Average best val acc of each fold: ",np.mean(best_val_acc))


print(128*'=')
print("All folds' acc: ",all_scores)
print("Average acc of each fold: ",np.mean(all_scores))

# Print score to console
print(128*'=')
PrintScore(AllTrue, AllPred, savePath=output_path)

# Print confusion matrix and save
ConfusionMatrix(AllTrue, AllPred, classes=['W','N1','N2','N3','REM'], savePath=output_path)

print('End of evaluating MFE.')
print('###train without contrastive learning###')
print(128 * '#')



