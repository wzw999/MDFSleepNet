import numpy as np
import scipy.io as scio
from os import path
from scipy import signal
from scipy.signal import butter
from scipy.signal.windows import hamming  # Corrected import
from sklearn.preprocessing import StandardScaler

# path_Extracted = './datasets/ISRUC/ISRUC_S1/ExtractedChannels/'
# path_RawData   = './datasets/ISRUC/ISRUC_S1/RawData/'
# path_output    = './datasets/ISRUC/ISRUC_S1/'
path_Extracted = './datasets/ISRUC/ISRUC_S3/ExtractedChannels/'
path_RawData   = './datasets/ISRUC/ISRUC_S3/RawData/'
path_output    = './datasets/ISRUC/ISRUC_S3/'
channels = ['C3_A2', 'C4_A1', 'F3_A2', 'F4_A1', 'O1_A2', 'O2_A1',
            'LOC_A2', 'ROC_A1','X1', 'X2']


def read_psg(path_Extracted, sub_id, channels, resample=3000):
    psg = scio.loadmat(path.join(path_Extracted, 'subject%d.mat' % (sub_id)))
    psg_use = []
    for c in channels:
        psg_use.append(
            np.expand_dims(signal.resample(psg[c], resample, axis=-1), 1))
    psg_use = np.concatenate(psg_use, axis=1)
    return psg_use

def read_label(path_RawData, sub_id, ignore=30):
    label = []
    with open(path.join(path_RawData, '%d/%d_1.txt' % (sub_id, sub_id))) as f:#s3
    # with open(path.join(path_RawData, '%d/%d/%d_1.txt' % (sub_id,sub_id, sub_id))) as f:#s1
        s = f.readline()
        while True:
            a = s.replace('\n', '')
            label.append(int(a))
            s = f.readline()
            if s == '' or s == '\n':
                break
    return np.array(label[:-ignore])

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def notch_filter(data, freq, fs, Q=30):
    b, a = signal.iirnotch(freq, Q, fs=fs)  # Use absolute frequency
    y = signal.lfilter(b, a, data)
    return y

def split_segments(data, labels, segment_length, target_length, fs):
    """
    Split data and labels into smaller segments, and apply Hamming window to each segment.
    :param data: np.ndarray, shape (num_segments, channels, samples_per_segment)
    :param labels: np.ndarray, shape (num_segments, num_classes)
    :param segment_length: int, original segment length in seconds
    :param target_length: int, target segment length in seconds
    :param fs: int, sampling frequency
    :return: split_data, split_labels
    """
    samples_per_segment = segment_length * fs
    samples_per_target = target_length * fs
    num_targets = samples_per_segment // samples_per_target

    split_data = []
    split_labels = []
    
    # 创建汉明窗
    hamming_window = hamming(samples_per_target)

    for i in range(data.shape[0]):
        for j in range(num_targets):
            start = j * samples_per_target
            end = start + samples_per_target
            segment = data[i, :, start:end]
            
            # 对每个通道应用汉明窗
            for ch in range(segment.shape[0]):
                segment[ch, :] = segment[ch, :] * hamming_window
                
            split_data.append(segment)
            split_labels.append(labels[i])

    return np.array(split_data), np.array(split_labels)

fs = 100  
eeg_lowcut = 0.3
eeg_highcut = 45

fold_label = []
fold_psg = []
fold_len = []

# for sub in range(1, 101):
for sub in range(8, 9):
    print('Read subject', sub)
    label = read_label(path_RawData, sub)
    psg = read_psg(path_Extracted, sub, channels)
    print('Subject', sub, ':', label.shape, psg.shape)
    assert len(label) == len(psg)

    # in ISRUC, 0-Wake, 1-N1, 2-N2, 3-N3, 5-REM
    label[label==5] = 4  # make 4 correspond to REM
    # fold_label.append(np.eye(5)[label])
    
    # Preprocess EEG, EOG, and EMG channels
    eeg_channels = ['C3_A2', 'C4_A1', 'F3_A2', 'F4_A1', 'O1_A2', 'O2_A1']
    eog_channels = ['LOC_A2', 'ROC_A1']
    emg_channels = ['X1', 'X2']
    
    eeg_data = psg[:, [channels.index(c) for c in eeg_channels]]
    eog_data = psg[:, [channels.index(c) for c in eog_channels]]
    emg_data = psg[:, [channels.index(c) for c in emg_channels]]
    
    eeg_data = butter_bandpass_filter(eeg_data, eeg_lowcut, eeg_highcut, fs)

    # Combine preprocessed channels
    psg_preprocessed = np.concatenate([eeg_data, eog_data, emg_data], axis=1)
    
    # Reshape data to (num_segments, channels, samples_per_segment)
    num_segments = len(label)
    psg_preprocessed = psg_preprocessed.reshape(num_segments, len(channels), -1)

    # Split into 1-second segments
    segment_length = 30  # Original segment length in seconds
    target_length = 30    # Target segment length in seconds
    # split_data, split_labels = split_segments(psg_preprocessed, np.eye(5)[label], segment_length, target_length, fs)

    if segment_length != target_length:
        split_data, split_labels = split_segments(psg_preprocessed, np.eye(5)[label], segment_length, target_length, fs)
    else:
        split_data = psg_preprocessed
        split_labels = np.eye(5)[label]

    fold_psg.append(split_data)
    fold_label.append(split_labels)
    fold_len.append(len(split_labels))

print('Preprocess over.')

np.savez(path.join(path_output, 'ISRUC_S3_8.npz'),
    Fold_data = np.array(fold_psg, dtype=object),
    Fold_label = np.array(fold_label, dtype=object),
    Fold_len = np.array(fold_len, dtype=object)
)
print('Saved to', path.join(path_output, 'ISRUC_S3_8.npz'))