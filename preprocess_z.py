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


def read_psg(path_Extracted, sub_id, channels, resample=None, expected_duration=30, fs=100):
    """
    读取PSG数据，确保时间长度一致性
    :param resample: 如果为None，则根据expected_duration和fs计算
    :param expected_duration: 期望的每个epoch时长（秒）
    :param fs: 采样率
    """
    if resample is None:
        resample = expected_duration * fs  # 确保采样点数与时间长度匹配
        
    psg = scio.loadmat(path.join(path_Extracted, 'subject%d.mat' % (sub_id)))
    psg_use = []
    for c in channels:
        # 检查原始数据长度
        original_data = psg[c]
        print(f"Channel {c}: original shape {original_data.shape}, resampling to {resample}")
        
        psg_use.append(
            np.expand_dims(signal.resample(original_data, resample, axis=-1), 1))
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
    
    # 返回标签时提供更多信息
    full_label = np.array(label)
    truncated_label = full_label[:-ignore] if ignore > 0 else full_label
    print(f"Label info: total epochs={len(full_label)}, after ignore={len(truncated_label)}, ignored={ignore}")
    return truncated_label


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
# eeg_lowcut = 0.3
# eeg_highcut = 45

fold_label = []
fold_psg = []
fold_len = []

# for sub in range(1, 101):
for sub in range(8,9):
    print('Read subject', sub)
    label = read_label(path_RawData, sub)
    psg = read_psg(path_Extracted, sub, channels, expected_duration=30, fs=fs)
    print('Subject', sub, ':', label.shape, psg.shape)
    
    # 检查数据和标签长度是否匹配
    if len(label) != len(psg):
        print(f"WARNING: Subject {sub} - Length mismatch! Label: {len(label)}, PSG: {len(psg)}")
        # 取较小的长度以确保匹配
        min_len = min(len(label), len(psg))
        label = label[:min_len]
        psg = psg[:min_len]
        print(f"Truncated to common length: {min_len}")
    
    assert len(label) == len(psg), f"Subject {sub}: Label and PSG length mismatch after truncation"

    # in ISRUC, 0-Wake, 1-N1, 2-N2, 3-N3, 5-REM
    label[label==5] = 4  # make 4 correspond to REM
    
    # Preprocess EEG, EOG, and EMG channels
    eeg_channels = ['C3_A2', 'C4_A1', 'F3_A2', 'F4_A1', 'O1_A2', 'O2_A1']
    eog_channels = ['LOC_A2', 'ROC_A1']
    emg_channels = ['X1', 'X2']
    
    eeg_data = psg[:, [channels.index(c) for c in eeg_channels]]
    eog_data = psg[:, [channels.index(c) for c in eog_channels]]
    emg_data = psg[:, [channels.index(c) for c in emg_channels]]
    
    # eeg_data = butter_bandpass_filter(eeg_data, eeg_lowcut, eeg_highcut, fs)

    # Combine preprocessed channels
    psg_preprocessed = np.concatenate([eeg_data, eog_data, emg_data], axis=1)
    
    # 验证数据维度
    expected_samples = 30 * fs  # 30秒 * 采样率
    num_segments = len(label)
    
    print(f"Before reshaping: {psg_preprocessed.shape}")
    print(f"Expected: ({num_segments}, {len(channels)}, {expected_samples})")
    
    # 检查数据是否已经是正确的形状
    if psg_preprocessed.shape == (num_segments, len(channels), expected_samples):
        print("✓ Data is already in correct shape, no reshaping needed")
        # 数据已经是正确的形状，不需要重塑
    else:
        # 需要重塑数据
        print(f"Reshaping data from {psg_preprocessed.shape} to ({num_segments}, {len(channels)}, -1)")
        
        # 验证总样本数是否匹配
        total_samples_current = psg_preprocessed.shape[0] * psg_preprocessed.shape[1] * psg_preprocessed.shape[2]
        total_samples_expected = num_segments * len(channels) * expected_samples
        
        if total_samples_current != total_samples_expected:
            print(f"WARNING: Total sample count mismatch!")
            print(f"Current total samples: {total_samples_current}")
            print(f"Expected total samples: {total_samples_expected}")
            print(f"Difference: {total_samples_current - total_samples_expected}")
        
        psg_preprocessed = psg_preprocessed.reshape(num_segments, len(channels), -1)
    
    # 最终验证
    print(f"Final data shape: {psg_preprocessed.shape}")
    if psg_preprocessed.shape[2] != expected_samples:
        print(f"ERROR: Samples per segment mismatch: {psg_preprocessed.shape[2]} vs {expected_samples}")
        print(f"This suggests a fundamental data structure issue!")
    else:
        print(f"✓ Data structure validated: {num_segments} epochs, {len(channels)} channels, {expected_samples} samples per epoch")

    split_data = psg_preprocessed
    split_labels = np.eye(5)[label]

    fold_psg.append(split_data)
    fold_label.append(split_labels)
    fold_len.append(len(split_labels))

print('Preprocess over.')

np.savez(path.join(path_output, 'ISRUC_S3_8_raw.npz'),
    Fold_data = np.array(fold_psg, dtype=object),
    Fold_label = np.array(fold_label, dtype=object),
    Fold_len = np.array(fold_len, dtype=object)
)
print('Saved to', path.join(path_output, 'ISRUC_S3_8_raw.npz'))