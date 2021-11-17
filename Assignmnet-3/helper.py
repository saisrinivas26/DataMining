from scipy.stats import kurtosis
from scipy.signal import find_peaks
from numpy.fft import fft
import numpy as np

pca_comps = 9

# preprocessing
def normalize(col):
    mean_col = col.mean()
    max_col = col.max()
    min_col = col.min()
    col = (col - mean_col) / (max_col - min_col)
    return col


def normalize_np(rY):
    return (rY - np.mean(rY))/(np.max(rY-np.mean(rY))-np.min(rY-np.mean(rY)))

def extract_differences(data):
    slopes = []
    # feature 2, slope of cgms
    for i in range(1, len(data)-1):
        numerator = (data[i-1] + data[i+1] - 2 * data[i])
        denominator = 10 * 60
        slope = numerator / denominator
        slopes.append(slope)

    peaks, _ = find_peaks(slopes)
    troughs, _ = find_peaks([-x for x in slopes])
    differences = []
    zero_crossings = [i for i in range(len(slopes)) if slopes[i] == 0]
    for i, t in enumerate(troughs):
        diff1 = 0
        diff2 = 0
        if i < len(peaks)-1:
            diff1 =abs(slopes[peaks[i+1]] - slopes[t])
        if i < len(peaks):
            diff2 =abs(slopes[peaks[i]] - slopes[t])
        if not diff1 and not diff2 :
            continue

        if diff1 >= diff2:
            differences.append(diff1)
        else:
            differences.append(diff2)
    differences.sort(key=lambda x: -x)
    zero_crossings.sort()
    return differences, zero_crossings


def extract_features(data):
    if len(data) not in set([24, 30]):
        return None

    data = [x if x is not None else 0 for x in data]

    max_cgm = max(data)
    min_cgm = min(data)

    # feature 1
    feature1 = max_cgm - min_cgm

    # features 2 - 7
    differences, zero_crossings = extract_differences(data)

    # feature 8 - time between cgm_max and cgm_min
    time_cgm_max = np.argmax(data)
    time_cgm_min = np.argmin(data)

    time_delta = (time_cgm_max - time_cgm_min) * 5
    feature8 = abs(time_delta)

    # feature9-11 get FFT omegas
    yf = 2 * np.abs(fft(data)).round(2)
    features9_11 = list(np.sort(yf)[-4:-1])

    feature_dict = {
        'max_difference_cgm':[feature1],
        'max_cgm_time_difference':[feature8],
        'mean_cgm':[np.mean(data, dtype=np.float64)],
        'std_cgm':[np.std(data, dtype=np.float64)],
        'kurtosis': [kurtosis(data)]
    }


    
    for i in range(3):
        velocity = differences[i] if i < len(differences) else 0
        feature_dict['velocity_{}'.format(i+1)] = [velocity]
        zero_crossing = zero_crossings[i] if i < len(zero_crossings) else 0
        feature_dict['zero_crossing_{}'.format(i+1)] = [zero_crossing]

    for i in range(3):
        fft_1 = features9_11[i] if i < len(features9_11) else 0
        feature_dict['fft_{}'.format(i+1)] = [fft_1]

    return feature_dict