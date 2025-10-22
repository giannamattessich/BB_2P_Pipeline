from scipy.signal import butter, filtfilt, firwin

"""
Functions to filter analog signals
"""

def butter_lowpass(cutoff, fs, order=4):
    """
    Apply generic butter filter

    Args:
        cutff (int): cutoff frequency for lowpass filter
        fs (int; default is intan sample rate 20e3): sampling rate of signal
    Returns: 
        b, a (ndarray, ndarray): numerator (b) and denominator (a) polynomials of the IIR filter.
    """
    # calculate nyquist with cutoff
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff=500, fs=20e3, order=3):
    """
    Apply both butter and filtfilt filters

    Args:
        data (array-like): 
        cutoff (int): cutoff frequency for lowpass filter
        fs (int; default is intan sample rate 20e3): sampling rate of signal
        order (int; default is 3): steepness of signal above cutoff frequency
    """
    b, a = butter_lowpass(cutoff=cutoff, fs=fs, order=order)
    y = filtfilt(b, a, data)
    return y


def sinc_lowpass_filter(signal, cutoff, fs, numtaps=101):
    """
    Apply a sinc lowpass filter to signal
    
    Args:
        signal (np.ndarray): signal to lowpass filter
        cutoff (int): cutoff frequency for filter
        fs (int or float): sampling rate of signal
        numtaps (int): length of filter/num coefficients
    """
    # calc nyquist sampling freq to avoid aliasing 
    nyquist_freq = fs / 2
    #normalize cutoff from 0 to 1, use to get coefficients for finite impulse response
    # apply forward and backward linear filters to sharpen signal and cancel phase shifts
    # convolves input signal with coefficients from FIR
    fir = firwin(numtaps, cutoff / nyquist_freq)
    return filtfilt(fir, [1.0], signal, axis=-1)

# '''Apply a sinc lowpass filter to signal'''
# def sinc_lowpass_filter(signal, cutoff, fs, numtaps=101, gpu_available=False):
#     # calc nyquist sampling freq to avoid aliasing 
#     nyquist_freq = fs / 2
#     # convert to cupy array for faster processing if gpu available else use numpy
#     if GPU_AVAILABLE and cp is not None:
#         signal = cp.asarray(signal)
#         #normalize cutoff from 0 to 1, use to get coefficients for finite impulse response
#         fir = firwin_gpu(numtaps, cutoff / nyquist_freq)
#         # apply forward and backward linear filters to sharpen signal and cancel phase shifts
#         # convolves input signal with coefficients from FIR
#         return filtfilt_gpu(fir, cp.asarray([1.0]), signal, axis=-1)
#     else:
#         fir = firwin(numtaps, cutoff / nyquist_freq)
#         return filtfilt(fir, [1.0], signal, axis=-1)