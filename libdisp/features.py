import numpy as np
from .utils import *

def rms(y=None, S=None, frame_length=2048, hop_length=512,
        center=True, pad_mode='reflect'):
    if y is not None and S is not None:
        raise ValueError('Either `y` or `S` should be input.')
    if y is not None:
        y = to_mono(y)
        if center:
            y = np.pad(y, int(frame_length // 2), mode=pad_mode)

        x = frame(y,
                       frame_length=frame_length,
                       hop_length=hop_length)

        # No normalization is necessary for time-domain input
        norm = 1
    elif S is not None:
        x, _ = spectrogram(y=y, S=S,
                            n_fft=frame_length,
                            hop_length=hop_length)

        # FFT introduces a scaling of n_fft to energy calculations
        norm = 2 * (x.shape[0] - 1)
    else:
        raise ValueError('Either `y` or `S` must be input.')

    return np.sqrt(np.mean(np.abs(x)**2, axis=0, keepdims=True) / norm)

def spectrogram(y=None, S=None, n_fft=2048, hop_length=512, power=1,
                 win_length=None, window='hann', center=True, pad_mode='reflect'):

    if S is not None:
        # Infer n_fft from spectrogram shape
        n_fft = 2 * (S.shape[0] - 1)
    else:
        # Otherwise, compute a magnitude spectrogram from input
        S = np.abs(stft(y, n_fft=n_fft, hop_length=hop_length,
                        win_length=win_length, center=center,
                        window=window, pad_mode=pad_mode))**power

    return S, n_fft

def frames_to_samples(frames, hop_length=512, n_fft=None):
    offset = 0
    if n_fft is not None:
        offset = int(n_fft // 2)

    return (np.asanyarray(frames) * hop_length + offset).astype(int)



