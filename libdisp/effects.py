import numpy as np
from .spectrum import stft, power_to_db
from .features import *
from .utils import valid_audio, to_mono


def trim(y, top_db=60, ref=np.max, frame_length=2048, hop_length=512):

    non_silent = _signal_to_frame_nonsilent(y,
                                            frame_length=frame_length,
                                            hop_length=hop_length,
                                            ref=ref,
                                            top_db=top_db)

    nonzero = np.flatnonzero(non_silent)

    if nonzero.size > 0:
        # Compute the start and end positions
        # End position goes one frame past the last non-zero
        start = int(frames_to_samples(nonzero[0], hop_length))
        end = min(y.shape[-1],
                  int(frames_to_samples(nonzero[-1] + 1, hop_length)))
    else:
        # The signal only contains zeros
        start, end = 0, 0

    # Build the mono/stereo index
    full_index = [slice(None)] * y.ndim
    full_index[-1] = slice(start, end)

    return y[tuple(full_index)], np.asarray([start, end])


def _signal_to_frame_nonsilent(y, frame_length=2048, hop_length=512, top_db=60,
                               ref=np.max):
    # Convert to mono
    y_mono = to_mono(y)

    # Compute the MSE for the signal
    mse = rms(y=y_mono,
                      frame_length=frame_length,
                      hop_length=hop_length)**2

    return (power_to_db(mse.squeeze(),
                             ref=ref,
                             top_db=None) > - top_db)


