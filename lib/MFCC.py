POWER_SPECTRUM_FLOOR = 1e-100

from numpy import *
import numpy.linalg as linalg


def hamming(n):
    """ Generate a hamming window of n points as a numpy array.  """
    return 0.54 - 0.46 * cos(2 * pi / n * (arange(n) + 0.5))

class MFCCExtractor(object):

    def __init__(self, fs, win_length_ms, win_shift_ms, FFT_SIZE, n_bands, n_coefs,
                 PRE_EMPH, verbose = False):
        self.PRE_EMPH = PRE_EMPH
        self.fs = fs
        self.n_bands = n_bands
        self.coefs = n_coefs
        self.FFT_SIZE = FFT_SIZE

        self.FRAME_LEN = int(float(win_length_ms) / 1000 * fs)
        self.FRAME_SHIFT = int(float(win_shift_ms) / 1000 * fs)

        self.window = hamming(self.FRAME_LEN)


        self.M, self.CF = self._mel_filterbank()

        dctmtx = MFCCExtractor.dctmtx(self.n_bands)
        self.D = dctmtx[1: self.coefs + 1]
        self.invD = linalg.inv(dctmtx)[:, 1: self.coefs + 1]

        self.verbose = verbose


    def dprint(self, msg):
        """ Debug print """
        if self.verbose:
            print(msg)

    def extract(self, signal):

        frames = (len(signal) - self.FRAME_LEN) / self.FRAME_SHIFT + 1
        feature = []
        for f in xrange(frames):
            # Windowing
            frame = signal[f * self.FRAME_SHIFT : f * self.FRAME_SHIFT +
                           self.FRAME_LEN] * self.window
            # Pre-emphasis
            frame[1:] -= frame[:-1] * self.PRE_EMPH
            # Power spectrum
            X = abs(fft.fft(frame, self.FFT_SIZE)[:self.FFT_SIZE / 2 + 1]) ** 2
            X[X < POWER_SPECTRUM_FLOOR] = POWER_SPECTRUM_FLOOR  # Avoid zero
            # Mel filtering, logarithm, DCT
            X = dot(self.D, log(dot(self.M, X)))
            feature.append(X)
        feature = row_stack(feature)
        # Show the MFCC spectrum before normalization
        # Mean & variance normalization
        if feature.shape[0] > 1:
            mu = mean(feature, axis=0)
            sigma = std(feature, axis=0)
            feature = (feature - mu) / sigma

        return feature

    def _mel_filterbank(self):
        
        f0 = 700.0 / self.fs
        fn2 = int(floor(self.FFT_SIZE / 2))
        lr = log(1 + 0.5 / f0) / (self.n_bands + 1)
        CF = self.fs * f0 * (exp(arange(1, self.n_bands + 1) * lr) - 1)
        bl = self.FFT_SIZE * f0 * (exp(array([0, 1, self.n_bands, self.n_bands + 1]) * lr) - 1)
        b1 = int(floor(bl[0])) + 1
        b2 = int(ceil(bl[1]))
        b3 = int(floor(bl[2]))
        b4 = min(fn2, int(ceil(bl[3]))) - 1
        pf = log(1 + arange(b1, b4 + 1) / f0 / self.FFT_SIZE) / lr
        fp = floor(pf)
        pm = pf - fp
        M = zeros((self.n_bands, 1 + fn2))
        for c in xrange(b2 - 1, b4):
            r = int(fp[c] - 1)
            M[r, c+1] += 2 * (1 - pm[c])
        for c in xrange(b3):
            r = int(fp[c])
            M[r, c+1] += 2 * pm[c]
        return M, CF
    @staticmethod
    def dctmtx(n):
        """ Return the DCT-II matrix of order n as a numpy array.  """
        x, y = meshgrid(range(n), range(n))
        D = sqrt(2.0 / n) * cos(pi * (2 * x + 1) * y / (2 * n))
        D[0] /= sqrt(2)
        return D

def get_mfcc_extractor(fs, win_length_ms=32, win_shift_ms=16,
                       FFT_SIZE=2048, n_filters=50, n_ceps=13,
                       pre_emphasis_coef=0.95):
    ret = MFCCExtractor(fs, win_length_ms, win_shift_ms, FFT_SIZE, n_filters,
                       n_ceps, pre_emphasis_coef)
    return ret

def extract(fs, signal=None, diff=False, **kwargs):
    signal = cast['float'](signal)
    ret = get_mfcc_extractor(fs, **kwargs).extract(signal)
    return ret