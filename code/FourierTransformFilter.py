import numpy as np

class FourierTransformFilter:
    """ A Python implementation of a fast Fourier transform filter """
    def __init__(self, fftlen = 256):
        """ 
        Constructor 

        Parameters
        ----------
        fftlen : int
            The length of the fast Fourier transform window (should be a power of 2)

        Returns
        -------
        obj
            A FourierTransformFilter object

        Notes
        -----
        This filter only returns the first half of the magnitude coefficients.
        This is because for a real-valued input signal, the magnitude coefficients are symmetric.
        """
        self.__fftlen = fftlen

    def filter(self, x):
        """ 
        Compute FFT magnitude coefficients of each channel from window of EMG data

        Parameters
        ----------
        x : numpy.ndarray (n_samples, n_channels)
            Input data to filter
        
        Returns
        -------
        numpy.ndarray (n_channels x fftlen / 2,)
            Filtered output data
        """
        #TODO
        feat = np.fft.fft( x, n = self.__fftlen, axis = 0 )
        return np.abs( feat[0:int(self.__fftlen/2.0)] ).flatten()

    def filterAdapt(self, x):
        """ 
        Compute FFT magnitude coefficients of each channel from window of EMG data

        Parameters
        ----------
        x : numpy.ndarray (n_samples, n_channels)
            Input data to filter
        
        Returns
        -------
        numpy.ndarray (n_channels x fftlen / 2,)
            Filtered output data
        """
        feat32 = np.fft.fft( x, n = 32, axis = 0 )
        feat32 = np.abs( feat32[0:16] ).flatten()

        feat64 = np.fft.fft( x, n = 64, axis = 0 )
        feat64 = np.abs( feat64[0:32] ).flatten()

        feat = np.concatenate((feat32[:3*x.shape[1]], feat64[6*x.shape[1]:19*x.shape[1]], feat32[-6*x.shape[1]:]), axis=0)

        return feat

if __name__ == '__main__':
    data = np.random.rand( 1024, 8 )
    fft = FourierTransformFilter(fftlen = 64)
    features = fft.filter( data )
    fft.filterAdapt( data )
    print( features )
