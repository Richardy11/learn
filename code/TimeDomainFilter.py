import numpy as np

class TimeDomainFilter:
    """ A Python implementation of a time domain filter """
    def __init__(self, eps_zc = 0.01, eps_ssc = 0.01):
        """
        Constructor

        Parameters
        ----------
        eps_zc : float
            Threshold to count a zero crossing
        eps_ssc : float
            Threshold to count a sign-change in the slope
        
        Returns
        -------
        obj
            A TimeDomainFilter object
        """
        self.__eps_zc = eps_zc
        self.__eps_ssc = eps_ssc

    def filter(self, x ):
        """
        Compute time domain features from a window of EMG data

        Parameters
        ----------
        x : numpy.ndarray (n_samples, n_channels)
            Input window of raw EMG

        Returns
        -------
        numpy.ndarray (5 x n_channels,)
        """
        # mean absolute value
        mav = np.mean( np.abs( x ), axis = 0 )
        
        # variance
        var = np.var( x, axis = 0 )
        
        # waveform length
        wl = np.sum( np.abs( np.diff( x, axis = 0 ) ), axis = 0 )
        
        # zero crossings
        zc = np.dstack( [ np.abs( x[1:,:] ) > self.__eps_zc,
                          np.abs( x[:-1,:] ) > self.__eps_zc,
                          np.multiply( x[1:,:], x[:-1,:] ) < 0 ] )
        zc = np.sum( np.sum( zc, axis = 2 ) == 3, axis = 0 )

        # slope-sign change
        dx = np.gradient( x, axis = 0 )
        ssc = np.dstack( [ np.abs( dx[1:,:] ) > self.__eps_ssc,
                           np.abs( dx[:-1,:] ) > self.__eps_ssc,
                           np.multiply( dx[1:,:], dx[:-1,:] ) < 0 ] )
        ssc = np.sum( np.sum( ssc, axis = 2 ) == 3, axis = 0 )

        return np.vstack( [ mav, var, wl, zc, ssc ] ).flatten()

if __name__ == '__main__':
    data = np.random.rand( 1000, 8 )
    td5 = TimeDomainFilter()
    features = td5.filter( data )
    print( features )