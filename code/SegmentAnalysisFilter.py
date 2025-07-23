import numpy as np

class SegmentAnalysisFilter:
    @staticmethod
    def mode( x, axis = 0 ):
        """
        Compute the mode of a numpy array

        Parameters
        ----------
        x : numpy.ndarray
            The numpy array to find the mode of
        axis : int
            The axis to compute the mode over

        Return
        ------
        int
            The mode of the specified axis
        numpy.ndarray
            The counts for each unique element on the specified axis
        """
        scores = np.unique( np.ravel( x ) )
        
        testshape = list( x.shape )
        testshape[axis] = 1

        oldmostfreq = np.zeros( testshape )
        oldcounts = np.zeros( testshape )

        for score in scores:
            template = ( x == score )
            counts = np.expand_dims( np.sum( template, axis ), axis )
            mostfrequent = np.where( counts > oldcounts, score, oldmostfreq )
            oldcounts = np.maximum( counts, oldcounts )
            oldmostfreq = mostfrequent

        return mostfrequent, oldcounts
    
    def __init__(self, sensitivities, lag_threshold):
        """
        Constructor

        Parameters
        ----------
        sensitivities : iterable (n_classes,)
            The class onset sensitivities
        lag_threshold : float
            The minimum amount of lag (as a ratio) to be tolerated
        """
        sensitivities = np.array( sensitivities )
        if not all( sensitivities >= 0.0 ):
            raise ValueError( 'Invalid class sensitivities:', sensitivities )

        if lag_threshold < 1.0:
            raise ValueError( 'Invalid lag threshold:', lag_threshold )

        self._sensitivities = sensitivities.astype( int )
        self._lag_threshold = lag_threshold

    def filter(self, raw_predictions, filtered_predictions):
        """
        Analyze if a segment should be updated or not

        Parameters
        ----------
        raw_predictions : numpy.ndarray (n_samples,)
            The raw predictions for a segment
        filtered_predictions : numpy.ndarray (n_samples,)
            The filtered predictions after post-classification filtering for a segment

        Return
        ------
        bool
            True if segment should be updated, False else
        """
        # compute comparative mode
        mode_flag = int( SegmentAnalysisFilter.mode( raw_predictions )[0] ) != int( SegmentAnalysisFilter.mode( filtered_predictions )[0] )

        # compute class entry lag
        onset = int( np.argmax( np.abs( np.diff( filtered_predictions ) ) > 0 ) + 1 )
        lag_flag = ( onset / self._sensitivities[ filtered_predictions[ onset ] ] ) > self._lag_threshold 

        return ( mode_flag or lag_flag )

    def set_sensitivities( self, sensitivities ):
        """
        Set the filter's internal class sensitivities

        Parameters
        ----------
        sensitivities : iterable (n_classes,)
            The new class sensitivities
        """
        sensitivities = np.array( sensitivities )
        if all( sensitivities >= 0.0 ):
            self._sensitivities = sensitivities.astype( int )

    def get_sensitivities( self ):
        """
        Get the filter's internal class sensitivities

        Return
        ------
        numpy.ndarray (n_classes,)
            The current class sensitivities
        """
        return self._sensitivities

    def set_lag_threshold( self, lag_threshold ):
        """
        Set the filter's lag threshold

        Parameters
        ----------
        lag_threshold : float
            The new lag threshold
        """
        if lag_threshold >= 1.0:
            self._lag_threshold = lag_threshold

    def get_lag_threshold( self ):
        """
        Get the current lag threshold

        Return
        ------
        int
            The current lag threshold
        """
        return self._lag_threshold

if __name__ == '__main__':
    N_SEGMENTS = 10
    N_SAMPLES = 100

    update = []
    analyzer = SegmentAnalysisFilter( sensitivities = [ 5.0, 10.0 ], lag_threshold = 2.0 )
    for _ in range( N_SEGMENTS ):
        raw_pred = ( np.random.rand( N_SAMPLES ) > 0.5 ).astype( int )
        filtered = ( np.random.rand( N_SAMPLES ) > 0.5 ).astype( int )

        update.append( analyzer.filter( raw_pred, filtered ) )

    print( update )