class VelocityRampFilter:
    """ A Python implementation of a velocity ramp output filter """
    def __init__( self, n_classes, increment = 1, decrement = 2, max_bin_size = 50, enabled = True ):
        """
        Constructor

        Parameters
        ----------
        n_classes : int
            The number of possible classes
        increment : int
            The amount to increment each bin by
        decrement : int
            The amount to decrement each bin by
        max_bin_size : int
            The maximum size of any class' bin

        Returns
        -------
        obj
            A VelocityRampFilter object

        Note
        ----
        This class assumes that the REST class is label 0
        """
        self.__num_classes = n_classes
        self.__increment = increment
        self.__decrement = decrement
        self.__max_bin_size = max_bin_size
        self.__enabled = enabled
        self.__bins = [ 0 ] * self.__num_classes

    def filter( self, pred ):
        """
        Compute velocity output from classifier prediction
        
        Parameters
        ----------
        pred : int
               Current classifier label prediction

        Returns
        -------
        float
            Proportional velocity output
        """
        if self.__enabled == True:
            for i in range( self.__num_classes ):
                if pred == i:
                    self.__bins[ i ] += self.__increment
                    self.__bins[ i ] = min( [ self.__bins[ i ], self.__max_bin_size ] )
                else:
                    self.__bins[ i ] -= self.__decrement
                    self.__bins[ i ] = max( [ self.__bins[ i ], 0 ] )
            if pred == 0: return 0.0
            else: return self.__bins[ pred ] / self.__max_bin_size
        else:
            if pred == 0: return 0.0
            else: return 1

    def reset( self ):
        """
        Resets the bin count for each class
        """
        self.__bins = [ 0 ] * self.__num_classes

if __name__ == '__main__':
    N_CLASSES = 7
    MAX_BIN_SIZE = 10

    predictions = []
    for i in range( N_CLASSES ):
        predictions.extend( [ i ] * MAX_BIN_SIZE )
    
    output = []
    vramp = VelocityRampFilter( n_classes = N_CLASSES, max_bin_size = MAX_BIN_SIZE )
    for pred in predictions:
        output.append( vramp.filter( pred ) )

    print( output )