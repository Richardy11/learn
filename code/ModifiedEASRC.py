import numpy as np
import numpy.matlib
from sklearn.preprocessing import normalize

class ModifiedEASRC:
    """ Python implementation of sparse representation model """
    @staticmethod 
    def sigmoid( x ):
        """
        Compute the sigmoid transformation

        Parameters
        ----------
        x : numpy.ndarray (m, n, k, ...)
            The sigmoidal input
        
        Returns
        -------
        numpy.ndarray (m, n, k, ...)
            The transformed values
        """
        return np.divide( 1.0, 1.0 + np.exp( -x ) )

    def __init__(self, elm_layers = 100, src_conf = 0.4, n_dot = 3):
        """
        Constructor

        Parameters
        ----------
        elm_layers : int
            The number of hidden layers for the ELM
        src_conf : float
            The confidence threshold, under which we pass to SRC for classification
        n_dot : int
            The number of vectors to consider in the average for each class' dot product

        Returns
        -------
        obj
            A ModifiedEASRC model
        """
        self.__dictionary = None    # X
        self.__labels = None        # y

        self.__elm = { 'weights' : None, 'biases' : None, 'beta' : None }
        self.__elm_layers = elm_layers

        self.__src_conf = src_conf
        self.__n_dot = n_dot
        
        # self.train( X, y )

    def train(self, X, y, seed = None):
        """
        Train the model

        Parameters
        ----------
        X : numpy.ndarray (n_samples, n_features)
            Training data
        y : numpy.ndarray (n_samples,)
            Training labels
        """
        self.__dictionary = normalize( X.T, norm = 'l2', axis = 0 )
        self.__labels = y

        # train ELM
        if seed is not None: np.random.seed( seed )

        n_classes = np.unique( self.__labels ).shape[0]
        n_samples, n_features = X.shape

        self.__elm['weights'] = np.random.uniform( 0, 1, ( self.__elm_layers, n_features ) )
        self.__elm['biases'] = np.random.uniform( 0, 1, ( self.__elm_layers, ) )
        self.__elm['beta'] = np.zeros( ( n_classes, self.__elm_layers ) )

        H = ModifiedEASRC.sigmoid( self.__elm['weights'] @ self.__dictionary + np.matlib.repmat( self.__elm['biases'], n_samples, 1 ).T ).T
        Y = np.zeros( ( n_samples, n_classes ) )
        for c in range( 0, n_classes ):
            Y[:, c] = ( self.__labels == c )

        C = np.exp( np.arange( -4, 4, 0.2 ) )
        LOO = np.inf * np.ones( ( C.shape[0] ) )

        U, S, _ = np.linalg.svd( H.T @ H )
        A = H @ U
        B = A.T @ Y

        # print( 'H:', H.shape ) # MATLAB: n_samples x hidden_layer
        # print( 'U:', U.shape ) # MATLAB: hidden_layer x hidden_layer
        # print( 'S:', S.shape ) # MATLAB: 1 x hidden_layer
        # print( 'A:', A.shape ) # MATLAB: n_samples x hidden_layer
        # print( 'B:', B.shape ) # MATLAB: hidden_layer x n_outputs

        for i in range( C.shape[0] ):
            tmp = np.multiply( A, np.matlib.repmat( np.divide( 1.0, S + C[i] ), n_samples, 1 ) )
            hat = np.sum( np.multiply( tmp, A ), axis = 1 )
            yhat = tmp @ B

            errdiff = np.divide( ( Y - yhat ), np.matlib.repmat( 1.0 - hat, n_classes, 1 ).T )
            frob = np.linalg.norm( errdiff, ord = 'fro' )
            LOO[i] = frob ** 2 / n_samples

        idx = np.argmin( LOO )
        opt = C[idx]
        beta = np.multiply( U, np.matlib.repmat( np.divide( 1.0, S + opt ), self.__elm_layers, 1 ) ) @ B
        self.__elm['beta'] = beta.T

    def load(self, X, y, W, b, beta):
        """
        Load the variables from a previous model

        Parameters
        ----------
        X : numpy.ndarray (n_samples, n_features)
            Training dictionary
        y : numpy.ndarray (n_samples,)
            Training labels
        W : numpy.ndarray (n_elm_layers, n_features)
            ELM weights
        b : numpy.ndarray (n_elm_layers)
            ELM biases
        beta : numpy.ndarray (n_classes, n_elm_layers)
            ELM beta values
        """
        self.__dictionary = normalize( X.T, norm = 'l2', axis = 0 )
        self.__labels = y

        self.__elm[ 'weights' ] = W
        self.__elm['biases'] = b
        self.__elm['beta'] = beta

    def predict(self, X):
        """
        Estimate output from given input

        Parameters
        ----------
        X : numpy.ndarray (n_samples, n_features)
            Testing data

        Returns
        -------
        numpy.ndarray (n_samples,)
            Estimated output
        """
        if len( X.shape ) == 1: X = np.expand_dims( X, axis=0 )
        X = normalize( X.T, norm = 'l2', axis = 0 )

        pred = np.zeros( ( X.shape[1], ), np.int )
        for i in range( 0, X.shape[1] ):
            # get the current sample
            x = X[:, i]
            
            # ELM classification
            H = ModifiedEASRC.sigmoid( self.__elm['weights'] @ x + self.__elm['biases'] )
            yhat = self.__elm['beta'] @ H
            yhat_diff = np.abs( yhat - np.amax( yhat ) )

            candidates = np.where( yhat_diff < self.__src_conf )[0]
            n_candidates = candidates.shape[0]

            if n_candidates == 1: 
                pred[i] = candidates[0]
            else:
                # SRC classification
                dist = np.zeros( ( n_candidates, ) )
                adaptX = [ self.__dictionary[:, self.__labels == k] for k in candidates ]
                adapty = np.hstack( [ candidates[k] * np.ones( adaptX[k].shape[1] ) for k in range( candidates.shape[0] ) ] )
                adaptX = np.hstack( adaptX )

                dot = np.dot( x, adaptX )
                for j in range( 0, n_candidates ):
                    dist[j] = -1.0 * np.partition( -dot[ np.equal( adapty, candidates[j] ) ], self.__n_dot )[:self.__n_dot].mean()
                pred[ i ] = candidates[ np.argmax( dist ) ]

        return pred

    @property
    def elm(self):
        return self.__elm['weights'], self.__elm['biases'], self.__elm['beta']

if __name__ == '__main__':
    import itertools
    import time

    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix

    import matplotlib as mpl
    mpl.use( 'QT5Agg' )

    import matplotlib.pyplot as plt
    from mpl_toolkits import axes_grid1

    def confusion_matrix( ytest, yhat, labels = [], cmap = 'viridis', ax = None, show = True ):
        """
        Computes (and displays) a confusion matrix given true and predicted classification labels

        Parameters
        ----------
        ytest : numpy.ndarray (n_samples,)
            The true labels
        yhat : numpy.ndarray (n_samples,)
            The predicted label
        labels : iterable
            The class labels
        cmap : str
            The colormap for the confusion matrix
        ax : axis or None
            A pre-instantiated axis to plot the confusion matrix on
        show : bool
            A flag determining whether we should plot the confusion matrix (True) or not (False)

        Returns
        -------
        numpy.ndarray
            The confusion matrix numerical values [n_classes x n_classes]
        axis
            The graphic axis that the confusion matrix is plotted on or None

        """
        def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
            """Add a vertical color bar to an image plot."""
            divider = axes_grid1.make_axes_locatable(im.axes)
            width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
            pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
            current_ax = plt.gca()
            cax = divider.append_axes("right", size=width, pad=pad)
            plt.sca(current_ax)
            return im.axes.figure.colorbar(im, cax=cax, **kwargs)

        cm = sk_confusion_matrix( ytest, yhat )
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        if ax is None:    
            fig = plt.figure()
            ax = fig.add_subplot( 111 )

        try:
            plt.set_cmap( cmap )
        except ValueError: cmap = 'viridis'

        im = ax.imshow( cm, interpolation = 'nearest', vmin = 0.0, vmax = 1.0, cmap = cmap )
        add_colorbar( im )

        if len( labels ):
            tick_marks = np.arange( len( labels ) )
            plt.xticks( tick_marks, labels, rotation=0 )
            plt.yticks( tick_marks, labels )

        thresh = 0.5 # cm.max() / 2.
        colors = mpl.cm.get_cmap( cmap )
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            r,g,b,_ = colors(cm[i,j])
            br = np.sqrt( r*r*0.241 + g*g*0.691 + b*b*0.068 )
            plt.text(j, i, format(cm[i, j], '.2f'),
                        horizontalalignment = "center",
                        verticalalignment = 'center',
                        color = "black" if br > thresh else "white")

        plt.ylabel('Actual')
        plt.xlabel('Predicted')

        ax.set_ylim( cm.shape[0] - 0.5, -0.5 )
        plt.tight_layout()
        if show: plt.show( block = True )
        
        return cm, ax

    fig = plt.figure( figsize = (10.0, 5.0) )
    ax = fig.add_subplot( 111 )
    
    data = load_digits()
    Xtrain, Xtest, ytrain, ytest = train_test_split( data.data, data.target, test_size = 0.33 )
    
    mdl = ModifiedEASRC( src_conf = 0.4 )
    mdl.train( Xtrain, ytrain )
    t0 = time.perf_counter()       
    yhat = mdl.predict( Xtest )
    tf = time.perf_counter()
    print( 1e3 * ( tf - t0 ) / ytest.size, 'MS PER PREDICTION' )

    cm = confusion_matrix( ytest, yhat, labels = data.target_names, ax = ax, show = False )

    ax.set_title( 'Digits Dataset Classification' )

    plt.tight_layout()
    plt.show()

    ax.set_title( 'Digits Dataset Classification' )

    plt.tight_layout()
    plt.show()
