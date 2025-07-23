import numpy as np
import pickle as pkl

class LinearDiscriminantAnalysis:
    """ Python implementation of linear discriminant analysis model """
    def __init__(self, X, y):
        """
        Constructor

        Parameters
        ----------
        X : numpy.ndarray (n_samples, n_features)
            Training data
        y : numpy.ndarray (n_samples,)
            Training labels

        Returns
        -------
        obj
            A LinearDiscriminantAnalysis model
        """
        n_classes = np.unique( y ).shape[0]
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        self.__model = LDA( n_components = n_classes - 1 )
        
        self.train( X, y )

    def train(self, X, y):
        """
        Train the model

        Parameters
        ----------
        X : numpy.ndarray (n_samples, n_features)
            Training data
        y : numpy.ndarray (n_samples,) if classifier, (n_samples, n_outputs) if regressor
            Training labels
        """
        self.__model = self.__model.fit( X, y )

    def predict(self, X):
        """
        Estimate output from given input

        Parameters
        ----------
        X : numpy.ndarray (n_samples, n_features)
            Testing data

        Returns
        -------
        numpy.ndarray (n_samples,) if classifier, (n_samples, n_outputs) if regressor
            Estimated output
        """
        return self.__model.predict( X )

if __name__ == '__main__':
    import itertools
    
    from sklearn.datasets import load_digits, load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix

    import matplotlib as mpl
    from mpl_toolkits import axes_grid1
    import matplotlib.pyplot as plt

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
            plt.xticks( tick_marks, labels, rotation=45 )
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
    plt_count = 0

    for task in [ 'classification', 'regression' ]:
        ax = fig.add_subplot( 1, 2, plt_count + 1 )
        plt_count += 1
        
        if task == 'classification': data = load_digits()
        elif task == 'regression': data = load_boston()
    
        #Xtrain, Xtest, ytrain, ytest = train_test_split( data.data, data.target, test_size = 0.33 )
        CUE_LIST = ['index_finger','middle_finger','ring_finger','little_finger','thumb','pronate','power','index','tripod','pinch','key']

        Xtrain = []
        ytrain = []
        Xtest = []
        ytest = []
        class_count = 0

        for movement in CUE_LIST:
            train = pkl.load( open( 'Data/Sonomyography_Subject_1_' + movement + '.pkl', "rb" ) )
            for trial in range(train[movement].size):
                if trial < train[movement].size - 1:
                    Xtrain.append(np.vstack(train[movement][0]['features'])[100:250,:40])
                    ytrain.append( class_count * np.ones( ( Xtrain[-1].shape[0], ) ) )
                    
                else:
                    Xtest.append(np.vstack(train[movement][0]['features'])[100:250,:40])
                    ytest.append( class_count * np.ones( ( Xtest[-1].shape[0], ) ) )
                    
            class_count += 1
        
        Xtrain = np.vstack(Xtrain)
        ytrain = np.hstack( ytrain )
        Xtest = np.vstack(Xtest)
        ytest = np.hstack( ytest )

        if task == 'classification': 
            mdl = LinearDiscriminantAnalysis( Xtrain, ytrain )        
            yhat = mdl.predict( Xtest )

        if task == 'classification': cm = confusion_matrix( ytest, yhat, labels = data.target_names, ax = ax, show = False )
        elif task == 'regression':
            ax.text( 0.4, 0.45, 'N/A', fontsize = 30 )
            # ax.plot( ytest, label = 'Actual' )
            # ax.plot( yhat, label = 'Predicted' )

            # ax.set_xlabel( 'Sample' )
            # ax.set_ylabel( 'Output Value' )
            # leg = ax.legend( frameon = False )

        ax.set_title( 'Digits Dataset Classification' if task == 'classification' else 'Boston Housing Regression' )
    plt.tight_layout()
    plt.show()