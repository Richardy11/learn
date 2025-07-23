import numpy as np
import numpy.matlib
import pickle
from DownSampler import DownSampler
from Segmentation import Segmentation

class Update:
    def __init__(self):
        self._downsample = DownSampler()

    def Update_class(self, X, L, cache, c_update):
        """
        Compute velocity output from classifier prediction
        
        Parameters
        ----------
        X : Current training dictionary

        L : Current label vector

        cache : cache of data vectors to incorporate

        c_update: class to update

        downsample: number of vectors to maintain

        Returns
        -------
        Matrix of updated 
        """
        X_new = X.copy().T
        # find class to downsample
        class_idx = np.where(L == c_update)
        class_to_downsample = X_new[ :, class_idx[-1] ]
        # downsample
        downsample = class_to_downsample.shape[1] - cache.shape[0]
        #class_maintain = self._downsample.k_medoids(class_to_downsample, downsample)
        k_means = self._downsample.k_means(class_to_downsample, downsample)
        
        # conctenate cached data and maintained class
        #new_class = np.concatenate((class_to_downsample[:,class_maintain[0]], cache.T), axis = 1)
        new_class = np.concatenate((k_means.T, cache.T), axis = 1)
        X_new[ :, class_idx[-1] ] = new_class

        return X_new

    def Update_class_LDA(self, X, L, cache, c_update, downsample = 10):
        """
        Compute velocity output from classifier prediction
        
        Parameters
        ----------
        X : Current training dictionary

        L : Current label vector

        cache : cache of data vectors to incorporate

        c_update: class to update

        downsample: number of vectors to maintain

        Returns
        -------
        Matrix of updated 
        """
        X_new = X.copy().T
        # find class to downsample
        class_idx = np.where(L == c_update)
        class_to_downsample = X_new[ :, class_idx[-1] ]
        class_size = class_to_downsample.shape[1]
        # downsample
        cmat,class_maintain = self._downsample.uniform(class_to_downsample.T, ((100-downsample)*class_size)//100)
        
        # conctenate cached data and maintained class
        new_class = np.concatenate((cmat.T, numpy.matlib.repmat(cache.T,1,6)), axis = 1)
        X_new[ :, class_idx[-1] ] = new_class

        return X_new

if __name__ == '__main__':
    
    update = Update()
    downsample = DownSampler()

    CLASSIFIER = 'LDA'
    CLASSES = ['rest', 'open', 'power', 'pronate', 'supinate']

    pkl_file = open(f"Data/Subject_1/Training_Data_Subject_1.pkl", 'rb')
    training_data = pickle.load(pkl_file)
    pkl_file.close()

    Xtrain = []
    ytrain = []
    class_count = 0
    for cue in CLASSES:
        if CLASSIFIER == 'EASRC':
            temp_class, Idx = downsample.uniform(training_data[0]['features'][cue], int(350//5) )
        else:
            temp_class = training_data[0]['features'][cue]
        Xtrain.append( temp_class )
        ytrain.append( class_count * np.ones( ( Xtrain[-1].shape[0], ) ) )
        class_count += 1

    Xtrain = np.vstack( Xtrain )[:,:8].T
    ytrain = np.hstack( ytrain )

    segmentation = Segmentation( Xtrain, ytrain )
    Segments = []
    Cache_list = []
    CR_list = []
    for i in reversed(Xtrain.T):
        segmentation.run_segmentation(i)
        if segmentation.new_activity is True:
            
            Segments.append(segmentation.activity)
            Segments_dictionary = np.hstack(Segments)
            Cache_list.append(segmentation.cache)
            Cache_dictionary = np.vstack(Cache_list)
            CR_list.append(segmentation.CR)
            CR = np.hstack(CR_list)
            segmentation.reinit()

    X = update.Update_class(Xtrain.T, ytrain, Cache_dictionary[-1], 2)
    X = update.Update_class_LDA(Xtrain.T, ytrain, Cache_dictionary[-1], 2)