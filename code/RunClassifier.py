import numpy as np
from sklearn.preprocessing import RobustScaler
from SpatialClassifier import SpatialClassifier
from ModifiedEASRC import ModifiedEASRC

class RunClassifier:
    def __init__(self, class_subdict_size, Xtrain, ytrain, classes, perc_lo, perc_hi, threshold_ratio):
        self.pred = 0
        self.NUM_CLASSES = len(classes)
        self.class_subdict_size = class_subdict_size

        self.mdl = []
        self.scaler = []

        self.init_classifiers(Xtrain, ytrain, classes, perc_lo, perc_hi, threshold_ratio)



#-----------------------------------------------------------------------------------------------------------------------
# PRIVATE METHODS
#-----------------------------------------------------------------------------------------------------------------------
    def emg_classify(self, feat, onset_threshold, CLASSIFIER = 'Spatial', onset_threshold_enable = True, return_aux_data = False):

        pred = self.pred
        feat_scaled = self.scaler.transform( feat.reshape( 1, -1 ) )
        axes_projection = []
            
        if CLASSIFIER == 'EASRC':
            if not onset_threshold_enable:
                pred = int( self.mdl.predict( feat_scaled )[0] )
            elif np.mean(feat[:,:8]) > onset_threshold:
                pred = int( self.mdl.predict( feat_scaled )[0] )    # classify feature vectore
            else:
                pred = 0
        elif CLASSIFIER == 'Spatial':
            pred,  axes_projection = self.SpatialClassifier.simultaneous_residuals(feat.reshape( 1, -1 ), False)
        elif CLASSIFIER == 'Simultaneous':
            pred,  axes_projection = self.SpatialClassifier.simultaneous_residuals(feat.reshape( 1, -1 ), True)
        else:
            pred = 0
        if return_aux_data:
            return pred, axes_projection
        else:
            return pred

    def init_classifiers(self, Xtrain, ytrain, classes, perc_lo, perc_hi, threshold_ratio, recalculate_RLDA = True):

        # create scaler
        self.scaler = RobustScaler()
        scaler = self.scaler
        scaler.fit( Xtrain )
        Xtrain_scaled = scaler.transform( Xtrain )

        self.mdl = ModifiedEASRC()
        self.mdl.train( Xtrain_scaled, ytrain )
        
        if recalculate_RLDA:
            self.SpatialClassifier = SpatialClassifier()
        self.class_pairs, self.radius = self.SpatialClassifier.train(   Xtrain = Xtrain, 
                                                                        ytrain = ytrain,
                                                                        classes = classes, 
                                                                        perc_lo = perc_lo, 
                                                                        perc_hi = perc_hi, 
                                                                        threshold_ratio = threshold_ratio,
                                                                        recalculate_RLDA = recalculate_RLDA )
        self.mu = []
        for i in range(len(classes)):
            self.mu.append(np.mean( Xtrain[ytrain==i,:] , axis=0 ))

        return self.class_pairs, self.radius
