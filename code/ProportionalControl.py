import numpy as np
from copy import deepcopy
from collections import deque

import scipy.io as sio
import scipy.stats as stats
import scipy.special as scsp

class Proportional_Control:
    def __init__(self, X, L, onset_threshold, classes, proportional_low = [0,0,0,0,0,0,0], proportional_high = [0,0,0,0,0,0,0]):
        
        self._classes = deepcopy(classes)
        self._n_classes = len(self._classes)
        self._range = np.zeros((2,self._n_classes))
        self._rms = deque()

        self.auto_percentile_low = 10
        self.auto_percentile_high = 50

        self._onset_threshold = onset_threshold

        self.Xtrain = X
        self.ytrain = L

        self.prop_limits = {}
        self.prop_percentile = {}
        for cl in self._classes:
            if cl.upper() != 'REST':
                self.prop_limits[cl] = [ 0, 0 ]
                self.prop_percentile[cl] = [ 0, 0 ]


        if len(np.unique(proportional_high)) == 1 and np.unique(proportional_high) == 0:
            self._init_limits()
        else:
            self.set_limits(proportional_low, proportional_high)

    def _init_limits(self):
                
        for key, cl in enumerate(self._classes):
            if cl.upper() != 'REST':
                class_temp = self.Xtrain[ :8, self.ytrain == key ]
                class_means = np.mean(class_temp,0)

                z_idx = np.where(class_means > self._onset_threshold)[0]
                z_scores = stats.zscore(class_means[z_idx])

                p_values = np.zeros(class_means.shape[0])

                for j in range(z_idx.shape[0]):
                    p_values[z_idx[j]] = 0.5 * (1 + scsp.erf(z_scores[j] / np.sqrt(2)))

                try:
                    idx = (np.abs(p_values - (self.auto_percentile_low/100) )).argmin()
                except:
                    Warning('Delete previous session data, or revert to oprevious class list')

                self._range[0][key] = class_means[idx]
                self.prop_limits[cl][0] = class_means[idx]

                idx = (np.abs(p_values - (self.auto_percentile_high/100) )).argmin()

                self._range[1][key] = class_means[idx]
                self.prop_limits[cl][1] =  class_means[idx]

                self.prop_percentile[cl][0] = self.auto_percentile_low/100
                self.prop_percentile[cl][1] = self.auto_percentile_high/100
    
    def set_limits(self, proportional_low, proportional_high ):

        for key, cl in enumerate(self._classes):
            if cl.upper() != 'REST':
                self.prop_limits[cl][0] = proportional_low[key-1]
                self.prop_limits[cl][1] = proportional_high[key-1]
        
        self.proportional_ranges()

    def proportional_ranges(self):

        closest_low = []
        closest_high = []

        for key, cl in enumerate(self._classes):

            if cl.upper() != 'REST':

                class_temp = self.Xtrain[ :8, self.ytrain == key ]
                class_means = np.mean(class_temp,0)

                z_idx = np.where(class_means > self._onset_threshold)[0]

                z_scores = stats.zscore(class_means[z_idx])

                p_values = np.zeros(class_means.size)

                closest_low.append(np.abs(class_means - self.prop_limits[cl][0]).argmin())
                closest_high.append(np.abs(class_means - self.prop_limits[cl][1]).argmin())

                for j in range(z_idx.size):
                    p_values[z_idx[j]] = 0.5 * (1 + scsp.erf(z_scores[j] / np.sqrt(2)))

                self.prop_percentile[cl][0] = p_values[closest_low[-1]]
                self.prop_percentile[cl][1] = p_values[closest_high[-1]]

                self._range[0][key] = self.prop_limits[cl][0]
                self._range[1][key] = self.prop_limits[cl][1]

    def proportional_percentile_rearrange(self, X, L ,onset_threshold):

        self._onset_threshold = onset_threshold
        self.Xtrain = X
        self.ytrain = L

        for key, cl in enumerate(self._classes):

            if cl.upper() != 'REST':

                class_temp = self.Xtrain[ :8, self.ytrain == key ]
                class_means = np.mean(class_temp,0)

                z_idx = np.where(class_means > self._onset_threshold)[0]

                z_scores = stats.zscore(class_means[z_idx])

                p_values = np.zeros(class_means.size)

                for j in range(z_idx.size):
                    p_values[z_idx[j]] = 0.5 * (1 + scsp.erf(z_scores[j] / np.sqrt(2)))

                idx = (np.abs(p_values - self.prop_percentile[cl][0])).argmin()

                self._range[0][key] = class_means[idx]

                idx = (np.abs(p_values - self.prop_percentile[cl][1])).argmin()

                self._range[1][key] = class_means[idx]       

    def Proportional(self, current_sample, pred, prop_smooth):
        
        current_mean = np.mean(current_sample[:, :8])
        self._rms.append(current_mean)

        if len(self._rms) > prop_smooth:
            self._rms.popleft()

        current_mean_rms = np.mean(self._rms)

        if pred == 0:
            return 0
        else:
            if current_mean_rms < self._range[0][pred]:
                return 0
            elif current_mean_rms > self._range[1][pred]:
                return 1
            else:
                v = ( current_mean_rms - self._range[0][pred] ) / ( self._range[1][pred] - self._range[0][pred] )
                return v

    @property
    def thresholds( self ):
        range_dict = {}
        for key, value in enumerate(self._classes):
            if value.upper() != 'REST':
                range_dict[value] = self._range[:,key]
        return range_dict
        
if __name__ == '__main__':

    from TrainingDataGeneration import TrainingDataGenerator
    from SpatialClassifier import SpatialClassifier
    from OnsetThreshold import Onset
    import os, pickle

    SUBJECT = 1
    offline_test = False

    PARAM_FILE = os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ), 
                 'data', 'Subject_%s' % SUBJECT, 'RESCU_Protocol_Subject_%s.pkl' % SUBJECT  )

    with open( PARAM_FILE, 'rb' ) as pkl:
        Parameters = pickle.load( pkl )

    EMG_SCALING_FACTOR = 10000.0

    tdg = TrainingDataGenerator()
    # compute training features and labels
    Xtrain, ytrain = tdg.emg_pre_extracted( CLASSES = Parameters['Calibration']['CUE_LIST'],
                                            PREVIOUS_FILE = '',
                                            practice_session = 1,
                                            CARRY_UPDATES = Parameters['Misc.']['CARRY_UPDATES'],
                                            total_training_samples = Parameters['Classification']['TOTAL_TRAINING_SAMPLES'],
                                            onset_threshold_enable = Parameters['Classification']['ONSET_THRESHOLD'],
                                            onset_scaler = Parameters['Calibration']['CALIB_ONSET_SCALAR'],
                                            gain = Parameters['General']['ELECTRODE_GAIN'],
                                            fft_length = Parameters['General']['FFT_LENGTH'],
                                            EMG_SCALING_FACTOR = EMG_SCALING_FACTOR,
                                            emg_window_size = Parameters['General']['WINDOW_SIZE'],
                                            emg_window_step = Parameters['General']['WINDOW_STEP'],
                                            CALIBRATION = Parameters['Calibration']['CALIB_METHOD'])
    
    Classifier = SpatialClassifier()
    Classifier.train(Xtrain = Xtrain, ytrain = ytrain, classes = Parameters['Calibration']['CUE_LIST'])
    onset_threshold_calculation = Onset( onset_scalar = Parameters['Classification']['ONSET_SCALAR'] )
    onset_threshold = onset_threshold_calculation.onset_threshold(  Xtrain = Xtrain, 
                                                                    ytrain = ytrain, 
                                                                    gain = Parameters['General']['ELECTRODE_GAIN'],
                                                                    classifier = 'EASRC')  
        
    prop = Proportional_Control(X = Xtrain.T, L = ytrain, onset_threshold = onset_threshold, classes = Parameters['Calibration']['CUE_LIST'])

    Xtrain = Xtrain[ytrain != 0,:]

    residuals = np.zeros(len(Parameters['Calibration']['CUE_LIST']))

    
    for i in Xtrain:
        pred = Classifier.simultaneous_residuals(i, simultaneous = False)[0]
        prop_val = prop.Proportional( (i/1).reshape(1,-1), pred-1, Parameters['Proportional Control']['PROP_SMOOTH'] )
        print(prop_val)