import scipy.stats as stats
import scipy.special as scsp
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class Onset:
    def __init__(self, outlier_inclusion = 50, onset_scalar = 1.1):
        self._outlier_inclusion = outlier_inclusion
        self._onset_scalar = onset_scalar
        self._onset_scalar_calc = 0
        self.stdscaler = MinMaxScaler()
        self.proj = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')

    def onset_threshold(self, Xtrain, ytrain, gain = 4, classifier = '', threshold_ratio = 0.3):

        if classifier == 'Spatial':
            self.stdscaler.fit(Xtrain)
            Xtrain_trans = self.stdscaler.transform(Xtrain)
            Xtrain_trans = self.proj.fit_transform(Xtrain_trans, ytrain)
            mu = []
            for i in range(len(np.unique( ytrain ))):
                mu.append(np.mean( Xtrain_trans[ytrain==i,:] , axis=0 ))

            #TODO: integrate better
            onset_threshold = 1000000
            for key, i in enumerate(mu):
                if key > 0:
                    axis = np.linalg.norm(i - mu[0])
                    onset_threshold = axis*threshold_ratio if axis*threshold_ratio < onset_threshold else onset_threshold

            return onset_threshold

        else:

            rest_data = Xtrain[ytrain == 0,:8]

            rest_means = np.mean( rest_data, 1 )
            z_scores = stats.zscore(rest_means)
            p_values = np.zeros(rest_means.size)

            for j in range(rest_means.size):
                p_values[j] = 0.5 * (1 + scsp.erf(z_scores[j] / np.sqrt(2)))

            idx = ( np.abs( p_values - (self._outlier_inclusion/100) ) ).argmin()
            
            min_mean = 1
            for k in range(1,len(np.unique( ytrain ))):
                temp_class_mean = np.mean(np.mean( Xtrain[ytrain == k,:8], 1 ),0)
                min_mean = temp_class_mean if temp_class_mean < min_mean else min_mean
            
            default_thresh = rest_means[idx] * self._onset_scalar# * gain/4
            dynamic_thresh = ( (min_mean - np.mean( rest_means, 0 ) ) * 0.3 + np.mean( rest_means, 0 ) )# * gain/4

            onset_threshold = default_thresh if dynamic_thresh < default_thresh else dynamic_thresh

            self._onset_scalar_calc = onset_threshold / rest_means[idx]

            return onset_threshold

    def calib_onset_threshold(self, rest_data, gain = 4):

        rest_means = np.mean( rest_data, 1 )
        z_scores = stats.zscore(rest_means)
        p_values = np.zeros(rest_means.size)

        for j in range(rest_means.size):
            p_values[j] = 0.5 * (1 + scsp.erf(z_scores[j] / np.sqrt(2)))

        idx = ( np.abs( p_values - (self._outlier_inclusion/100) ) ).argmin()
        
        default_thresh = rest_means[idx] * self._onset_scalar

        onset_threshold = default_thresh

        return onset_threshold

    def get_current_scalar(self):
        return self._onset_scalar_calc