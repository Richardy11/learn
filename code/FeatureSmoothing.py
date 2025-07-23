from collections import deque
import numpy as np

class Smoothing:
    """ A Python implementation of a uniform vote output filter """
    def __init__( self, Parameters ):
        self.smoothing_options = {  'FILTER_LENGTH': Parameters['Classification']['FILTER_LENGTH'],
                                    'EXPONENTIAL_FILTER_SCALE': Parameters['Classification']['EXPONENTIAL_FILTER_SCALE'],
                                    'BEZIER_PROJECTION_RATIO': Parameters['Classification']['BEZIER_PROJECTION_RATIO']}

        self.smoothing_options_keys = list(self.smoothing_options.keys())

        self.buffsize = Parameters['Classification']['FILTER_LENGTH']

        self.smoothing =  Parameters['Classification']['FEATURE_FILTER']

        self.feat_buffer = deque()

    def simple_exponential_smoothing(self, series, alpha):
        """
        Given a series, alpha, beta and n_preds (number of
        forecast/prediction steps), perform the prediction.
        """
        n_record = series.shape[0]
        results = np.zeros((n_record, series.shape[1]))

        results[0,:] = series[0,:] 
        for t in range(1, n_record):
            results[t,:] = alpha * series[t,:] + (1 - alpha) * results[t - 1, :]

        return results

    def filter( self, feat ):

        self.feat_buffer.append(feat)
        if len(self.feat_buffer) > self.buffsize:
            while len(self.feat_buffer) > self.buffsize:
            
                self.feat_buffer.popleft()
            if self.smoothing == 'Linear':
                return np.mean(self.feat_buffer, axis=0).reshape(1, -1)

            elif self.smoothing == 'Exponential':
                exp_smooth = self.simple_exponential_smoothing(np.array(self.feat_buffer), self.smoothing_options['EXPONENTIAL_FILTER_SCALE'])
                return exp_smooth[-1].reshape(1, -1)

            elif self.smoothing == 'Kalman Filter':
                pass
        
        else:
            return feat.reshape(1, -1)

    def set_params(self, filter, type, value):
        self.smoothing_options[type] = value
        self.buffsize = self.smoothing_options['FILTER_LENGTH']
        self.smoothing = filter

