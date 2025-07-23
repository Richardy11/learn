import numpy as np
import os
from copy import deepcopy as dp
import pickle
from numpy import linalg
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import os
import csv
import matplotlib.pyplot as plt

from scipy.spatial import distance

from Q15Functions import Q15Functions

class SpatialClassifier:
    def __init__(self):
        self.wrist = ['PALM DOWN', 'PALM UP', 'FLEXION', 'EXTENTION']
        self.elbow = ['ELBOW BEND', 'ELBOW EXTEND']

    def train(self, Xtrain, ytrain, classes, threshold_ratio, perc_lo, perc_hi, recalculate_RLDA = True):

        self.numClasses = len(np.unique(ytrain))
        self.classes = dp(classes)
        self.grips = dp(classes)
        for i in self.wrist:
            try:
                self.grips.remove(i)
            except: pass

        for i in self.elbow:
            try:
                self.grips.remove(i)
            except: pass

        self.testQ15 = False
        
        self.Xtrain_full = dp(Xtrain)
        self.Xtrain = Xtrain
        self.ytrain = ytrain

        if recalculate_RLDA:
            self.stdscaler = MinMaxScaler()
            self.stdscaler.fit(self.Xtrain)
        self.Xtrain = self.stdscaler.transform(self.Xtrain)

        if recalculate_RLDA:
            self.proj = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
            self.proj.fit(self.Xtrain, self.ytrain)

        if self.testQ15:
            self.q15 = Q15Functions()
            self.q15.fit(self.proj.scalings_, self.proj._max_components)
            self.Xtrain = self.q15.transform(self.Xtrain, save=True)        
        else:
            self.Xtrain = self.proj.transform(self.Xtrain)
  
        self.rest_to_class_means = []
        self.class_to_class_axes = {}
        for i in self.classes:
            self.class_to_class_axes[i] = {}
            for j in self.classes:
                self.class_to_class_axes[i][j] = {}
        self.mu = []
        self.radius = []

        for key_i, i in enumerate(self.classes):
            self.mu.append( np.mean( self.Xtrain[self.ytrain==key_i,:] , axis=0 ) )
            self.rest_to_class_means.append( self.mu[key_i] - self.mu[0] )

        self.onset_threshold_calc(0.05)

        for key_i, i in enumerate(self.classes):
            for key_j, j in enumerate(self.classes):
                self.class_to_class_axes[i][j] = self.mu[key_i] - self.mu[key_j]

            if key_i == 0:
                self.radius.append( [self.threshold_radius for i in range(len(self.classes)-1)] )
            else:
                self.radius.append( [self.threshold_radius for i in range(len(self.classes)-1)] )

            '''if key_i == 0:
                self.radius.append( np.var(self.Xtrain[self.ytrain == key_i, :], axis = 0 ) )
            else:
                self.radius.append( np.var(self.Xtrain[self.ytrain == key_i, :], axis = 0 ) )'''

        self.class_pairs = []
        for i in self.classes:
            for j in self.classes:
                if ((i in self.wrist and j not in self.wrist) or (i == 'REST')) and j != 'REST':
                    self.class_pairs.append((i, j))

        self.threshold_ratio = threshold_ratio
        self.perc_lo = perc_lo
        self.perc_hi = perc_hi

        self.onset_threshold_calc(threshold_ratio)
        self.proportional_levels(perc_lo, perc_hi)

        return self.class_pairs, self.radius

    def SeparabilityScore(self, Xtrain = [], ytrain = [], train_percentage = 90, classes = [], perc_lo = [], perc_hi = [], threshold_ratio = 0.3 ):

        if len(Xtrain) > 0 and len(ytrain) > 0:
            Xtrain_full = Xtrain
            ytrain = ytrain
            self.numClasses = len(np.unique(ytrain))

            self.classes = classes
            self.perc_lo = perc_lo
            self.perc_hi = perc_hi
            self.threshold_ratio = threshold_ratio

        else:
            Xtrain_full = dp(self.Xtrain_full)
            ytrain= dp(self.ytrain)

        train_percentage /= 100
        
        Xtrain_train = []
        Xtrain_test = []
        ytrain_train = []
        ytrain_test = []

        test_length = []

        for i in range(self.numClasses):
            
            tempClass = Xtrain_full[ytrain == i, :]
            tempClassLabels = ytrain[ytrain == i]
            tempPercentageTrain = int(len(tempClassLabels) * train_percentage)
            Xtrain_train.append(tempClass[:tempPercentageTrain, :])
            Xtrain_test.append(tempClass[tempPercentageTrain:, :])
            ytrain_train.append(tempClassLabels[:tempPercentageTrain])
            ytrain_test.append(tempClassLabels[tempPercentageTrain:])

            test_length.append(len(ytrain_test[-1]))

        Xtrain_train = np.vstack( Xtrain_train )
        Xtrain_test = np.vstack( Xtrain_test )
        ytrain_train = np.hstack( ytrain_train )
        ytrain_test = np.hstack( ytrain_test )

        testClassifier = SpatialClassifier()
        testClassifier.train(Xtrain = Xtrain_train, ytrain = ytrain_train, classes = self.classes, perc_lo = self.perc_lo , perc_hi = self.perc_hi, threshold_ratio = self.threshold_ratio )
        confmat = np.zeros((self.numClasses, self.numClasses))
        for i, val in enumerate(Xtrain_test):
            confmat[ int(testClassifier.simultaneous_residuals(val, simultaneous = False)[0]), int(ytrain_test[i]) ] += 1

        for i, val in enumerate(test_length):
            confmat[i,:] /= val
        
        return confmat

    def ClosestPointOnLine(self, a, b, p):
                ap = p-a
                ab = b-a
                result = a + np.dot(ap,ab)/np.dot(ab,ab) * ab
                return result

    def onset_threshold_calc(self, threshold_ratio = 3/10):
        self.onset_threshold = 1000000
        for key, i in enumerate(self.mu):
            if key > 0:
                axis = np.linalg.norm(i - self.mu[0])
                if axis*threshold_ratio < self.onset_threshold:
                    self.onset_threshold = axis*threshold_ratio
                    self.threshold_radius = i*threshold_ratio
                    self.threshold_radius_norm = np.linalg.norm(self.threshold_radius)
                    self.channel_max = axis

    def proportional_levels(self, perc_lo, perc_hi):
        self.proportional_range = {}
        for key, cl in enumerate(self.classes):
            if key != 0:
                tempClass = self.Xtrain[self.ytrain == key, :]
                self.proportional_range[cl] = { 'min': self.onset_threshold,
                                                'max': 0,
                                                'perc_min': perc_lo[key-1],
                                                'perc_max': perc_hi[key-1],
                                                'current_min': 0,
                                                'current_max': 0,
                                                }
                for feat in tempClass:
                    projection = self.ClosestPointOnLine(self.mu[0], self.mu[key], feat)
                    dist = np.linalg.norm(projection - self.mu[0])# - self.onset_threshold
                    self.proportional_range[cl]['max'] = dist if dist > self.proportional_range[cl]['max'] else self.proportional_range[cl]['max']

                self.proportional_range[cl]['current_min'] = self.proportional_range[cl]['min'] + ( (self.proportional_range[cl]['max'] - self.proportional_range[cl]['min'] ) * self.proportional_range[cl]['perc_min'] )
                self.proportional_range[cl]['current_max'] = self.proportional_range[cl]['min'] + (( self.proportional_range[cl]['max'] - self.proportional_range[cl]['min'] ) * self.proportional_range[cl]['perc_max'])

    def simultaneous_residuals(self, x, simultaneous = True):

        x = self.stdscaler.transform(x.reshape(1,-1))

        if self.testQ15:            
            x = self.q15.transform(x)        
        else:
            x = self.proj.transform(x)

        self.projection = []
        self.axes_distance = []

        self.distancFromRest = np.linalg.norm(x-self.mu[0])

        if self.distancFromRest < self.onset_threshold:
            if not simultaneous:
                return 0, []
            else:
                return [0, 0], []

        distance_from_axes = np.zeros(len(self.class_pairs))

        if not simultaneous:
            simultaneous_prop = 1
        else:
            simultaneous_prop = 1.25

        # PARALLEL SPATIAL CLASSIFIER BASED ON AXES

        in_cluster = np.zeros(self.numClasses)
        axes_projection = []

        for i in range(len(self.classes)):
            dist_from_mu = np.linalg.norm(self.mu[i] - x)
            if dist_from_mu < np.linalg.norm(self.radius[i]):
                in_cluster[i] = 1

        for key, i in enumerate(self.class_pairs):

            if not simultaneous and i[0] != 'REST':

                distance_from_axes[key] = 100000000

            else:

                c1 = i[0]
                c2 = i[1]

                c1_idx = self.classes.index(c1)
                c2_idx = self.classes.index(c2)

                axis_projection = self.ClosestPointOnLine(self.mu[c1_idx], self.mu[c2_idx], x)
                
                axes_projection.append(axis_projection)

                x_axis_dist = np.linalg.norm(x - axis_projection)

                distance_from_axes[key] = x_axis_dist

                self.axes_distance = dp(distance_from_axes[:key+1])

        try:
            selected_pair = self.class_pairs[np.argmin(distance_from_axes)]
        except:
            pass
        selected_projection = axes_projection[np.argmin(distance_from_axes)]

        axis_len = np.linalg.norm(self.class_to_class_axes[selected_pair[0]][selected_pair[1]])

        #prop_dist = np.linalg.norm(selected_projection-self.mu[0])

        obj_velocity = []
        velocity = []

        for cl in self.classes:
            if cl != 'REST':
                temp_projection = axes_projection[self.class_pairs.index(('REST', cl))]
                prop_dist = np.linalg.norm(temp_projection-self.mu[0])

                obj_velocity.append( (prop_dist - self.proportional_range[cl]['min']*simultaneous_prop) / (self.proportional_range[cl]['max']*simultaneous_prop -  self.proportional_range[cl]['min']*simultaneous_prop) )
                velocity.append( (prop_dist - self.proportional_range[cl]['current_min']*simultaneous_prop) / (self.proportional_range[cl]['current_max']*simultaneous_prop - self.proportional_range[cl]['current_min']*simultaneous_prop) )
                velocity[-1] = 1 if velocity[-1] > 1 else velocity[-1]
                velocity[-1] = 0 if velocity[-1] < 0 else velocity[-1]

        if 'REST' in selected_pair and not simultaneous:
            selected_pair = [selected_pair[1]]
        elif 'REST' not in selected_pair and simultaneous:

            class_1 = np.linalg.norm(self.mu[self.classes.index(selected_pair[0])] -  selected_projection)
            class_2 = np.linalg.norm(self.mu[self.classes.index(selected_pair[1])] -  selected_projection)
            selected_pair = list(selected_pair)
            if class_1 < axis_len/4:
                selected_pair[1] = 'REST'
            elif class_2 < axis_len/4:
                selected_pair[0] = 'REST'

        output = []
        for i in selected_pair:
            output.append(self.classes.index(i))
        if not simultaneous:
            output = output[0]
        else:
            simultaneous_output = [0, 0]

            for i, val in enumerate(output):
                if self.classes[val] in self.grips:
                    simultaneous_output[0] = val
                if self.classes[val] in self.wrist:
                    simultaneous_output[1] = val
            
            output = simultaneous_output 

        self.projection = axes_projection 

        return output, [velocity, obj_velocity] #axes_projection 

if __name__ == '__main__':

    from TrainingDataGeneration import TrainingDataGenerator
    import csv

    EMG_SCALING_FACTOR = 10000.0

    SUBJECT = 1
    offline_test = False

    PARAM_FILE = os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ), 
                 'data', 'Subject_%s' % SUBJECT, 'RESCU_Protocol_Subject_%s.pkl' % SUBJECT  )

    with open( PARAM_FILE, 'rb' ) as pkl:
        Parameters = pickle.load( pkl )# compute training features and labels

    tdg = TrainingDataGenerator()
    
    Xtrain, ytrain = tdg.emg_pre_extracted( CLASSES = Parameters['Calibration']['CUE_LIST'],
                                            PREVIOUS_FILE = '',
                                            practice_session = 0,
                                            CARRY_UPDATES = Parameters['Misc.']['CARRY_UPDATES'],
                                            total_training_samples = Parameters['Classification']['TOTAL_TRAINING_SAMPLES'],
                                            onset_threshold_enable = Parameters['Classification']['ONSET_THRESHOLD'],
                                            onset_scaler = Parameters['Calibration']['CALIB_ONSET_SCALAR'],
                                            gain = Parameters['General']['ELECTRODE_GAIN'],
                                            fft_length = Parameters['General']['FFT_LENGTH'],
                                            EMG_SCALING_FACTOR = EMG_SCALING_FACTOR,
                                            emg_window_size = Parameters['General']['WINDOW_SIZE'],
                                            emg_window_step = Parameters['General']['WINDOW_STEP'],
                                            CALIBRATION = Parameters['Calibration']['CALIB_METHOD'],
                                            FL = False,
                                            subject = SUBJECT)

    '''PARAM_FILE = os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ), 
                 'data', 'Subject_%s' % SUBJECT, 'calibration_data', 'Xtrain.csv' )

    with open(PARAM_FILE, 'w', newline='') as f:
        # create the csv writer
        writer = csv.writer(f, delimiter=' ')

        # write a row to the csv file
        writer.writerows(Xtrain)

    PARAM_FILE = os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ), 
                 'data', 'Subject_%s' % SUBJECT, 'calibration_data', 'ytrain.csv' )

    with open(PARAM_FILE, 'w', newline='') as f:
        # create the csv writer
        writer = csv.writer(f, delimiter=' ')

        # write a row to the csv file
        writer.writerow(ytrain)'''

    Classifier = SpatialClassifier()
    Classifier.train(Xtrain = Xtrain, ytrain = ytrain, classes = Parameters['Calibration']['CUE_LIST'],perc_lo = [0 for i in range(len(Parameters['Calibration']['CUE_LIST']))] , perc_hi = [1 for i in range(len(Parameters['Calibration']['CUE_LIST']))], threshold_ratio = 0.3 )
    
    residuals = np.zeros(len(Parameters['Calibration']['CUE_LIST']))

    for i in Xtrain:
        residuals[Classifier.simultaneous_residuals(i, simultaneous = True)[0]] += 1
    print(residuals)

    Classifier.SeparabilityScore()

    