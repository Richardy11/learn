import copy
import math
import os
import pickle
import random
import sys
import time
from collections import deque
import warnings

import matplotlib
import numpy as np
import psutil

matplotlib.use( 'QT5Agg')

from scipy.io import loadmat

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Init import InitGUI
from local_addresses import address
from Reviewer import ReviewerPlot
from SepRep import SepRepPlot

from Bebionic3 import Bebionic3
from DownSampler import DownSampler
from FeatureSmoothing import Smoothing
from FourierTransformFilter import FourierTransformFilter
from MyoArmband import MyoArmband
from OnsetThreshold import Onset
from OutputFilter import OutputFilter
from ProportionalControl import Proportional_Control
from RealTimeControlPlot import RealTimeControlPlot
from RealTimeDataPlot import RealTimeDataPlot
from RealTimeGlidePlot import RealTimeGlidePlot
from RunClassifier import RunClassifier
from Segmentation import Segmentation
from SenseController import SenseController
from TimeDomainFilter import TimeDomainFilter
from TrainingDataGeneration import TrainingDataGenerator
from UpdateClassifier import Update
from UpdateGUI import UpdateGUI
from VelocityRampFilter import VelocityRampFilter
from MyoTrain import MyoTrain
from FittsLawTask import FittsLawTask, FittsLawTaskResults

EMG_SCALING_FACTOR = 10000.0

SUBJECT = 1
RUN_MODE = 0
SESSION = 1 # NOTE: This variable may need to be looked at later
CONNECT = True

if len(sys.argv) > 1:
    #print( '\nSUBJECT:', str(sys.argv[1]) )
    SUBJECT = sys.argv[1]
    if len(sys.argv) > 2:
        if sys.argv[2] == 'Calib':
            CONNECT = False
        else:
            RUN_MODE = int(sys.argv[2])

FILE_DIRECTORY = os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ), 'data', 'Subject_%s' % SUBJECT ) # if Rescu.py is in main folder

class RESCU:
    def __init__( self, subject = 1, run_mode = 0, source = False ):

        self.subject = subject

        self.run_FLT = run_mode == 1
        self.run_SR = run_mode == 2

        self.local_source_info = address()

        self.SAVE_FILE = os.path.join( FILE_DIRECTORY, 'Subject_%s_Session_%s.pkl' % ( self.subject, SESSION ) )
        self.PREVIOUS_FILE = os.path.join( FILE_DIRECTORY, 'Subject_%s_Session_%s.pkl' % ( self.subject, SESSION - 1 ) )
        self.EXPERIMENTAL_DATA_FILE = os.path.join( FILE_DIRECTORY, 'Chris_Data.pkl' )

        self.init = InitGUI( online = True, preload = self.subject )
        self.Parameters = self.init.load_subject_parameters(self.subject)

        if self.Parameters['General']['SOURCE'] == 'Myo Band':
            self.window_size = self.Parameters['General']['WINDOW_SIZE']//5
        else:
            self.window_size = self.Parameters['General']['WINDOW_SIZE']

        # download training data
        print( 'Importing training data...', end = '', flush = True)
        # check if longterm updating is active
        if self.Parameters['Misc.']['CARRY_UPDATES']:
            # check if updates have occured in previous sessions and load latest set
            self.practice_session = 1
            while True:
                self.PREVIOUS_FILE = os.path.join( FILE_DIRECTORY, 'Subject_%s_Practice_%s.pkl' % ( self.subject, self.practice_session ) )
                if os.path.isfile(self.PREVIOUS_FILE):
                    self.practice_session += 1
                else:
                    self.SAVE_FILE = os.path.join( FILE_DIRECTORY, 'Subject_%s_Practice_%s.pkl' % ( self.subject, self.practice_session ) )
                    self.PREVIOUS_FILE = os.path.join( FILE_DIRECTORY, 'Subject_%s_Practice_%s.pkl' % ( self.subject, self.practice_session - 1 ) )
                    if self.practice_session > 1:
                        print( 'Importing previous session data...', end = '', flush = True)
                    break

        CALIBRATION_DATA = os.path.join( FILE_DIRECTORY, 'calibration_data' , 'calibrate_subject_%d.mat' % int(self.subject) )
        # load original training set
        self.training_data = loadmat( CALIBRATION_DATA )['calibrate'][0][0]
        print( 'Done!' )

        # create feature extracting filters
        print( 'Creating EMG feature filters...', end = '', flush = True )
        self.td5 = TimeDomainFilter()
        if self.Parameters['General']['FFT_LENGTH'] in ['Custom Adapt']:
            self.fft = FourierTransformFilter( fftlen = 64 )
        else:
            self.fft = FourierTransformFilter( fftlen = int(self.Parameters['General']['FFT_LENGTH']) )
        print( 'Done!' )

        # create downsampler
        print( 'Downsampling training data...', end = '', flush = True )
        self.dsample = DownSampler()
        print( 'Done!' )

        if self.run_FLT:
            self.current_cue_list = self.Parameters['Fitts Law']['CUE_LIST']
        else:
            self.current_cue_list = self.Parameters['Calibration']['CUE_LIST']

        if self.Parameters['Misc.']['EXPERIMENT'] == False:
            self.current_method = "Method_1"
        else:
            self.current_method = "Method_%s" % self.Parameters['Fitts Law']['Method']
            

        self.tdg = TrainingDataGenerator()
        # compute training features and labels
        if self.practice_session > 1 and self.Parameters['Misc.']['CARRY_UPDATES']:
            output = self.tdg.emg_extract_onset( CLASSES = self.current_cue_list, 
                                            FL = self.run_FLT, 
                                            subject = self.subject, 
                                            practice_session = self.practice_session, 
                                            PREVIOUS_FILE = self.PREVIOUS_FILE, 
                                            CARRY_UPDATES = self.Parameters['Misc.']['CARRY_UPDATES'],
                                            method = self.current_method)

            self.Xtrain = output[0]
            self.ytrain = output[1]
            self.Xtrain_original = output[2]
            self.ytrain_original = output[3]

        else:
            output = self.tdg.emg_pre_extracted(   CLASSES = self.current_cue_list,
                                                                PREVIOUS_FILE = self.PREVIOUS_FILE,
                                                                practice_session = self.practice_session,
                                                                CARRY_UPDATES = self.Parameters['Misc.']['CARRY_UPDATES'],
                                                                total_training_samples = self.Parameters['Classification']['CLUSTER_TRAINING_SAMPLES'],
                                                                onset_threshold_enable = self.Parameters['Classification']['ONSET_THRESHOLD'],
                                                                onset_scaler = self.Parameters['Calibration']['CALIB_ONSET_SCALAR'],
                                                                gain = self.Parameters['General']['ELECTRODE_GAIN'],
                                                                fft_length = self.Parameters['General']['FFT_LENGTH'],
                                                                EMG_SCALING_FACTOR = EMG_SCALING_FACTOR,
                                                                emg_window_size = self.window_size,
                                                                emg_window_step = self.Parameters['General']['WINDOW_STEP'],
                                                                CALIBRATION = self.Parameters['Calibration']['CALIB_METHOD'],
                                                                FL = self.run_FLT,
                                                                subject = self.subject,
                                                                method = self.current_method)
            self.Xtrain = output[0]
            self.ytrain = output[1]
            self.Xtrain_original = copy.deepcopy(self.Xtrain)
            self.ytrain_original = copy.deepcopy(self.ytrain)

        if max(self.ytrain) != len(self.current_cue_list)-1:
            print('Training data set is not up to date, number of labels must be equal to number of movement types. This can also happen if updtes from previous sessions exist, but are out of sync with current list of movements')
        

        self.method_order = np.array([1, 2])
        random.shuffle(self.method_order)

        self.storage = { 'cue_finished' :    [0],
                    'Current_Xtrain':   self.Xtrain,
                    'Current_ytrain':   self.ytrain,
                    'emg' :             [],
                    'features' :        [],
                    'predictions' :     [],
                    'filtered_output' : [],
                    'velocity' :        [],
                    'proportional' :    [],
                    'updates':          [],
                    'Original_Xtrain' : self.Xtrain_original,
                    'Original_ytrain' : self.ytrain_original,
                    'Xtrain' :          [self.Xtrain],
                    'ytrain' :          [self.ytrain],
                    'Segments' :        [],
                    'Cache' :           [],
                    'CR' :              [],
                    'segment_onset' :   [],
                    'segment_offset' :  [] }

        self.Protocol_Data = {  'Method_Order': self.method_order,
                                'Method_1':     copy.deepcopy(self.storage),
                                'Method_2':     copy.deepcopy(self.storage)}

        # initiate onset threshold
        
        print( 'Create onset threshold training data...', end = '', flush = True )
        if not self.Parameters['Classification']['ONSET_THRESHOLD']:
            self.onset_threshold_calculation = Onset( onset_scalar = 1.1 )
        else:
            self.onset_threshold_calculation = Onset( onset_scalar = self.Parameters['Classification']['ONSET_SCALAR'] )

        self.onset_threshold = self.onset_threshold_calculation.onset_threshold(  Xtrain = self.Xtrain, 
                                                                        ytrain = self.ytrain, 
                                                                        gain = self.Parameters['General']['ELECTRODE_GAIN'],
                                                                        classifier = self.Parameters['Classification']['CLASSIFIER'])  
        print( 'Done!' )

        print( 'Create Smoothor...', end = '', flush = True )
        self.smoothor = Smoothing( self.Parameters )
        print( 'Done!' )

        self.generate_sensitivities()
        self.generate_proportionals()

        # train classifier
        print( 'Training classifier...', end = '', flush = True )
        
        self.run_classifier = RunClassifier(    class_subdict_size = self.Parameters['Classification']['CLUSTER_TRAINING_SAMPLES'], 
                                                Xtrain = self.Xtrain_original, 
                                                ytrain = self.ytrain_original, 
                                                classes = self.current_cue_list,
                                                perc_lo = [i for i in self.PERCENTAGES_LOW], 
                                                perc_hi = [i for i in self.PERCENTAGES_HIGH],
                                                threshold_ratio = self.Parameters['Classification']['THRESHOLD_RATIO'] )
        print( 'Done!' )

        print( 'Creating proportional control filters...', end = '', flush = True )
    
        self.spatial_prop = copy.deepcopy(self.run_classifier)

        if self.practice_session > 1 and self.Parameters['Misc.']['CARRY_UPDATES']:
            self.run_classifier.init_classifiers(  Xtrain = self.Xtrain,
                                                    ytrain = self.ytrain, 
                                                    classes = self.current_cue_list,
                                                    perc_lo = [i for i in self.PERCENTAGES_LOW], 
                                                    perc_hi = [i for i in self.PERCENTAGES_HIGH],
                                                    threshold_ratio = self.Parameters['Classification']['THRESHOLD_RATIO'],
                                                    recalculate_RLDA = self.Parameters['Classification']['REGENERATE_CLASSIFIER']
                                                )
            self.spatial_prop.init_classifiers(  Xtrain = self.Xtrain,
                                                    ytrain = self.ytrain, 
                                                    classes = self.current_cue_list,
                                                    perc_lo = [i for i in self.PERCENTAGES_LOW], 
                                                    perc_hi = [i for i in self.PERCENTAGES_HIGH],
                                                    threshold_ratio = self.Parameters['Classification']['THRESHOLD_RATIO'],
                                                    recalculate_RLDA = self.Parameters['Classification']['REGENERATE_CLASSIFIER']
                                                )



        self.prop = Proportional_Control(   X = self.Xtrain.T, 
                                            L = self.ytrain, 
                                            onset_threshold = self.onset_threshold_calculation.onset_threshold(     Xtrain = self.Xtrain, 
                                                                                                                    ytrain = self.ytrain, 
                                                                                                                    gain = self.Parameters['General']['ELECTRODE_GAIN'],
                                                                                                                    classifier = ''),
                                            classes = self.current_cue_list, 
                                            proportional_low =  self.PROPORTIONAL_LOW, 
                                            proportional_high = self.PROPORTIONAL_HIGH )
        print( 'Done!' )

        
        # create output filters
        print( 'Creating output filter...', end = '', flush = True )
        self.primary = OutputFilter(    sensitivities = self.SENSITIVITIES, 
                                        continuous = self.Parameters['Output Filtering'], 
                                        classes = self.current_cue_list, 
                                        uniform_size = self.Parameters['Output Filtering']['UNIFORM_FILTER_SIZE'] )

        self.secondary = OutputFilter(  sensitivities = self.SENSITIVITIES, 
                                        continuous = self.Parameters['Output Filtering'], 
                                        classes = self.current_cue_list, 
                                        uniform_size = self.Parameters['Output Filtering']['UNIFORM_FILTER_SIZE'] )

        #TODO Make sure enable check happens in realtime
        self.vramp = VelocityRampFilter( n_classes = len( self.current_cue_list ),
                                         increment = self.Parameters['Proportional Control']['VELOCITY_RAMP_INCREMENT'],
                                         decrement = self.Parameters['Proportional Control']['VELOCITY_RAMP_DECREMENT'],
                                         max_bin_size = self.Parameters['Proportional Control']['VELOCITY_RAMP_SIZE'] )
        print( 'Done!' )

        if self.Parameters['Signal Outputting']['SHOW_MYOTRAIN']:
            # create MyoTrain interface
            if self.checkIfProcessRunning('MyoTrain'):
                print('A MyoTrain is currently running')
                print( 'Creating MyoTrain UDP connection...', end = '', flush = True )
                self.myotrain = MyoTrain( path = os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ), 'myotrain', 'MyoTrain_R.exe' ), running = True )
            
            else:
                print( 'Creating MyoTrain UDP connection...', end = '', flush = True )
                self.myotrain = MyoTrain( path = os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ), 'myotrain', 'MyoTrain_R.exe' ), running = False )
                time.sleep( 5 )

            print( 'Done!')

        # create Sense interface
        print( 'Creating device interface...', end = '', flush = True )
        if type(source) == bool:
            if self.Parameters['General']['SOURCE'] == 'Sense':
                self.source = SenseController( name = self.local_source_info.sense_name,
                                            mac = self.local_source_info.sense_mac,
                                            gain = self.Parameters['General']['ELECTRODE_GAIN'],
                                            num_electrodes = self.Parameters['General']['DEVICE_ELECTRODE_COUNT'],
                                            srate = self.Parameters['General']['SAMPLING_RATE'] )
            elif self.Parameters['General']['SOURCE'] == 'Myo Band':
                self.source = MyoArmband( name = self.local_source_info.myo_name, mac = self.local_source_info.myo_mac )
            else:
                raise RuntimeError( 'Unsupported EMG interface provided:', self.source )
        else:
            self.source = source
        print( 'Done!' )

        if self.Parameters['Signal Outputting']['PROSTHESIS']:      
            if self.Parameters['Signal Outputting']['TERMINAL_DEVICE'] == 'BeBionic':

                print( 'Creating bebionic control...', end = '', flush = True )

                self.moves = [ 'rest', 'open', 'close', 'pronate', 'supinate',
                        'elbow_flex', 'elbow_extend', 'tripod', 'active_index', 'key_lateral',
                        'rest', 'open', 'pronate', 'supinate', 'elbow_flex', 'elbow_extend' ]
                self.prosthesis = Bebionic3( self.local_source_info.sense_name, False )

                print( 'Done!')
            
            elif self.Parameters['Signal Outputting']['TERMINAL_DEVICE'] == 'ProETD':

                print( 'Creating ProETD control...', end = '', flush = True )

                self.moves = [ 'rest', 'open', 'close', 'pronate', 'supinate',
                        'elbow_flex', 'elbow_extend' ]
                self.prosthesis = Bebionic3( self.local_source_info.sense_name, False )

                print( 'Done!')


        # Create update GUI
        if self.Parameters['Classification']['RESCU']:

            # create segmentor
            print( 'Creating segmentor...', end = '', flush = True )
            self.seg_update_k = int( self.Parameters['Classification']['CLUSTER_TRAINING_SAMPLES'] * ( self.Parameters['Classification']['UPDATE_K'] / 100 ) )
            self.segmentor = Segmentation(onset_threshold = self.onset_threshold_calculation.onset_threshold(   Xtrain = self.Xtrain, 
                                                                                                                ytrain = self.ytrain, 
                                                                                                                gain = self.Parameters['General']['ELECTRODE_GAIN'],
                                                                                                                classifier = ''),
                                            overwrite = self.seg_update_k,
                                            plot = self.Parameters['Classification']['SEGMENTATION_PLOT'],
                                            plot_max = np.amax(np.mean(self.Xtrain[:,:8],1))*1.5)
            print( 'Done!' )
            
            # create updater GUI
            print( 'Creating updater + GUI...', end = '', flush = True )
            self.update = Update()
            self.gui = UpdateGUI( classes = tuple( self.current_cue_list ), all_classes = self.Parameters['Calibration']['ALL_CLASSES'], n_segments = 5 )
            print( 'Done!' )

        # show ploting interfaces
        if self.Parameters['Signal Outputting']['SHOW_MAV']:
            print( 'Creating real-time data plot...', end = '', flush = True )
            if self.Parameters['Classification']['CLASSIFIER'] in ['Spatial', 'Simultaneous']:
                channel_max = self.run_classifier.SpatialClassifier.channel_max
            else:            
                if self.Parameters['General']['SOURCE'] == 'Myo Band':
                    channel_max = 0.3/50
                elif self.Parameters['General']['SOURCE'] == 'Sense':
                    channel_max = 0.3
            
            self.mav_plot = RealTimeDataPlot(   num_channels = self.Parameters['General']['DEVICE_ELECTRODE_COUNT'],
                                                channel_min = 0,
                                                channel_max = channel_max,
                                                sampling_rate = self.Parameters['General']['SAMPLING_RATE'] / self.Parameters['General']['WINDOW_STEP'],
                                                buffer_time = 5,
                                                title = "MAV Signals",
                                                classifier =  self.Parameters['Classification']['CLASSIFIER'],
                                                source = self.Parameters['General']['SOURCE'])

            if self.Parameters['Classification']['ONSET_THRESHOLD'] is False:
                self.mav_plot.set_threshold( 0 )
            else:
                self.mav_plot.set_threshold( self.onset_threshold_calculation.onset_threshold(  Xtrain = self.Xtrain, 
                                                                                                ytrain = self.ytrain, 
                                                                                                gain = self.Parameters['General']['ELECTRODE_GAIN'],
                                                                                                classifier = '') )

            if self.Parameters['Classification']['CLASSIFIER'] in ['Spatial', 'Simultaneous']:
                    self.mav_plot.set_threshold( self.run_classifier.SpatialClassifier.threshold_radius_norm )
            print( 'Done!' )
        
        if self.Parameters['Signal Outputting']['SHOW_CONTROL']:
            print( 'Creating real-time control plot...', end = '', flush = True )
            self.ctrl_plot = RealTimeControlPlot(   self.current_cue_list,
                                                    num_channels = 2, 
                                                    sampling_rate = self.Parameters['General']['SAMPLING_RATE'] / self.Parameters['General']['WINDOW_STEP'], 
                                                    buffer_time = 5, 
                                                    title = "Control Signal (filtered and raw)" )
            print( 'Done!' )

        print( 'Creating real-time velocity plot...', end = '', flush = True )
        self.set_velocity_thresholds() 

        if self.Parameters['Signal Outputting']['SHOW_VELOCITY']:        
            self.glide = RealTimeGlidePlot( classes = self.current_cue_list, 
                                            all_classes = self.Parameters['Calibration']['ALL_CLASSES'], 
                                            prop_lo = [i for i in self.PROPORTIONAL_LOW], 
                                            prop_hi = [i for i in self.PROPORTIONAL_HIGH],
                                            perc_lo = [i for i in self.PERCENTAGES_LOW], 
                                            perc_hi = [i for i in self.PERCENTAGES_HIGH],  
                                            bar_min = self.onset_threshold_calculation.onset_threshold( Xtrain = self.Xtrain, 
                                                                                                        ytrain = self.ytrain, 
                                                                                                        gain = self.Parameters['General']['ELECTRODE_GAIN'],
                                                                                                        classifier = ''), 
                                            bar_max = np.amax(np.mean(self.Xtrain[:,:8],1)),
                                            method = self.Parameters['Proportional Control']['PROPORTIONAL_METHOD'],
                                            spatial_velocity_dicts = self.spatial_prop.SpatialClassifier.proportional_range)
        print( 'Done!' )

        if self.Parameters['Signal Outputting']['SHOW_REVIEWER']:
            print( 'Creating real-time SepRep plot...', end = '', flush = True )
            #TODO
            self.reviewer = SepRepPlot( SUBJECT = self.subject, 
                                        CLASSES = self.current_cue_list, 
                                        Xtrain = self.Xtrain, 
                                        ytrain = self.ytrain,
                                        CLASSIFIER = self.Parameters['Classification']['CLASSIFIER'], 
                                        ALL_CLASSES = self.Parameters['Calibration']['ALL_CLASSES'],
                                        CLUSTER_TRAINING_SAMPLES = self.Parameters['Classification']['CLUSTER_TRAINING_SAMPLES'],
                                        realtime = True,
                                        perc_lo = [i for i in self.PERCENTAGES_LOW], 
                                        perc_hi = [i for i in self.PERCENTAGES_HIGH],
                                        threshold_ratio = self.run_classifier.SpatialClassifier.onset_threshold, #self.Parameters['Classification']['THRESHOLD_RATIO'],
                                        recalculate_RLDA = self.Parameters['Classification']['REGENERATE_CLASSIFIER'],
                                        proj = self.run_classifier.SpatialClassifier.proj,
                                        stdscaler = self.run_classifier.SpatialClassifier.stdscaler,
                                        classifier_obj = self.run_classifier)

            
            print( 'Done!' )

        if self.run_FLT:
            self.FLT = FittsLawTask()

        self.run_main_process()

    def reinitialize_parameters(self, changed):
        
        if changed == ['Classification', 'REGENERATE_CLASSIFIER']:
            
            if self.Parameters['Classification']['REGENERATE_CLASSIFIER']:
                Xtrain_to_use = self.storage['Current_Xtrain']
                ytrain_to_use = self.storage['Current_ytrain']
            else:
                Xtrain_to_use = self.storage['Original_Xtrain']
                ytrain_to_use = self.storage['Original_ytrain']

            # train classifier
            print( 'Training classifier...', end = '', flush = True )
            
            self.run_classifier = RunClassifier(    class_subdict_size = self.Parameters['Classification']['CLUSTER_TRAINING_SAMPLES'], 
                                                    Xtrain = Xtrain_to_use, 
                                                    ytrain = ytrain_to_use, 
                                                    classes = self.current_cue_list,
                                                    perc_lo = [i for i in self.PERCENTAGES_LOW], 
                                                    perc_hi = [i for i in self.PERCENTAGES_HIGH],
                                                    threshold_ratio = self.Parameters['Classification']['THRESHOLD_RATIO'] )
        
            self.spatial_prop = copy.deepcopy(self.run_classifier)

            self.run_classifier.init_classifiers(  Xtrain = self.Xtrain,
                                                        ytrain = self.ytrain, 
                                                        classes = self.current_cue_list,
                                                        perc_lo = [i for i in self.PERCENTAGES_LOW], 
                                                        perc_hi = [i for i in self.PERCENTAGES_HIGH],
                                                        threshold_ratio = self.Parameters['Classification']['THRESHOLD_RATIO'],
                                                        recalculate_RLDA = self.Parameters['Classification']['REGENERATE_CLASSIFIER']
                                                    )
            self.spatial_prop.init_classifiers(  Xtrain = self.Xtrain,
                                                    ytrain = self.ytrain, 
                                                    classes = self.current_cue_list,
                                                    perc_lo = [i for i in self.PERCENTAGES_LOW], 
                                                    perc_hi = [i for i in self.PERCENTAGES_HIGH],
                                                    threshold_ratio = self.Parameters['Classification']['THRESHOLD_RATIO'],
                                                    recalculate_RLDA = self.Parameters['Classification']['REGENERATE_CLASSIFIER'])
            
            if self.Parameters['Signal Outputting']['SHOW_REVIEWER']:

                try:
                    self.reviewer.close()
                except: pass
                  
                self.reviewer = SepRepPlot( SUBJECT = self.subject, 
                                            CLASSES = self.current_cue_list, 
                                            Xtrain = self.Xtrain, 
                                            ytrain = self.ytrain,
                                            CLASSIFIER = self.Parameters['Classification']['CLASSIFIER'], 
                                            ALL_CLASSES = self.Parameters['Calibration']['ALL_CLASSES'],
                                            CLUSTER_TRAINING_SAMPLES = self.Parameters['Classification']['CLUSTER_TRAINING_SAMPLES'],
                                            realtime = True,
                                            perc_lo = [i for i in self.PERCENTAGES_LOW], 
                                            perc_hi = [i for i in self.PERCENTAGES_HIGH],
                                            threshold_ratio = self.run_classifier.SpatialClassifier.onset_threshold, #self.Parameters['Classification']['THRESHOLD_RATIO'],
                                            recalculate_RLDA = self.Parameters['Classification']['REGENERATE_CLASSIFIER'],
                                            proj = self.run_classifier.SpatialClassifier.proj,
                                            stdscaler = self.run_classifier.SpatialClassifier.stdscaler,
                                            classifier_obj = self.run_classifier)

        
        if changed == ['General', 'FL']:
            self.reinitialize_FLT()
                    
        if changed == ['Classification', 'UPDATE_K']:
                    
            self.seg_update_k = int( self.Parameters['Classification']['CLUSTER_TRAINING_SAMPLES'] * ( self.Parameters['Classification']['UPDATE_K'] / 100 ) )

        if changed[0] == 'Output Filtering':
            self.generate_sensitivities()
            self.primary = OutputFilter( sensitivities = self.SENSITIVITIES, continuous = self.Parameters['Output Filtering'], classes = self.current_cue_list, uniform_size = self.Parameters['Output Filtering']['UNIFORM_FILTER_SIZE'] )
            #TODO fix analyzer
            #self.segmentor.set_analyzer_sensitivities(self.SENSITIVITIES)

        if changed == ['Proportional Control', 'PROPORTIONAL_METHOD']:
            
            self.set_velocity_thresholds()
            self.generate_proportionals() 

            try:
                self.glide.close()
            except:
                pass      

            if self.Parameters['Signal Outputting']['SHOW_VELOCITY']:

                self.glide = RealTimeGlidePlot( classes = self.current_cue_list, 
                                                all_classes = self.Parameters['Calibration']['ALL_CLASSES'], 
                                                prop_lo = [i for i in self.PROPORTIONAL_LOW], 
                                                prop_hi = [i for i in self.PROPORTIONAL_HIGH],
                                                perc_lo = [i for i in self.PERCENTAGES_LOW], 
                                                perc_hi = [i for i in self.PERCENTAGES_HIGH],  
                                                bar_min = self.onset_threshold_calculation.onset_threshold( Xtrain = self.Xtrain, 
                                                                                                        ytrain = self.ytrain, 
                                                                                                        gain = self.Parameters['General']['ELECTRODE_GAIN'],
                                                                                                        classifier = ''), 
                                                bar_max = np.amax(np.mean(self.Xtrain[:,:8],1)),
                                                method = self.Parameters['Proportional Control']['PROPORTIONAL_METHOD'],
                                                spatial_velocity_dicts = self.spatial_prop.SpatialClassifier.proportional_range)
            else:
                self.glide.close()

        if changed[0] == 'Proportional Control':
            
            self.generate_proportionals()
            if self.Parameters['Proportional Control']['PROPORTIONAL_METHOD'] == 'Spatial':
                self.run_classifier.SpatialClassifier.proportional_levels( self.PERCENTAGES_LOW, self.PERCENTAGES_HIGH )
                self.spatial_prop.SpatialClassifier.proportional_levels( self.PERCENTAGES_LOW, self.PERCENTAGES_HIGH )
            elif self.Parameters['Proportional Control']['PROPORTIONAL_METHOD'] == 'Amplitude':
                self.prop.set_limits(self.PROPORTIONAL_LOW, self.PROPORTIONAL_HIGH)
                    
        if changed == ['Classification', 'RESCU']:
            # create segmentor
            if self.Parameters['Classification']['RESCU']:

                self.seg_update_k = int( self.Parameters['Classification']['CLUSTER_TRAINING_SAMPLES'] * ( self.Parameters['Classification']['UPDATE_K'] / 100 ) )
                self.segmentor = Segmentation(onset_threshold = self.onset_threshold_calculation.onset_threshold(   Xtrain = self.Xtrain, 
                                                                                                                    ytrain = self.ytrain, 
                                                                                                                    gain = self.Parameters['General']['ELECTRODE_GAIN'],
                                                                                                                    classifier = ''),
                                            overwrite = self.seg_update_k,
                                            plot = self.Parameters['Classification']['SEGMENTATION_PLOT'],
                                            plot_max = np.amax(np.mean(self.Xtrain[:,:8],1))*1.5)

                self.update = Update()
                try:
                    self.gui.close()
                except: pass
                self.gui = UpdateGUI( classes = tuple( self.current_cue_list ), all_classes = self.Parameters['Calibration']['ALL_CLASSES'], n_segments = 5 )
                
            else:
                if self.Parameters['Classification']['SEGMENTATION_PLOT']:
                    self.segmentor.close()
                self.gui.close()

        if changed == ['Classification', 'SEGMENTATION_PLOT']:
            # create segmentor
            if self.Parameters['Classification']['SEGMENTATION_PLOT']:

                self.seg_update_k = int( self.Parameters['Classification']['CLUSTER_TRAINING_SAMPLES'] * ( self.Parameters['Classification']['UPDATE_K'] / 100 ) )
                self.segmentor = Segmentation(onset_threshold = self.onset_threshold_calculation.onset_threshold(   Xtrain = self.Xtrain, 
                                                                                                                    ytrain = self.ytrain, 
                                                                                                                    gain = self.Parameters['General']['ELECTRODE_GAIN'],
                                                                                                                    classifier = ''),
                                            overwrite = self.seg_update_k,
                                            plot = self.Parameters['Classification']['SEGMENTATION_PLOT'],
                                            plot_max = np.amax(np.mean(self.Xtrain[:,:8],1))*1.5)
                
            else:
                self.segmentor.close()

        if changed == ['Classification', 'UPDATE_K']:
            # create segmentor
            self.seg_update_k = int( self.Parameters['Classification']['CLUSTER_TRAINING_SAMPLES'] * ( self.Parameters['Classification']['UPDATE_K'] / 100 ) )
            self.segmentor = Segmentation(onset_threshold = self.onset_threshold_calculation.onset_threshold(   Xtrain = self.Xtrain, 
                                                                                                                ytrain = self.ytrain, 
                                                                                                                gain = self.Parameters['General']['ELECTRODE_GAIN'],
                                                                                                                classifier = ''),
                                        overwrite = self.seg_update_k,
                                        plot = self.Parameters['Classification']['SEGMENTATION_PLOT'],
                                        plot_max = np.amax(np.mean(self.Xtrain[:,:8],1))*1.5)


        if changed[0] == 'Classification' and (changed[1] in ['FEATURE_FILTER', 'FILTER_LENGTH', 'EXPONENTIAL_FILTER_SCALE', 'BEZIER_PROJECTION_RATIO']):
            self.smoothor.set_params( filter = self.Parameters['Classification']['FEATURE_FILTER'], type = changed[1], value = self.Parameters['Classification'][changed[1]] )

        if changed == ['Signal Outputting', 'PROSTHESIS']:

            if not hasattr(self, 'prosthesis'):

                if self.Parameters['Signal Outputting']['TERMINAL_DEVICE'] == 'BeBionic':

                    print( 'Creating bebionic control...', end = '', flush = True )

                    self.moves = [ 'rest', 'open', 'close', 'pronate', 'supinate',
                            'elbow_flex', 'elbow_extend', 'tripod', 'active_index', 'key_lateral',
                            'rest', 'open', 'pronate', 'supinate', 'elbow_flex', 'elbow_extend' ]
                    self.prosthesis = Bebionic3( self.local_source_info.sense_mac, False )

                    print( 'Done!')
                
                elif self.Parameters['Signal Outputting']['TERMINAL_DEVICE'] == 'ProETD':

                    print( 'Creating ProETD control...', end = '', flush = True )

                    self.moves = [ 'rest', 'open', 'close', 'pronate', 'supinate',
                            'elbow_flex', 'elbow_extend' ]
                    self.prosthesis = Bebionic3( self.local_source_info.sense_mac, False )

                    print( 'Done!')
        
        if changed == ['Signal Outputting', 'SHOW_MAV']:

            if self.Parameters['Signal Outputting']['SHOW_MAV']:
                
                if self.Parameters['Classification']['CLASSIFIER'] in ['Spatial', 'Simultaneous']:
                    channel_max = self.run_classifier.SpatialClassifier.channel_max
                else:            
                    if self.Parameters['General']['SOURCE'] == 'Myo Band':
                        channel_max = 0.3/50
                    elif self.Parameters['General']['SOURCE'] == 'Sense':
                        channel_max = 0.3
                
                self.mav_plot = RealTimeDataPlot(   num_channels = self.Parameters['General']['DEVICE_ELECTRODE_COUNT'],
                                                    channel_min = 0,
                                                    channel_max = channel_max,
                                                    sampling_rate = self.Parameters['General']['SAMPLING_RATE'] / self.Parameters['General']['WINDOW_STEP'],
                                                    buffer_time = 5,
                                                    title = "MAV Signals",
                                                    classifier =  self.Parameters['Classification']['CLASSIFIER'],
                                                    source = self.Parameters['General']['SOURCE'])

                if self.Parameters['Classification']['ONSET_THRESHOLD'] is False:
                    self.mav_plot.set_threshold( 0 )
                if self.Parameters['Classification']['CLASSIFIER'] in ['Spatial', 'Simultaneous']:
                    self.mav_plot.set_threshold( self.run_classifier.SpatialClassifier.threshold_radius_norm )

            else:
                
                self.mav_plot.close()

        if changed == ['Signal Outputting', 'SHOW_CONTROL']:

            if self.Parameters['Signal Outputting']['SHOW_CONTROL']:
                
                self.ctrl_plot = RealTimeControlPlot(   self.current_cue_list,
                                                        num_channels = 2, 
                                                        sampling_rate = self.Parameters['General']['SAMPLING_RATE'] / self.Parameters['General']['WINDOW_STEP'], 
                                                        buffer_time = 5, 
                                                        title = "Control Signal (filtered and raw)" )
                
            else:                
                self.ctrl_plot.close()

        if changed == ['Signal Outputting', 'SHOW_VELOCITY']:
            
            self.set_velocity_thresholds()
            self.generate_proportionals() 

            try:
                self.glide.close()
            except:
                pass      

            if self.Parameters['Signal Outputting']['SHOW_VELOCITY']:

                self.glide = RealTimeGlidePlot( classes = self.current_cue_list, 
                                                all_classes = self.Parameters['Calibration']['ALL_CLASSES'], 
                                                prop_lo = [i for i in self.PROPORTIONAL_LOW], 
                                                prop_hi = [i for i in self.PROPORTIONAL_HIGH],
                                                perc_lo = [i for i in self.PERCENTAGES_LOW], 
                                                perc_hi = [i for i in self.PERCENTAGES_HIGH],  
                                                bar_min = self.onset_threshold_calculation.onset_threshold( Xtrain = self.Xtrain, 
                                                                                                        ytrain = self.ytrain, 
                                                                                                        gain = self.Parameters['General']['ELECTRODE_GAIN'],
                                                                                                        classifier = ''), 
                                                bar_max = np.amax(np.mean(self.Xtrain[:,:8],1)),
                                                method = self.Parameters['Proportional Control']['PROPORTIONAL_METHOD'],
                                                spatial_velocity_dicts = self.spatial_prop.SpatialClassifier.proportional_range)
            else:
                self.glide.close()

        if changed == ['Signal Outputting', 'SHOW_MYOTRAIN']:

            if self.Parameters['Signal Outputting']['SHOW_MYOTRAIN']:
                # create MyoTrain interface
                if self.checkIfProcessRunning('MyoTrain'):
                    
                    print('A MyoTrain is currently running')
                    print( 'Creating MyoTrain UDP connection...', end = '', flush = True )
                    self.myotrain = MyoTrain( path = os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ), 'myotrain', 'MyoTrain_R.exe' ), running = True )
                
                else:
                    print( 'Creating MyoTrain UDP connection...', end = '', flush = True )
                    self.myotrain = MyoTrain( path = os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ), 'myotrain', 'MyoTrain_R.exe' ), running = False )
                    time.sleep( 5 )
            
            else:
                self.myotrain.__del__()

                print( 'Done!')

        if changed == ['Signal Outputting', 'SHOW_REVIEWER']:

            if self.Parameters['Signal Outputting']['SHOW_REVIEWER']:
                  
                self.reviewer = SepRepPlot( SUBJECT = self.subject, 
                                            CLASSES = self.current_cue_list, 
                                            Xtrain = self.Xtrain, 
                                            ytrain = self.ytrain,
                                            CLASSIFIER = self.Parameters['Classification']['CLASSIFIER'], 
                                            ALL_CLASSES = self.Parameters['Calibration']['ALL_CLASSES'],
                                            CLUSTER_TRAINING_SAMPLES = self.Parameters['Classification']['CLUSTER_TRAINING_SAMPLES'],
                                            realtime = True,
                                            perc_lo = [i for i in self.PERCENTAGES_LOW], 
                                            perc_hi = [i for i in self.PERCENTAGES_HIGH],
                                            threshold_ratio = self.run_classifier.SpatialClassifier.onset_threshold, #self.Parameters['Classification']['THRESHOLD_RATIO'],
                                            recalculate_RLDA = self.Parameters['Classification']['REGENERATE_CLASSIFIER'],
                                            proj = self.run_classifier.SpatialClassifier.proj,
                                            stdscaler = self.run_classifier.SpatialClassifier.stdscaler,
                                            classifier_obj = self.run_classifier)
            else:
                
                self.reviewer.close()

        if changed[0] == 'Classification':
            if not self.Parameters['Classification']['ONSET_THRESHOLD']:
                self.onset_threshold_calculation = Onset( onset_scalar = 1.1 )
            else:
                self.onset_threshold_calculation = Onset( onset_scalar = self.Parameters['Classification']['ONSET_SCALAR'] )

            self.onset_threshold = self.onset_threshold_calculation.onset_threshold(  Xtrain = self.Xtrain, 
                                                                                        ytrain = self.ytrain, 
                                                                                        gain = self.Parameters['General']['ELECTRODE_GAIN'],
                                                                                        classifier = '',
                                                                                        threshold_ratio = self.Parameters['Classification']['THRESHOLD_RATIO'])  
            
            self.run_classifier.SpatialClassifier.onset_threshold_calc(threshold_ratio = self.Parameters['Classification']['THRESHOLD_RATIO'])
            self.run_classifier.SpatialClassifier.proportional_levels( self.PERCENTAGES_LOW, self.PERCENTAGES_HIGH )
            if self.Parameters['Signal Outputting']['SHOW_REVIEWER']:
                time.sleep(0.05)
                self.reviewer.add([0,0,[self.run_classifier.SpatialClassifier.onset_threshold]])
                time.sleep(0.05)
            #TODO onset threshold for spatial classifier will be wrong
            if self.Parameters['Classification']['RESCU']:
                self.segmentor.onset_data_caching = self.Parameters['Classification']['ONSET_DATA_CACHING']
            if self.Parameters['Signal Outputting']['SHOW_MAV']:
                if self.Parameters['Classification']['ONSET_THRESHOLD']:
                    self.mav_plot.set_threshold(self.onset_threshold)
                if self.Parameters['Classification']['CLASSIFIER'] in ['Spatial', 'Simultaneous']:
                    self.mav_plot.set_threshold( self.run_classifier.SpatialClassifier.threshold_radius_norm )
        
        if changed == ['Fitts Law', 'Method'] and self.Parameters['Misc.']['EXPERIMENT']:
            
            self.current_method = "Method_%s" % self.Parameters['Fitts Law']['Method']
            if self.practice_session > 1 and self.Parameters['Misc.']['CARRY_UPDATES']:
                output = self.tdg.emg_extract_onset( CLASSES = self.current_cue_list, 
                                                FL = self.run_FLT, 
                                                subject = self.subject, 
                                                practice_session = self.practice_session, 
                                                PREVIOUS_FILE = self.PREVIOUS_FILE, 
                                                CARRY_UPDATES = self.Parameters['Misc.']['CARRY_UPDATES'],
                                                method = self.current_method)

                self.Xtrain = output[0]
                self.ytrain = output[1]
                self.Xtrain_original = output[2]
                self.ytrain_original = output[3]

            else:
                output = self.tdg.emg_pre_extracted(   CLASSES = self.current_cue_list,
                                                                    PREVIOUS_FILE = self.PREVIOUS_FILE,
                                                                    practice_session = self.practice_session,
                                                                    CARRY_UPDATES = self.Parameters['Misc.']['CARRY_UPDATES'],
                                                                    total_training_samples = self.Parameters['Classification']['CLUSTER_TRAINING_SAMPLES'],
                                                                    onset_threshold_enable = self.Parameters['Classification']['ONSET_THRESHOLD'],
                                                                    onset_scaler = self.Parameters['Calibration']['CALIB_ONSET_SCALAR'],
                                                                    gain = self.Parameters['General']['ELECTRODE_GAIN'],
                                                                    fft_length = self.Parameters['General']['FFT_LENGTH'],
                                                                    EMG_SCALING_FACTOR = EMG_SCALING_FACTOR,
                                                                    emg_window_size = self.window_size,
                                                                    emg_window_step = self.Parameters['General']['WINDOW_STEP'],
                                                                    CALIBRATION = self.Parameters['Calibration']['CALIB_METHOD'],
                                                                    FL = self.run_FLT,
                                                                    subject = self.subject,
                                                                    method = self.current_method)
                self.Xtrain = output[0]
                self.ytrain = output[1]
                self.Xtrain_original = copy.deepcopy(self.Xtrain)
                self.ytrain_original = copy.deepcopy(self.ytrain)

            self.update_training_data(method_change = True)

    def reinitialize_FLT(self):
        
        if not self.run_FLT:
            
            self.run_FLT = True
        
            #self.current_cue_list = self.Parameters['Fitts Law']['CUE_LIST']
            
            
            self.tdg = TrainingDataGenerator()
            # compute training features and labels
            '''output = self.tdg.emg_pre_extracted(   CLASSES = self.current_cue_list,
                                                                PREVIOUS_FILE = [],
                                                                practice_session = 1,
                                                                CARRY_UPDATES = self.Parameters['Misc.']['CARRY_UPDATES'],
                                                                total_training_samples = self.Parameters['Classification']['CLUSTER_TRAINING_SAMPLES'],
                                                                onset_threshold_enable = self.Parameters['Classification']['ONSET_THRESHOLD'],
                                                                onset_scaler = self.Parameters['Calibration']['CALIB_ONSET_SCALAR'],
                                                                gain = self.Parameters['General']['ELECTRODE_GAIN'],
                                                                fft_length = self.Parameters['General']['FFT_LENGTH'],
                                                                EMG_SCALING_FACTOR = EMG_SCALING_FACTOR,
                                                                emg_window_size = self.window_size,
                                                                emg_window_step = self.Parameters['General']['WINDOW_STEP'],
                                                                CALIBRATION = self.Parameters['Calibration']['CALIB_METHOD'],
                                                                FL = self.run_FLT,
                                                                subject = self.subject)
            self.Xtrain = output[0]
            self.ytrain = output[1]
            self.Xtrain_original = copy.deepcopy(self.Xtrain)'''

            # train classifier
            '''print( 'Training classifier...', end = '', flush = True )
            
            self.run_classifier = RunClassifier(    class_subdict_size = self.Parameters['Classification']['CLUSTER_TRAINING_SAMPLES'], 
                                                    Xtrain = self.Xtrain_original, 
                                                    ytrain = self.ytrain, 
                                                    classes = self.current_cue_list,
                                                    perc_lo = [i for i in self.PERCENTAGES_LOW], 
                                                    perc_hi = [i for i in self.PERCENTAGES_HIGH],
                                                    threshold_ratio = self.Parameters['Classification']['THRESHOLD_RATIO'] )
            print( 'Done!' )

            print( 'Creating proportional control filters...', end = '', flush = True )
        
            self.spatial_prop = copy.deepcopy(self.run_classifier)

            self.prop = Proportional_Control(   X = self.Xtrain.T, 
                                                L = self.ytrain, 
                                                onset_threshold = self.onset_threshold_calculation.onset_threshold(     Xtrain = self.Xtrain, 
                                                                                                                        ytrain = self.ytrain, 
                                                                                                                        gain = self.Parameters['General']['ELECTRODE_GAIN'],
                                                                                                                        classifier = ''),
                                                classes = self.current_cue_list, 
                                                proportional_low =  self.PROPORTIONAL_LOW, 
                                                proportional_high = self.PROPORTIONAL_HIGH )'''
            print( 'Done!' )

            # Create update GUI
            '''if self.Parameters['Classification']['RESCU']:

                # create segmentor
                print( 'Creating segmentor...', end = '', flush = True )
                self.segmentor = Segmentation(onset_threshold = self.onset_threshold_calculation.onset_threshold(   Xtrain = self.Xtrain, 
                                                                                                                    ytrain = self.ytrain, 
                                                                                                                    gain = self.Parameters['General']['ELECTRODE_GAIN'],
                                                                                                                    classifier = ''),
                                                overwrite = self.seg_update_k,
                                                plot = self.Parameters['Classification']['SEGMENTATION_PLOT'],
                                                plot_max = np.amax(np.mean(self.Xtrain[:,:8],1))*1.5)
                print( 'Done!' )
                
            # show ploting interfaces
            if self.Parameters['Signal Outputting']['SHOW_MAV']:
                print( 'Creating real-time data plot...', end = '', flush = True )
                if self.Parameters['Classification']['CLASSIFIER'] in ['Spatial', 'Simultaneous']:
                    channel_max = self.run_classifier.SpatialClassifier.channel_max
                else:            
                    if self.Parameters['General']['SOURCE'] == 'Myo Band':
                        channel_max = 0.3/50
                    elif self.Parameters['General']['SOURCE'] == 'Sense':
                        channel_max = 0.3
                
                try:
                    self.mav_plot.close()
                except:
                    pass
                
                self.mav_plot = RealTimeDataPlot(   num_channels = self.Parameters['General']['DEVICE_ELECTRODE_COUNT'],
                                                    channel_min = 0,
                                                    channel_max = channel_max,
                                                    sampling_rate = self.Parameters['General']['SAMPLING_RATE'] / self.Parameters['General']['WINDOW_STEP'],
                                                    buffer_time = 5,
                                                    title = "MAV Signals",
                                                    classifier =  self.Parameters['Classification']['CLASSIFIER'],
                                                    source = self.Parameters['General']['SOURCE'])

                if self.Parameters['Classification']['ONSET_THRESHOLD'] is False:
                    self.mav_plot.set_threshold( 0 )
                else:
                    self.mav_plot.set_threshold( self.onset_threshold_calculation.onset_threshold(  Xtrain = self.Xtrain, 
                                                                                                    ytrain = self.ytrain, 
                                                                                                    gain = self.Parameters['General']['ELECTRODE_GAIN'],
                                                                                                    classifier = '') )

                if self.Parameters['Classification']['CLASSIFIER'] in ['Spatial', 'Simultaneous']:
                        self.mav_plot.set_threshold( self.run_classifier.SpatialClassifier.threshold_radius_norm )
                print( 'Done!' )

            if self.Parameters['Signal Outputting']['SHOW_VELOCITY']:        
                try:
                    self.glide.close()
                except:
                    pass
                
                self.glide = RealTimeGlidePlot( classes = self.current_cue_list, 
                                                all_classes = self.Parameters['Calibration']['ALL_CLASSES'], 
                                                prop_lo = [i for i in self.PROPORTIONAL_LOW], 
                                                prop_hi = [i for i in self.PROPORTIONAL_HIGH],
                                                perc_lo = [i for i in self.PERCENTAGES_LOW], 
                                                perc_hi = [i for i in self.PERCENTAGES_HIGH],  
                                                bar_min = self.onset_threshold_calculation.onset_threshold( Xtrain = self.Xtrain, 
                                                                                                            ytrain = self.ytrain, 
                                                                                                            gain = self.Parameters['General']['ELECTRODE_GAIN'],
                                                                                                            classifier = ''), 
                                                bar_max = np.amax(np.mean(self.Xtrain[:,:8],1)),
                                                method = self.Parameters['Proportional Control']['PROPORTIONAL_METHOD'],
                                                spatial_velocity_dicts = self.spatial_prop.SpatialClassifier.proportional_range)
            print( 'Done!' )

            if self.Parameters['Signal Outputting']['SHOW_REVIEWER']:
                print( 'Creating real-time reviewer plot...', end = '', flush = True )

                try:
                    self.reviewer.close()
                except:
                    pass

                self.reviewer = SepRepPlot( SUBJECT = self.subject, 
                                            CLASSES = self.current_cue_list, 
                                            Xtrain = self.Xtrain, 
                                            ytrain = self.ytrain,
                                            CLASSIFIER = self.Parameters['Classification']['CLASSIFIER'], 
                                            ALL_CLASSES = self.Parameters['Calibration']['ALL_CLASSES'],
                                            CLUSTER_TRAINING_SAMPLES = self.Parameters['Classification']['CLUSTER_TRAINING_SAMPLES'],
                                            realtime = True,
                                            perc_lo = [i for i in self.PERCENTAGES_LOW], 
                                            perc_hi = [i for i in self.PERCENTAGES_HIGH],
                                            threshold_ratio = self.run_classifier.SpatialClassifier.onset_threshold, #self.Parameters['Classification']['THRESHOLD_RATIO'],
                                            recalculate_RLDA = self.Parameters['Classification']['REGENERATE_CLASSIFIER'],
                                            proj = self.run_classifier.SpatialClassifier.proj,
                                            stdscaler = self.run_classifier.SpatialClassifier.stdscaler,
                                            classifier_obj = self.run_classifier)
                print( 'Done!' )'''

        self.FLT = FittsLawTask(classes = self.current_cue_list)

    
    def generate_sensitivities(self):
        self.SENSITIVITIES = {}
        for i, cl in enumerate(self.current_cue_list):
            if self.Parameters['Output Filtering']['INDIV_SENS'] == False and cl != 'REST':
                self.SENSITIVITIES[cl] = self.Parameters['Output Filtering']['GLOBAL_SENS']
            else:
                self.SENSITIVITIES[cl] = self.Parameters['Output Filtering'][cl.replace(' ', '_')+'_SENS']
    
    def generate_proportionals(self):
        self.PROPORTIONAL_LOW = []
        self.PROPORTIONAL_HIGH = []
        self.PERCENTAGES_LOW = []
        self.PERCENTAGES_HIGH = []

        for cl in self.current_cue_list:
            if cl != 'REST':
                self.PROPORTIONAL_LOW.append(self.Parameters['Proportional Control'][cl.replace(' ', '_')+'_VEL_LOW'][0])
                self.PROPORTIONAL_HIGH.append(self.Parameters['Proportional Control'][cl.replace(' ', '_')+'_VEL_HIGH'][0])
                self.PERCENTAGES_LOW.append( self.Parameters['Proportional Control'][cl.replace(' ', '_')+'_VEL_LOW'][1] )
                self.PERCENTAGES_HIGH.append( self.Parameters['Proportional Control'][cl.replace(' ', '_')+'_VEL_HIGH'][1])

    def set_velocity_thresholds(self):
        velocity_dict = self.prop.thresholds
        for cl in self.current_cue_list:
            if cl != 'REST':
                self.Parameters['Proportional Control'][cl.replace(' ', '_')+'_VEL_LOW'][0] = velocity_dict[cl][0]
                self.Parameters['Proportional Control'][cl.replace(' ', '_')+'_VEL_HIGH'][0] = velocity_dict[cl][1]



    def checkIfProcessRunning(self, processName):
        '''
        Check if there is any running process that contains the given name processName.
        '''
        #Iterate over the all the running process
        for proc in psutil.process_iter():
            # Check if process name contains the given name string.
            try:
                if processName in proc.name():
                    return True
            except psutil.AccessDenied:
                pass
        return False

    def update_training_data(self, seprep = False, method_change = False):

        if seprep or method_change:
            self.reinitialize_parameters(['Output Filtering', ''])
            
            self.generate_proportionals() 
            # train classifier
            print( 'Training classifier...', end = '', flush = True )
            
            self.run_classifier = RunClassifier(    class_subdict_size = self.Parameters['Classification']['CLUSTER_TRAINING_SAMPLES'], 
                                                    Xtrain = self.Xtrain, 
                                                    ytrain = self.ytrain, 
                                                    classes = self.current_cue_list,
                                                    perc_lo = [i for i in self.PERCENTAGES_LOW], 
                                                    perc_hi = [i for i in self.PERCENTAGES_HIGH],
                                                    threshold_ratio = self.Parameters['Classification']['THRESHOLD_RATIO'] )
        
            self.spatial_prop = copy.deepcopy(self.run_classifier)

            self.run_classifier.init_classifiers(  Xtrain = self.Xtrain,
                                                        ytrain = self.ytrain, 
                                                        classes = self.current_cue_list,
                                                        perc_lo = [i for i in self.PERCENTAGES_LOW], 
                                                        perc_hi = [i for i in self.PERCENTAGES_HIGH],
                                                        threshold_ratio = self.Parameters['Classification']['THRESHOLD_RATIO'],
                                                        recalculate_RLDA = self.Parameters['Classification']['REGENERATE_CLASSIFIER']
                                                    )
            self.spatial_prop.init_classifiers(  Xtrain = self.Xtrain,
                                                    ytrain = self.ytrain, 
                                                    classes = self.current_cue_list,
                                                    perc_lo = [i for i in self.PERCENTAGES_LOW], 
                                                    perc_hi = [i for i in self.PERCENTAGES_HIGH],
                                                    threshold_ratio = self.Parameters['Classification']['THRESHOLD_RATIO'],
                                                    recalculate_RLDA = self.Parameters['Classification']['REGENERATE_CLASSIFIER'])

            self.vramp = VelocityRampFilter( n_classes = len( self.current_cue_list ),
                                                increment = self.Parameters['Proportional Control']['VELOCITY_RAMP_INCREMENT'],
                                                decrement = self.Parameters['Proportional Control']['VELOCITY_RAMP_DECREMENT'],
                                                max_bin_size = self.Parameters['Proportional Control']['VELOCITY_RAMP_SIZE'] )

            if self.Parameters['Classification']['RESCU']:

                self.reinitialize_parameters(['Classification','RESCU'])
                '''self.gui.close()
                # create updater GUI
                print( 'Creating updater + GUI...', end = '', flush = True )
                self.gui = UpdateGUI( classes = tuple( self.current_cue_list ), n_segments = 5 )
                print( 'Done!' )'''
            

        self.reinitialize_parameters(['Classification',''])

        if seprep: 
            self.reviewer.add_classifier(self.run_classifier)

        # recalculate velocity thresholds
        if self.Parameters['Signal Outputting']['SHOW_VELOCITY']:

            if seprep or method_change:
                self.prop = Proportional_Control(   X = self.Xtrain.T, 
                                                    L = self.ytrain, 
                                                    onset_threshold = self.onset_threshold_calculation.onset_threshold(     Xtrain = self.Xtrain, 
                                                                                                                            ytrain = self.ytrain, 
                                                                                                                            gain = self.Parameters['General']['ELECTRODE_GAIN'],
                                                                                                                            classifier = ''),
                                                    classes = self.current_cue_list, 
                                                    proportional_low =  self.PROPORTIONAL_LOW, 
                                                    proportional_high = self.PROPORTIONAL_HIGH )

                self.set_velocity_thresholds()
            else:
                self.prop.proportional_percentile_rearrange(self.Xtrain.T,
                                                            self.ytrain, 
                                                            self.onset_threshold_calculation.onset_threshold( Xtrain = self.Xtrain, 
                                                                                                                ytrain = self.ytrain, 
                                                                                                                gain = self.Parameters['General']['ELECTRODE_GAIN'],
                                                                                                                classifier = ''))
            self.glide.close()

            self.reinitialize_parameters(['Signal Outputting','SHOW_VELOCITY'])

        if self.Parameters['Signal Outputting']['SHOW_REVIEWER'] and not seprep:
            self.reviewer.close()
            self.reinitialize_parameters(['Signal Outputting','SHOW_REVIEWER'])

        if self.Parameters['Signal Outputting']['SHOW_CONTROL'] and seprep:
            self.ctrl_plot.close()
            print( 'Creating real-time control plot...', end = '', flush = True )
            self.ctrl_plot = RealTimeControlPlot(   self.current_cue_list,
                                                    num_channels = 2, 
                                                    sampling_rate = self.Parameters['General']['SAMPLING_RATE'] / self.Parameters['General']['WINDOW_STEP'], 
                                                    buffer_time = 5, 
                                                    title = "Control Signal (filtered and raw)" )
            print( 'Done!' )
            
        # reinitialize classifier
        self.Protocol_Data[self.current_method]['Current_Xtrain'] = copy.deepcopy(self.Xtrain)
        self.Protocol_Data[self.current_method]['Current_ytrain'] = copy.deepcopy(self.ytrain)

        self.run_classifier.init_classifiers(  Xtrain = self.Xtrain,
                                                ytrain = self.ytrain, 
                                                classes = self.current_cue_list,
                                                perc_lo = [i for i in self.PERCENTAGES_LOW], 
                                                perc_hi = [i for i in self.PERCENTAGES_HIGH],
                                                threshold_ratio = self.Parameters['Classification']['THRESHOLD_RATIO'],
                                                recalculate_RLDA = self.Parameters['Classification']['REGENERATE_CLASSIFIER']
                                                )

        if self.Parameters['Classification']['ONSET_THRESHOLD'] is False:
            self.mav_plot.set_threshold( 0 )
        else:
            self.mav_plot.set_threshold( self.onset_threshold_calculation.onset_threshold(    Xtrain = self.Xtrain, 
                                                                                                ytrain = self.ytrain, 
                                                                                                gain = self.Parameters['General']['ELECTRODE_GAIN'],
                                                                                                classifier = '') )
        if self.Parameters['Classification']['CLASSIFIER'] in ['Spatial', 'Simultaneous']:
            self.mav_plot.set_threshold( self.run_classifier.SpatialClassifier.threshold_radius_norm )

        with open( self.SAVE_FILE, 'wb' ) as f:
            pickle.dump( self.Protocol_Data, f )

        data = None
        emg_window = []
        while data is not None:
            data = self.source.state                            

    def run_main_process(self):

        def simple_exponential_smoothing(series, alpha):
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

        prev_mov = 'REST'

        primary_relative_proportional_smoothor = Proportional_Smoothor()
        primary_objective_proportional_smoothor = Proportional_Smoothor()
        secondary_relative_proportional_smoothor = Proportional_Smoothor()
        secondary_objective_proportional_smoothor = Proportional_Smoothor()

        output_smoothor = Output_Smoothor()

        print( 'Starting data streaming...', end = '\n', flush = True )
        self.source.run()
        try:
            emg_window = []

            move_sent = 0

            raw_output_predictions = []
            output_predictions = []

            while True:
                
                data = self.source.state
                # check for new data
                if data is not None:
                    emg_window.append( data[:self.Parameters['General']['DEVICE_ELECTRODE_COUNT']].copy() )
                    # if window full
                    if len( emg_window ) >= self.window_size:

                        t = time.time()
                        # extract features
                        win = np.vstack( emg_window ) / EMG_SCALING_FACTOR
                        if self.Parameters['General']['FFT_LENGTH'] in ['Custom Adapt']:
                            feat = np.hstack( [ self.td5.filter( win ), self.fft.filterAdapt( win ) ] )
                        else:
                            feat = np.hstack( [ self.td5.filter( win ), self.fft.filter( win ) ] )[:-8]

                        CLASSIFIER = self.Parameters['Classification']['CLASSIFIER']
                        if CLASSIFIER == 'Simultaneous' or CLASSIFIER == 'Spatial' or CLASSIFIER == 'EASRC':
                            feat_filtered = self.smoothor.filter( feat )
                        else:
                            feat_filtered = feat
        
                        # classification
                        pred = self.run_classifier.emg_classify(feat_filtered[:,:], self.onset_threshold, CLASSIFIER, self.Parameters['Classification']['ONSET_THRESHOLD'])
                        if CLASSIFIER == 'Simultaneous':

                            grip_pred = pred[0]
                            wrist_pred = pred[1]
                            # TODO Handle this
                            pred = grip_pred

                        raw_pred = pred
                        raw_output_predictions.append( raw_pred )

                        self.Protocol_Data[self.current_method]['predictions'].append( pred )

                        # output smoothing
                        pred = output_smoothor.get_mode(current_output = pred, output_smooth = self.Parameters['Output Filtering']['SMOOTHING_FILTER_LENGTH'])

                        # output filtering

                        if CLASSIFIER != 'Simultaneous':
                            filtered_pred = self.primary.filter( pred = self.current_cue_list[ pred ], filter = self.Parameters['Output Filtering']['OUTPUT_FILTER'] )
                            pred = self.current_cue_list.index( filtered_pred )    # primary filter
                        else:
                            filtered_pred = self.primary.filter( pred = self.current_cue_list[ grip_pred ], filter = self.Parameters['Output Filtering']['OUTPUT_FILTER'] )
                            pred = self.current_cue_list.index( filtered_pred ) 

                            filtered_secondary_pred = self.secondary.filter( pred = self.current_cue_list[ wrist_pred ], filter = self.Parameters['Output Filtering']['OUTPUT_FILTER'] )
                            secondary_pred = self.current_cue_list.index( filtered_secondary_pred )

                        # calculate velocity
                        # TODO: implement for simultaneous
                        speed = self.vramp.filter( pred ) if self.Parameters['Proportional Control']['VELOCITY_RAMP_ENABLED'] and CLASSIFIER != 'Simultaneous' else 1.0               # velocty ramp filter
                        if pred == 0:
                            prop_vel = 0
                            obj_prop_vel = 0
                        else:
                            prop_vel = 1
                            obj_prop_vel = 1
                        secondary_prop_vel = 0
                        secondary_obj_prop_vel = 0
                    
                        if self.Parameters['Proportional Control']['PROPORTIONAL_ENABLED']:
                            if self.Parameters['Proportional Control']['PROPORTIONAL_METHOD'] == 'Amplitude':

                                prop_vel = self.prop.Proportional( feat_filtered, pred, self.Parameters['Proportional Control']['PROP_SMOOTH'] )

                            elif self.Parameters['Proportional Control']['PROPORTIONAL_METHOD'] == 'Spatial':
                                _, prop_vel = self.spatial_prop.emg_classify(feat_filtered, self.onset_threshold, CLASSIFIER, self.Parameters['Classification']['ONSET_THRESHOLD'], return_aux_data = True)
                                if len(prop_vel) != 0:
                                    obj_prop_vel = prop_vel[1]
                                    prop_vel = prop_vel[0]
                                else:
                                    prop_vel = 0
                                    obj_prop_vel = 0
                                
                                if type(prop_vel) == list and len(prop_vel) > 0:

                                    try:
                                        if secondary_pred > 0:
                                            secondary_prop_vel = secondary_relative_proportional_smoothor.get_mean(prop_vel[secondary_pred-1], self.Parameters['Proportional Control']['PROP_SMOOTH'])
                                            secondary_obj_prop_vel = secondary_objective_proportional_smoothor.get_mean(obj_prop_vel[secondary_pred-1], self.Parameters['Proportional Control']['PROP_SMOOTH'])

                                    except: pass

                                    prop_vel = primary_relative_proportional_smoothor.get_mean(prop_vel[pred-1], self.Parameters['Proportional Control']['PROP_SMOOTH'])
                                    obj_prop_vel = primary_objective_proportional_smoothor.get_mean(obj_prop_vel[pred-1], self.Parameters['Proportional Control']['PROP_SMOOTH'])

                        output_predictions.append( pred )

                        # plot mav data
                        if self.Parameters['Signal Outputting']['SHOW_MAV']:
                            if self.Parameters['Classification']['CLASSIFIER'] in ['Spatial', 'Simultaneous']:
                                self.mav_plot.add( [copy.deepcopy(feat_filtered[:,:8]), self.run_classifier.SpatialClassifier.distancFromRest] )
                            else:                
                                self.mav_plot.add( copy.deepcopy(feat_filtered[:,:8]) )

                        if self.Parameters['Signal Outputting']['SHOW_REVIEWER']:
                            self.reviewer.add(copy.deepcopy([feat_filtered, pred]))
                            try:
                                self.Xtrain, self.ytrain, classes = self.reviewer.state3
                                print(classes)
                                if self.run_FLT:
                                    self.Parameters['Fitts Law']['CUE_LIST'] = copy.deepcopy(classes)
                                else:
                                    self.Parameters['Calibration']['CUE_LIST'] = copy.deepcopy(classes)
                                
                                self.current_cue_list = copy.deepcopy(classes)

                                for _class in self.Parameters['Calibration']['ALL_CLASSES']:
                                    self.Parameters['Calibration'][_class] = _class in self.current_cue_list

                                self.init.set_parameters( parameters = self.Parameters )
                                self.init.save_subject_parameters(subject=SUBJECT, parameters = self.Parameters)
                                self.update_training_data(seprep=True)
                            except Exception as e:
                                pass #print(e)

                        # plot control data
                        if self.Parameters['Signal Outputting']['SHOW_CONTROL']:
                            
                            control_out = []
                            control_out.append(self.current_cue_list[ pred ])
                            if CLASSIFIER == 'Simultaneous':
                                control_out.append(self.current_cue_list[ secondary_pred ])
                            else:
                                control_out.append(self.current_cue_list[ raw_pred ])
                            self.ctrl_plot.add( tuple( control_out ) )
                        
                        veloc = speed * prop_vel
                        secondary_veloc = speed * secondary_prop_vel
                        # plot velocity data
                        if self.Parameters['Signal Outputting']['SHOW_VELOCITY']:
                            
                            if self.Parameters['Proportional Control']['PROPORTIONAL_METHOD'] == 'Amplitude':
                                self.glide.add( self.current_cue_list[pred], veloc, np.mean(feat_filtered[:, :8]) )

                            elif self.Parameters['Proportional Control']['PROPORTIONAL_METHOD'] == 'Spatial':
                                self.glide.add( self.current_cue_list[pred], veloc, obj_prop_vel )
                        
                        if self.run_FLT:
                            
                            if CLASSIFIER == 'Simultaneous' and secondary_pred != 0:

                                simultaneous_veloc_scalar = 1
                                pri_pred = self.current_cue_list[pred] #self.Parameters['Fitts Law']['CUE_LIST'][pred]
                                sec_pred = self.current_cue_list[secondary_pred] #self.Parameters['Fitts Law']['CUE_LIST'][secondary_pred]
                                pri_vel = veloc * simultaneous_veloc_scalar
                                sec_vel = secondary_veloc * simultaneous_veloc_scalar
                                #print('Veloc: ', [pri_vel,sec_vel], '  Pred: ', [pri_pred,sec_pred])
                                self.FLT.add(tuple([[pri_vel,sec_vel],[pri_pred,sec_pred], feat_filtered]))
                            else:
                                self.FLT.add(tuple([veloc, self.current_cue_list[pred], feat_filtered]))

                            if self.FLT._exit_event.is_set():
                                #FittsLawTaskResults(self.subject)
                                self.run_FLT = False
                            
                            try:
                                temp_method = self.FLT.state
                                if temp_method is not None:
                                    self.Parameters['Fitts Law']['Method'] = temp_method
                                    self.reinitialize_parameters(['Fitts Law', 'Method'])

                            except: pass
                        # plot mav data
                        if self.Parameters['Signal Outputting']['SHOW_MYOTRAIN']:
                            
                            if prev_mov != self.current_cue_list[pred]:
                                self.myotrain.publish( msg = self.current_cue_list[pred], speed = speed * prop_vel )
                                prev_mov = self.current_cue_list[pred]

                        if self.Parameters['Signal Outputting']['PROSTHESIS']:
                            if move_sent != pred:
                                self.prosthesis.publish( self.moves[ pred ] )
                                move_sent = copy.deepcopy(pred)

                        if self.Parameters['Classification']['RESCU']:
                            # run segmentation
                            # TODO save segmentation data, fix rest segments
                            self.segmentor.segmentation_process(   feat = feat,
                                                                    pred = pred,
                                                                    gui = self.gui,
                                                                    CONFIDENCE_REJECTION = self.Parameters['Classification']['CONFIDENCE_REJECTION'],
                                                                    raw_output_predictions = raw_output_predictions,
                                                                    output_predictions = output_predictions)

                            if self.segmentor.new_onset:
                                self.Protocol_Data[self.current_method]['segment_onset'].append( len(self.Protocol_Data[self.current_method]['features']) )
                            
                            if self.segmentor.new_activity:
                                self.Protocol_Data[self.current_method]['Cache'].extend(self.segmentor.current_cache)
                                self.Protocol_Data[self.current_method]['Segments'] = self.segmentor.activity
                                
                            # run updating
                            update_state = self.gui.state
                            
                            if update_state is not None:
                                # if GUI exit pressed turn everything off
                                if update_state == -1:  # continue button was pressed, we are done, close everything
                                    if self.Parameters['Signal Outputting']['SHOW_MAV']:
                                        self.mav_plot.close()
                                    if self.Parameters['Signal Outputting']['SHOW_CONTROL']:
                                        self.ctrl_plot.close()
                                    if self.Parameters['Signal Outputting']['SHOW_VELOCITY']:
                                        self.glide.close()
                                    self.init.close()
                                    if self.Parameters['Classification']['RESCU']:
                                        self.gui.close()
                                        if self.Parameters['Classification']['SEGMENTATION_PLOT']:
                                            self.segmentor.close()
                                    break # break out of while loop, clean up sense controller
                                # send segment to reviewer
                                elif type(update_state[0]) == int and update_state[0] == -2:
                                    if self.Parameters['Signal Outputting']['SHOW_REVIEWER']:
                                        self.reviewer.add_segment(copy.deepcopy(update_state[1]))
                                        
                                # Update
                                else:
                                    if self.Parameters['Misc.']['EXPERIMENT'] and self.current_method == 'Method_2':
                                        pass
                                        '''self.Protocol_Data[self.current_method]['updates'].append( len(self.segmentor.get_cache_dictionary)-update_state[2]-1 )
                                        self.Protocol_Data[self.current_method]['segment_offset'].append( len(self.Protocol_Data[self.current_method]['features']) )

                                        with open( self.EXPERIMENTAL_DATA_FILE, 'wb' ) as f:
                                            pickle.dump( self.Protocol_Data, f )'''
                                    else:
                                        
                                        try:
                                            if update_state[0] == -3:
                                                try:
                                                    self.Xtrain = self.Protocol_Data[self.current_method]['Xtrain'][-2]
                                                    del self.Protocol_Data[self.current_method]['Xtrain'][-2] 
                                                    self.ytrain = self.Protocol_Data[self.current_method]['ytrain'][-2]
                                                    del self.Protocol_Data[self.current_method]['ytrain'][-2]   
                                                except: continue                                
                                        except:
                                            # Update
                                            if self.current_method == 'Method_1':
                                                self.Xtrain = self.update.Update_class( self.Xtrain, self.ytrain, update_state[0], update_state[1] ).T
                                                # save data
                                                #self.Protocol_Data[self.current_method]['updates'].append( len(self.segmentor.get_cache_dictionary())-update_state[2]-1 )
                                                self.Protocol_Data[self.current_method]['Xtrain'].append( self.Xtrain )              # store new training matrix
                                                self.Protocol_Data[self.current_method]['ytrain'].append( self.ytrain ) 
                                            

                                        self.update_training_data()


                        self.Protocol_Data[self.current_method]['features'].append( feat.copy() )
                        self.Protocol_Data[self.current_method]['filtered_output'].append( pred )
                        self.Protocol_Data[self.current_method]['velocity'].append( speed * prop_vel )
                        
                        emg_window = emg_window[self.Parameters['General']['WINDOW_STEP']:]

                        t2 = time.time()
                        #print( (t2-t)  * 1000 )

                init_change = self.init.change
                
                if init_change is not None:
                    if init_change == ['General', 'EXIT_BUTTON']:  # close button was pressed, we are done, close everything
                        if self.Parameters['Signal Outputting']['SHOW_MAV']:
                            self.mav_plot.close()
                        if self.Parameters['Signal Outputting']['SHOW_CONTROL']:
                            self.ctrl_plot.close()
                        if self.Parameters['Signal Outputting']['SHOW_VELOCITY']:
                            self.glide.close()
                        if self.Parameters['Signal Outputting']['SHOW_REVIEWER']:
                            self.reviewer.close()
                        if self.Parameters['Classification']['RESCU']:
                            self.gui.close()
                            if self.Parameters['Classification']['SEGMENTATION_PLOT']:
                                self.segmentor.close()
                        if self.run_FLT:
                            self.FLT.close()
                        self.init.close()
                        break # break out of while loop, clean up sense controller
                    self.Parameters = self.init.parameters
                    if self.Parameters['General']['SOURCE'] == 'Myo Band':
                        self.window_size = self.Parameters['General']['WINDOW_SIZE']//5
                    else:
                        self.window_size = self.Parameters['General']['WINDOW_SIZE']
                    self.reinitialize_parameters(init_change)
                
                try:
                    velocity_change = self.glide.change
                    if velocity_change:
                        velocity = self.glide.thresholds
                        if self.Parameters['Proportional Control']['PROPORTIONAL_METHOD'] == 'Spatial':
                            for key, tr in velocity.items():
                                self.Parameters['Proportional Control'][key.replace(' ', '_')+'_VEL_LOW'][1] = tr[0]
                                self.Parameters['Proportional Control'][key.replace(' ', '_')+'_VEL_HIGH'][1] = tr[1]
                                
                        elif self.Parameters['Proportional Control']['PROPORTIONAL_METHOD'] == 'Amplitude':
                            for key, tr in velocity.items():
                                self.Parameters['Proportional Control'][key.replace(' ', '_')+'_VEL_LOW'][0] = tr[0]
                                self.Parameters['Proportional Control'][key.replace(' ', '_')+'_VEL_HIGH'][0] = tr[1]
                        
                        self.init.set_parameters( parameters = self.Parameters )
                        self.reinitialize_parameters(['Proportional Control', ''])
                except:
                    pass
        
        finally:
            print( 'Cleaning up Sense controller...', end = '', flush = True )
            self.source.stop()
            self.source.close()
            print( 'Done!' )

        print( 'Script has ended!' )

class Proportional_Smoothor:
    def __init__( self ):
        self._rms = deque()
    
    def get_mean( self, current_prop, prop_smooth ):
        self._rms.append(current_prop)

        while len(self._rms) > prop_smooth:
            self._rms.popleft()

        current_mean_rms = np.mean(self._rms)

        return current_mean_rms

class Output_Smoothor:
    def __init__( self ):
        self._rms = deque()
    
    def get_mode( self, current_output, output_smooth ):
        self._rms.append(current_output)

        while len(self._rms) > output_smooth:
            self._rms.popleft()

        current_majority_rms = max( set(self._rms), key = self._rms.count)

        return current_majority_rms

if __name__ == '__main__':
    RESCU()