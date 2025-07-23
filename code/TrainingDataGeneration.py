import numpy as np
import pickle
from scipy.io import loadmat

from DownSampler import DownSampler
from OnsetThreshold import Onset
from TimeDomainFilter import TimeDomainFilter
from FourierTransformFilter import FourierTransformFilter
import os
from os import path

downsample = DownSampler()

class TrainingDataGenerator:
    def __init__(self):
        pass

#-----------------------------------------------------------------------------------------------------------------------
# PRIVATE METHODS
#-----------------------------------------------------------------------------------------------------------------------
    def emg_extract_onset(self, CLASSES = [], FL = False, subject = 1, practice_session = 0, PREVIOUS_FILE = [], CARRY_UPDATES = False, class_subdict_size = 50, method = 'Method_1'):
        
        dsample = DownSampler()

        if FL == False:
            folder = '.\Data\Subject_%d' % int(subject)
            folder2 = 'calibration_data'
            filename = 'calibrate_subject_%d.mat' % int(subject)
            if not path.exists(folder):
                os.mkdir(folder)
            if not path.exists(folder + '/' + folder2):
                os.mkdir(folder + '/' + folder2)
            filepath = folder + '/' + folder2 + '/' + filename
            if not path.exists(filepath):
                f = open(filepath, "w")
                f.write("")
                f.close()
            training_file = filepath

        elif FL == True:
            folder = '.\Data\Subject_%d' % int(subject)
            folder2 = 'calibration_data'
            filename = 'calibratefittslaw_subject_%d.mat' % int(subject)
            if not path.exists(folder):
                os.mkdir(folder)
            if not path.exists(folder + '/' + folder2):
                os.mkdir(folder + '/' + folder2)
            filepath = folder + '/' + folder2 + '/' + filename
            if not path.exists(filepath):
                f = open(filepath, "w")
                f.write("")
                f.close()
            training_file = filepath

        training_data = loadmat( training_file )['calibrate'][0][0]

        if CARRY_UPDATES and practice_session > 1:
            pkl_file = open(PREVIOUS_FILE, 'rb')
            Protocol_Data = pickle.load(pkl_file)
            pkl_file.close()

            Xtrain = Protocol_Data[method]['Current_Xtrain']
            ytrain = Protocol_Data[method]['Current_ytrain']

            pkl_file = open(PREVIOUS_FILE, 'rb')
            Protocol_Data = pickle.load(pkl_file)
            pkl_file.close()

            Xtrain_ori = Protocol_Data[method]['Original_Xtrain']
            ytrain_ori = Protocol_Data[method]['Original_ytrain']

            print("Loaded calibration dictionary from SESSION ", practice_session-1)

            return [Xtrain, ytrain, Xtrain_ori, ytrain_ori]

        else:

            Xtrain = []
            Xtrain_full = []
            ytrain = []

            class_count = 0
            for cue in CLASSES:
                Xtrain_full.append(np.vstack( training_data[cue][0][0] ))
                temp_class, Idx = dsample.uniform(training_data[cue][0][0], int( class_subdict_size ) )
                Xtrain.append( temp_class )
                ytrain.append( class_count * np.ones( ( Xtrain[-1].shape[0], ) ) )
                class_count += 1
        
            Xtrain = np.vstack( Xtrain )
            ytrain = np.hstack( ytrain )

            return [Xtrain, ytrain]
    
    def emg_pre_extracted(  self, PREVIOUS_FILE, practice_session, CARRY_UPDATES, CLASSES, total_training_samples, onset_threshold_enable, onset_scaler, gain = 4, fft_length = 64, EMG_SCALING_FACTOR = 10000, emg_window_size = 200, emg_window_step = 10, 
                            CALIBRATION = 'Onset Calibration', onset_inclusion = 50, FL = False, subject = 1, method = 'Method_1'):
        
        dsample = DownSampler()
        td5 = TimeDomainFilter()
        fft = FourierTransformFilter( fftlen = fft_length )

        NUM_CLASSES = len(CLASSES)

        if CALIBRATION == 'Onset Calibration':
            return self.emg_extract_onset(CLASSES = CLASSES, FL = FL, subject = subject, practice_session = practice_session, PREVIOUS_FILE = PREVIOUS_FILE, class_subdict_size = total_training_samples, method = method)
        else:
            folder = '.\Data\Subject_%d' % int(subject)
            filename = 'train.mat'
            if not path.exists(folder):
                os.mkdir(folder)
            filepath = folder + '/' + filename
            if not path.exists(filepath):
                f = open(filepath, "w")
                f.write("")
                f.close()
            training_file = filepath

            training_data = loadmat( training_file )['train'][0]

        if CARRY_UPDATES and practice_session > 1:
            pkl_file = open(PREVIOUS_FILE, 'rb')
            Protocol_Data = pickle.load(pkl_file)
            pkl_file.close()

            Xtrain = Protocol_Data[method]['Current_Xtrain']
            ytrain = Protocol_Data[method]['Current_ytrain']

            print("Loaded calibration dictionary from SESSION ", practice_session-1)

        else:

            num_trials = len( training_data )
            Xtrain = []
            Xtrain_full = []
            ytrain = []


            if CALIBRATION == 'Onset Calibration':
                class_count = 0
                for cue in CLASSES:
                    Xtrain_full.append(np.vstack( training_data[cue][0][0] ))
                    temp_class, Idx = dsample.uniform(training_data[cue][0][0], int( total_training_samples ) )
                    Xtrain.append( temp_class )
                    ytrain.append( class_count * np.ones( ( Xtrain[-1].shape[0], ) ) )
                    class_count += 1
            else:
                for i in range( len( CLASSES ) ):
                    class_data = []
                    for j in range( num_trials ):
                        raw_data = training_data[ j ][ CLASSES[ i ] ][0][0]
                        num_samples = raw_data.shape[0]
                        idx = 0
                        while idx + emg_window_size < num_samples:
                            window = raw_data[idx:(idx+emg_window_size),:] / EMG_SCALING_FACTOR
                            time_domain = td5.filter( window ).flatten()
                            freq_domain = fft.filter( window ).flatten()
                            class_data.append( np.hstack( [ time_domain, freq_domain ] )[:-8] )
                            idx += emg_window_step
                    
                    tmp, _ = dsample.uniform( np.vstack( class_data ), int( total_training_samples ) )
                    Xtrain.append( tmp )
                    ytrain.append( i * np.ones( ( Xtrain[-1].shape[0], ) ) )
        
            Xtrain = np.vstack( Xtrain )
            ytrain = np.hstack( ytrain )

            if onset_threshold_enable:
                onset_threshold_calculation = Onset( 50, 1.1 )
            else:
                onset_threshold_calculation = Onset( onset_inclusion, onset_scaler )

            onset_threshold = onset_threshold_calculation.onset_threshold(Xtrain,ytrain, gain)

            try:
                class_count = 0
                Xtrain_temp = []
                ytrain_temp = []
                for i in range(NUM_CLASSES):
                    remaining_in_class = []
                    
                    for k in Xtrain_full[i]:
                        tempmean = np.mean(k[:8,])
                        if i > 0:
                            if tempmean > onset_threshold:
                                remaining_in_class.append(k)
                        else:
                            remaining_in_class.append(k)
                    temp_class, Idx = dsample.uniform(np.vstack(remaining_in_class), int( total_training_samples ) )
                    
                    Xtrain_temp.append( temp_class )
                    ytrain_temp.append( class_count * np.ones( ( Xtrain_temp[-1].shape[0], ) ) )
                    class_count += 1

                Xtrain = np.vstack( Xtrain_temp )
                ytrain = np.hstack( ytrain_temp )
            except:
                print("not enough datapoints")
    
        return [Xtrain, ytrain]

    def smg_pre_extracted(self, CLASSES, FL = False, subject = 1, folder=[]):
       
        NUM_CLASSES = len(CLASSES)

        #TODO: Fix this
        total_training_samples=len( CLASSES )

        dsample = DownSampler()

        subject_folder =  os.path.join( '.','Data',f'Subject_{subject}','SMG')
        
        if len(folder) ==0 :
            calibration_folder = 'calibrate'
        else:
            calibration_folder = folder
        calibration_filename = 'calibrate_subject_%d.mat' % int(subject)
        training_file_fullpath = os.path.join(subject_folder,calibration_folder,'calibrate',calibration_filename)
        print(training_file_fullpath)
        all_training_data = loadmat( training_file_fullpath )
        training_data = all_training_data['calibrate'][0][0]                            

        Xtrain = []
        ytrain = []

        percent = 0.75
        trim = 0.25

        class_count = 0
        for cue in CLASSES:
            # Xtrain_full.append(np.vstack( training_data[cue][0][0] ))
            # temp_class, Idx = dsample.uniform(training_data[cue][0][0], int( total_training_samples // len( CLASSES ) ) )
            #TODO: fix
            Xtrain.append( training_data[cue][0][0][ int( training_data[cue][0][0].shape[0]*(trim) ):int(training_data[cue][0][0].shape[0]*(1-trim)),:] )
            ytrain.append( class_count * np.ones( ( Xtrain[-1].shape[0], ) ) )
            class_count += 1

        Xtrain = np.vstack( Xtrain )
        ytrain = np.hstack( ytrain )

        print(Xtrain.shape)
        return Xtrain, ytrain    