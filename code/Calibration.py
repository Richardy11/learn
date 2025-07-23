import os
import pickle
import sys
import time
from collections import deque
import subprocess

import matplotlib
import numpy as np

matplotlib.use( 'QT5Agg')

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.io import savemat

#sys.path.append( os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), 'code' ) )

from FourierTransformFilter import FourierTransformFilter
from MyoArmband import MyoArmband
from OnsetThreshold import Onset
from RealTimeDataPlot import RealTimeDataPlot
from SenseController import SenseController
from TimeDomainFilter import TimeDomainFilter
from Rescu import RESCU
from FeatureSmoothing import Smoothing

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Init import InitGUI
from local_addresses import address

from os import path

SUBJECT = 1

local_surce_info = address()

if __name__ == '__main__':

    if len(sys.argv) > 1:
        #print( 'SUBJECT:', str(sys.argv[1]) )
        SUBJECT = sys.argv[1]
        autorun = True

    CUE_FLDR = os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ), 'cues' )
	
    init = InitGUI( online = True, preload = SUBJECT, visualize = False )
    Parameters = init.load_subject_parameters(SUBJECT)

    EMG_SCALING_FACTOR = 10000.0

    if Parameters['General']['SOURCE'] == 'Myo Band':
        window_size = Parameters['General']['WINDOW_SIZE']//5
    else:
        window_size = Parameters['General']['WINDOW_SIZE']

    cue_count = ( Parameters['General']['SAMPLING_RATE'] * Parameters['Calibration']['CUE_DURATION'] - window_size ) / Parameters['General']['WINDOW_STEP']
    #cue_count = Parameters['Calibration']['CUE_DURATION'] * Parameters['General']['WINDOW_STEP']

    mav_buffer = deque()

    if Parameters['General']['FL'] == True:
            cuelist = Parameters['Fitts Law']['CUE_LIST']
    else:
        cuelist = Parameters['Calibration']['CUE_LIST']

    # download cue images
    print( 'Importing cue images...', end = '', flush = True )
    cue_images = {}
    for cue in cuelist:
        img = mpimg.imread( os.path.join( CUE_FLDR, '%s.png' % cue.lower() ) )
        cue_images.update( { cue.lower() : img } )
    print( 'Done!' )

    # create data filters
    print( 'Creating EMG feature filters...', end = '', flush = True )
    td5 = TimeDomainFilter()
    if Parameters['General']['FFT_LENGTH'] in ['Custom Adapt']:
        fft = FourierTransformFilter( fftlen = 64 )
    else:
        fft = FourierTransformFilter( fftlen = int(Parameters['General']['FFT_LENGTH']) )
    print( 'Done!' )

    # create onset threshold
    print( 'Creating onset threshold computer...', end = '', flush = True )
    onset =  Onset( onset_scalar = Parameters['Calibration']['CALIB_ONSET_SCALAR'] )
    print( 'Done!' )

    # create EMG interface
    print( 'Creating device interface...', end = '', flush = True )
    if Parameters['General']['SOURCE'] == 'Sense':
        name = local_surce_info.sense_name
        mac = local_surce_info.sense_mac
        emg = SenseController( name = name,
                                    mac = mac,
                                    gain = Parameters['General']['ELECTRODE_GAIN'],
                                    num_electrodes = Parameters['General']['DEVICE_ELECTRODE_COUNT'],
                                    srate = Parameters['General']['SAMPLING_RATE'] )
    elif Parameters['General']['SOURCE'] == 'Myo Band':
        name = local_surce_info.myo_name
        mac = local_surce_info.myo_mac
        emg = MyoArmband( name = name, mac = mac )

    else:
        raise RuntimeError( 'Unsupported EMG interface provided' )
    print( 'Done!' )

    print( 'Create Smoothor...', end = '', flush = True )
    smoothor = Smoothing( Parameters )
    print( 'Done!' )

    print( 'Creating real-time data plot...', end = '', flush = True )
    if Parameters['General']['SOURCE'] == 'Myo Band':
        channel_max = 0.3/50
    elif Parameters['General']['SOURCE'] == 'Sense':
        channel_max = 0.3
    mav_plot = RealTimeDataPlot(    num_channels = Parameters['General']['DEVICE_ELECTRODE_COUNT'],
                                    channel_min = 0,
                                    channel_max = channel_max,
                                    sampling_rate = Parameters['General']['SAMPLING_RATE'] / Parameters['General']['WINDOW_STEP'],
                                    buffer_time = 5,
                                    title = "MAV Signals" )
    print( 'Done!' )
    print( 'Starting data collection...', end = '\n', flush = True )
    emg.run()
    try:
        calibrate = np.zeros( ( Parameters['Calibration']['NUM_OF_TRIALS'], ), dtype = object ) # training data

        cue_fig = plt.figure(figsize = ( 6, 6 ), tight_layout = 3)
        matplotlib.rcParams['font.size'] = 7
        mngr = plt.get_current_fig_manager()
        geom = mngr.window.geometry()
        x, y, dx, dy = geom.getRect()
        mngr.window.setGeometry( 510, 470, 550, 550 )
        plt.ion()
        for trial in range( Parameters['Calibration']['NUM_OF_TRIALS'] ):
            calibrate[ trial ] = {}
            for cue in cuelist: 
                calibrate[ trial ].update( { cue : [] } )
        
            print( '\tTrial %02d...' % ( trial+1 ) )
            threshold = 0.0
            for cue in cuelist:
                print( '\t\t%s...' % cue.upper(), end = '\t', flush = True )
                
                # show the rest cue
                plt.imshow( cue_images[ 'rest' ] )
                plt.axis( 'off' )
                plt.show( block = False )

                # wait through rest cue
                emg_window = []
                feature_count = 0
                smoothor = Smoothing( Parameters )
                t0 = time.perf_counter()
                while ( time.perf_counter() - t0 ) < Parameters['Calibration']['CUE_DELAY']:
                    
                    try:
                        if np.mean( feat_filtered[:Parameters['General']['DEVICE_ELECTRODE_COUNT']] ) > threshold and cue != 'REST':
                            t0 = time.perf_counter()
                    except:
                        pass
                    cue_fig.canvas.flush_events()
                    
                    data = emg.state
                    if data is not None:
                        emg_window.append( data[:Parameters['General']['DEVICE_ELECTRODE_COUNT']].copy() / EMG_SCALING_FACTOR )
                    
                    if len( emg_window ) == window_size:
                        win = np.vstack( emg_window )
                        if Parameters['General']['FFT_LENGTH'] in ['Custom Adapt']:
                            feat = np.hstack( [ td5.filter( win ), fft.filterAdapt( win ) ] )
                        else:
                            feat = np.hstack( [ td5.filter( win ), fft.filter( win ) ] )[:-Parameters['General']['DEVICE_ELECTRODE_COUNT']]

                        feat_filtered = smoothor.filter( feat )[0]

                        emg_window = emg_window[Parameters['General']['WINDOW_STEP']:Parameters['General']['WINDOW_SIZE']]

                # set up the cue image
                plt.imshow( cue_images[ cue.lower() ] )
                plt.axis( 'off' )
                plt.show( block = False )
                plt.pause( 0.1 )

                # collect calibration data
                emg_window = []
                feature_count = 0
                smoothor = Smoothing( Parameters )
                while feature_count < cue_count:
                    data = emg.state
                    if data is not None:
                        emg_window.append( data[:Parameters['General']['DEVICE_ELECTRODE_COUNT']].copy() / EMG_SCALING_FACTOR )
                    
                    if len( emg_window ) == window_size:
                        win = np.vstack( emg_window )
                        if Parameters['General']['FFT_LENGTH'] in ['Custom Adapt']:
                            feat = np.hstack( [ td5.filter( win ), fft.filterAdapt( win ) ] )
                        else:
                            feat = np.hstack( [ td5.filter( win ), fft.filter( win ) ] )[:-Parameters['General']['DEVICE_ELECTRODE_COUNT']]

                        feat_filtered = smoothor.filter( feat )[0]

                        emg_window = emg_window[Parameters['General']['WINDOW_STEP']:]
                        mav_plot.add( feat_filtered[:8] )
                        # print( np.mean( feat[:Parameters['General']['DEVICE_ELECTRODE_COUNT']] ))
                        if np.mean( feat_filtered[:Parameters['General']['DEVICE_ELECTRODE_COUNT']] ) > threshold:
                            calibrate[trial][cue].append( feat_filtered )
                            feature_count += 1

                            # progress bar
                            percent = feature_count / cue_count
                            print( '', end = '\r', flush = True )
                            print( '\t\t%s...[%s] %.2f%%' % ( cue.upper(), ( int( 10 * percent ) * '=' ) + ( int( 10 * ( 1.0 - percent ) ) * ' ' ), 100 * percent ), end = '\t', flush = True )

                calibrate[trial][cue] = np.vstack( calibrate[trial][cue] )
                print( calibrate[trial][cue].shape )

                # calibrate rest data
                if cue == 'REST':
                    threshold = onset.calib_onset_threshold( calibrate[trial][cue][:,:Parameters['General']['DEVICE_ELECTRODE_COUNT']], gain = Parameters['General']['ELECTRODE_GAIN'] )
                    mav_plot.set_threshold( threshold )
                    # print( 'CURRENT THRESHOLD: %.2f' % threshold )

    finally:


        print( 'Saving data...', end = '', flush = True )
        if Parameters['General']['FL'] == True:
            folder = '.\Data\Subject_%d' % int(SUBJECT)
            folder2 = 'calibration_data'
            filename = 'calibratefittslaw_subject_%d.mat' % int(SUBJECT)
        else:
            folder = '.\Data\Subject_%d' % int(SUBJECT)
            folder2 = 'calibration_data'
            filename = 'calibrate_subject_%d.mat' % int(SUBJECT)
        if not path.exists(folder):
            os.mkdir(folder)
        if not path.exists(folder + '/' + folder2):
            os.mkdir(folder + '/' + folder2)
        filepath = folder + '/' + folder2 + '/' + filename
        if not path.exists(filepath):
            f = open(filepath, "w")
            f.write("")
            f.close()
        savemat(filepath, mdict = {'calibrate' : calibrate})

        practice_session = 1
        FILE_DIRECTORY = os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ), 'data', 'Subject_%s' % SUBJECT )
        print( 'Deleting previous session data...', end = '', flush = True)
        while True:
            PREVIOUS_FILE = os.path.join( FILE_DIRECTORY, 'Subject_%s_Practice_%s.pkl' % ( SUBJECT, practice_session ) )
            if os.path.isfile(PREVIOUS_FILE):
                
                practice_session += 1
                os.remove(PREVIOUS_FILE)
            else:
                print( 'Done')
                break

        print( 'Done!' )
    
    if not autorun:
        emg.stop()
        emg.close()
    mav_plot.close()
    plt.close('all')
    if autorun:
        RESCU(subject = SUBJECT, run_mode = 0, source = emg)
    
