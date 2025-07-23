import argparse
import inspect
import os
import pickle
import random
import sys
import time
from collections import deque

import matplotlib
import numpy as np

matplotlib.use( 'QT5Agg')

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.io import savemat
from os import path

from FourierTransformFilter import FourierTransformFilter
from MyoArmband import MyoArmband
from OnsetThreshold import Onset
from RealTimeDataPlot import RealTimeDataPlot
from SenseController import SenseController
from TimeDomainFilter import TimeDomainFilter
from DownSampler import DownSampler

sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ))))

from init import InitGUI
from local_addresses import address

SUBJECT = 1

local_surce_info = address()

if __name__ == '__main__':

    CUE_FLDR = os.path.join( 'cues' )
	
    init = InitGUI( online = True, preload = SUBJECT, visualize = False )
    Parameters = init.load_subject_parameters(SUBJECT)

    EMG_SCALING_FACTOR = 10000.0

    cue_count = ( Parameters['General']['SAMPLING_RATE'] * Parameters['Calibration']['CUE_DURATION'] - Parameters['General']['WINDOW_SIZE'] ) / Parameters['General']['WINDOW_STEP']

    mav_buffer = deque()

    # download cue images
    print( 'Importing cue images...', end = '', flush = True )
    cue_images = {}
    for cue in ['REST',Parameters['Signal Level']['Movement']]:
        img = mpimg.imread( os.path.join( CUE_FLDR, '%s.png' % cue.lower() ) )
        cue_images.update( { cue.lower() : img } )
    print( 'Done!' )

    # create data filters
    print( 'Creating EMG feature filters...', end = '', flush = True )
    td5 = TimeDomainFilter()
    fft = FourierTransformFilter( fftlen = Parameters['General']['FFT_LENGTH'] )
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
        calibrate = np.zeros( ( 3, ), dtype = object ) # training data
        maxlist = []
        cue_fig = plt.figure(figsize = ( 6, 6 ), tight_layout = 3)
        matplotlib.rcParams['font.size'] = 7
        mngr = plt.get_current_fig_manager()
        geom = mngr.window.geometry()
        x, y, dx, dy = geom.getRect()
        mngr.window.setGeometry( 510, 470, 550, 550 )
        plt.ion()
        threshold = 0.0
        for trial in range( 3 ):
            templist = []
            calibrate[ trial ] = {}
            for cue in ['REST',Parameters['Signal Level']['Movement']]: 
                calibrate[ trial ].update( { cue : [] } )
        
            print( '\tTrial %02d...' % ( trial+1 ) )
            for cue in ['REST',Parameters['Signal Level']['Movement']]:
                print( '\t\t%s...' % cue.upper(), end = '\t', flush = True )
                
                # show the rest cue
                plt.imshow( cue_images[ 'rest' ] )
                plt.axis( 'off' )
                plt.show( block = False )

                # wait through rest cue
                if trial > 0:
                    t0 = time.perf_counter()
                    while ( time.perf_counter() - t0 ) < 1:
                        cue_fig.canvas.flush_events()
                        while emg.state is not None:
                            pass
                elif trial == 0:
                    t0 = time.perf_counter()
                    while ( time.perf_counter() - t0 ) < 3:
                        cue_fig.canvas.flush_events()
                        while emg.state is not None:
                            pass


                # set up the cue image
                plt.imshow( cue_images[ cue.lower() ] )
                plt.axis( 'off' )
                plt.show( block = False )
                plt.pause( 0.1 )

                # collect calibration data
                emg_window = []
                feature_count = 0

                if trial > 0:
                    if cue == 'REST':
                        cuenum = cue_count/2
                    elif cue != 'REST':
                        cuenum = cue_count
                elif trial == 0:
                    cuenum = cue_count

                while feature_count < cuenum:
                    data = emg.state
                    if data is not None:
                        emg_window.append( data[:Parameters['General']['DEVICE_ELECTRODE_COUNT']].copy() / EMG_SCALING_FACTOR )
                    
                    if len( emg_window ) == Parameters['General']['WINDOW_SIZE']:
                        win = np.vstack( emg_window )
                        feat = np.hstack( [ td5.filter( win ), fft.filter( win ) ] )[:-Parameters['General']['DEVICE_ELECTRODE_COUNT']]                 
                        emg_window = emg_window[Parameters['General']['WINDOW_STEP']:]
                        mav_plot.add( feat[:8] )
                        #put MAV data into list
                        if trial == 0:
                            if cue == Parameters['Signal Level']['Movement']:
                                maxlist.append(feat[:8])
                        elif trial > 0:
                            if cue == Parameters['Signal Level']['Movement']:
                                templist.append(feat[:8])
                        if cue != 'REST':
                            if np.mean( feat[:Parameters['General']['DEVICE_ELECTRODE_COUNT']] ) > threshold:
                                calibrate[trial][cue].append( feat )
                                feature_count += 1

                                # progress bar
                                percent = feature_count / cue_count
                                print( '', end = '\r', flush = True )
                                print( '\t\t%s...[%s] %.2f%%' % ( cue.upper(), ( int( 10 * percent ) * '=' ) + ( int( 10 * ( 1.0 - percent ) ) * ' ' ), 100 * percent ), end = '\t', flush = True )
                        elif cue == 'REST':
                            if trial == 0:
                                if np.mean( feat[:Parameters['General']['DEVICE_ELECTRODE_COUNT']] ) > threshold:
                                    calibrate[trial][cue].append( feat )
                                    feature_count += 1

                                    # progress bar
                                    percent = feature_count / cue_count
                                    print( '', end = '\r', flush = True )
                                    print( '\t\t%s...[%s] %.2f%%' % ( cue.upper(), ( int( 10 * percent ) * '=' ) + ( int( 10 * ( 1.0 - percent ) ) * ' ' ), 100 * percent ), end = '\t', flush = True )
                            if trial > 0:
                                calibrate[trial][cue].append( feat )
                                feature_count += 1

                                # progress bar
                                percent = feature_count / cue_count
                                print( '', end = '\r', flush = True )
                                print( '\t\t%s...[%s] %.2f%%' % ( cue.upper(), ( int( 10 * percent ) * '=' ) + ( int( 10 * ( 1.0 - percent ) ) * ' ' ), 100 * percent ), end = '\t', flush = True )



                calibrate[trial][cue] = np.vstack( calibrate[trial][cue] )
                print( calibrate[trial][cue].shape )
    
                # calibrate rest data
                if trial == 0:
                    if cue == 'REST':
                        threshold = onset.calib_onset_threshold( calibrate[trial][cue][:,:Parameters['General']['DEVICE_ELECTRODE_COUNT']], gain = Parameters['General']['ELECTRODE_GAIN'] )
                        mav_plot.set_threshold( threshold )
                elif trial > 0:
                    if cue == 'REST':
                        if np.mean(feat[:Parameters['General']['DEVICE_ELECTRODE_COUNT']]) < threshold:
                            threshold = onset.calib_onset_threshold( calibrate[trial][cue][:,:Parameters['General']['DEVICE_ELECTRODE_COUNT']], gain = Parameters['General']['ELECTRODE_GAIN'] )
                            mav_plot.set_threshold( threshold )


    finally:

        dsample = DownSampler()

        Xtrain = []
        Xtrain_full = []
        ytrain = []
        class_count = 0
        for cue in ['REST',Parameters['Signal Level']['Movement']]:
            for trial in range(3):
                Xtrain_full.append(np.vstack( calibrate[trial][cue] ))
                temp_class, Idx = dsample.uniform(calibrate[trial][cue], int( 350 // 2 ) )
                Xtrain.append( temp_class )
                ytrain.append( class_count * np.ones( ( Xtrain[-1].shape[0], ) ) )
            class_count += 1

        Xtrain = np.vstack( Xtrain )
        ytrain = np.hstack( ytrain )

        print( 'Saving data...', end = '', flush = True )
        folder = '.\Data\Subject_%d' % int(SUBJECT)
        folder2 = 'calibration_data'
        filename = 'calibratesignallevel_subject_%d.mat' % int(SUBJECT)
        if not path.exists(folder):
            os.mkdir(folder)
        if not path.exists(folder + '/' + folder2):
            os.mkdir(folder + '/' + folder2)
        filepath = folder + '/' + folder2 + '/' + filename
        if not path.exists(filepath):
            f = open(filepath, "w")
            f.write("")
            f.close()
        savemat( filepath, mdict = { 'lower' : np.mean(threshold), 'upper' : np.mean(maxlist), 'Xtrain': Xtrain, 'ytrain': ytrain } )

        print( 'Done!' )

    emg.stop()
    emg.close()
    mav_plot.close()
    plt.close('all')
