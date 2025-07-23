import sys
import warnings

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox,
                             QDoubleSpinBox, QFormLayout, QFrame, QGridLayout,
                             QLabel, QLineEdit, QPushButton, QSizePolicy,
                             QSlider, QSpacerItem, QSpinBox, QTabWidget,
                             QVBoxLayout, QWidget, QWidgetItem)

warnings.simplefilter("ignore", UserWarning)
sys.coinit_flags = 2

import multiprocessing as mp
import os
import pickle
import queue
import threading as th
import copy

import subprocess

from numpy.lib.twodim_base import _trilu_indices_form_dispatcher

sys.path.append( os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), 'code' ) )

from os import path

class DoubleSlider(QSlider):

    # create our our signal that we can connect to if necessary
    doubleValueChanged = pyqtSignal(float)

    def __init__(self, decimals=2, *args, **kargs):
        super(DoubleSlider, self).__init__( *args, **kargs)
        self._multi = 10 ** decimals
        self.setTickPosition(QSlider.TicksBothSides)
        self.setOrientation(Qt.Horizontal)

        self.valueChanged.connect(self.emitDoubleValueChanged)

    def emitDoubleValueChanged(self):
        value = float(super(DoubleSlider, self).value())/self._multi
        self.doubleValueChanged.emit(value)

    def value(self):
        return float(super(DoubleSlider, self).value()) / self._multi

    def setMinimum(self, value):
        return super(DoubleSlider, self).setMinimum(value * self._multi)

    def setMaximum(self, value):
        return super(DoubleSlider, self).setMaximum(value * self._multi)

    def setSingleStep(self, value):
        return super(DoubleSlider, self).setSingleStep(value * self._multi)

    def singleStep(self):
        return float(super(DoubleSlider, self).singleStep()) / self._multi

    def setValue(self, value):
        super(DoubleSlider, self).setValue(int(value * self._multi))

class InitGUI:
    
    GLOBAL_CLASSES_DICTIONARY = {
        'REST'          :   {
            'TYPE'                  :   '',
            'TRANSIENT_GROUP'       :   '',
            'TRANSIENT_ANTAGONIST'  :   ''
        },
        'OPEN'          :   {
            'TYPE'                  :   'GRIP',
            'TRANSIENT_GROUP'       :   '',
            'TRANSIENT_ANTAGONIST'  :   ''
        },
        'POWER'          :   {
            'TYPE'                  :   'GRIP',
            'TRANSIENT_GROUP'       :   'POWER',
            'TRANSIENT_ANTAGONIST'  :   'OPEN'
        },
        'TRIPOD'          :   {
            'TYPE'                  :   'GRIP',
            'TRANSIENT_GROUP'       :   'TRIPOD',
            'TRANSIENT_ANTAGONIST'  :   'OPEN'
        },
        'KEY'          :   {
            'TYPE'                  :   'GRIP',
            'TRANSIENT_GROUP'       :   'KEY',
            'TRANSIENT_ANTAGONIST'  :   'OPEN'
        },
        'INDEX'          :   {
            'TYPE'                  :   'GRIP',
            'TRANSIENT_GROUP'       :   'INDEX',
            'TRANSIENT_ANTAGONIST'  :   'OPEN'
        },
        'PINCH'          :   {
            'TYPE'                  :   'GRIP',
            'TRANSIENT_GROUP'       :   'PINCH',
            'TRANSIENT_ANTAGONIST'  :   'OPEN'
        },
        'PALM DOWN'          :   {
            'TYPE'                  :   'WRIST',
            'TRANSIENT_GROUP'       :   'WRIST',
            'TRANSIENT_ANTAGONIST'  :   'PALM UP'
        },
        'PALM UP'          :   {
            'TYPE'                  :   'WRIST',
            'TRANSIENT_GROUP'       :   'WRIST',
            'TRANSIENT_ANTAGONIST'  :   'PALM DOWN'
        },
        'FLEXION'          :   {
            'TYPE'                  :   'WRIST',
            'TRANSIENT_GROUP'       :   'WRIST_FLEXION',
            'TRANSIENT_ANTAGONIST'  :   'EXTENSION'
        },
        'EXTENSION'          :   {
            'TYPE'                  :   'WRIST',
            'TRANSIENT_GROUP'       :   'WRIST_FLEXION',
            'TRANSIENT_ANTAGONIST'  :   'FLEXION'
        },
        'ELBOW BEND'          :   {
            'TYPE'                  :   'ELBOW',
            'TRANSIENT_GROUP'       :   'ELBOW',
            'TRANSIENT_ANTAGONIST'  :   'ELBOW EXTEND'
        },
        'ELBOW EXTEND'          :   {
            'TYPE'                  :   'ELBOW',
            'TRANSIENT_GROUP'       :   'ELBOW',
            'TRANSIENT_ANTAGONIST'  :   'ELBOW BEND'
        }
    }
    
    GLOBAL_CLASSES_LIST = []
    for key, value in enumerate(GLOBAL_CLASSES_DICTIONARY.keys()):
        GLOBAL_CLASSES_LIST.append( value )
    
    GLOBAL_TRANSIENT_GROUPS_LIST = []
    GLOBAL_TRANSIENT_CALIBRATION_CLASSES_DICTIONARY = {}
    for key, value in enumerate(GLOBAL_CLASSES_DICTIONARY.keys()):
        if GLOBAL_CLASSES_DICTIONARY[value]['TRANSIENT_GROUP'] != '' and GLOBAL_CLASSES_DICTIONARY[value]['TRANSIENT_GROUP'] not in GLOBAL_TRANSIENT_GROUPS_LIST:
            GLOBAL_TRANSIENT_GROUPS_LIST.append( GLOBAL_CLASSES_DICTIONARY[value]['TRANSIENT_GROUP'] )
            GLOBAL_TRANSIENT_CALIBRATION_CLASSES_DICTIONARY[GLOBAL_CLASSES_DICTIONARY[value]['TRANSIENT_GROUP']] = value


    PARAMETERS = {          'General':      {'SUBJECT_NAME' : 1, 'DEVICE_ELECTRODE_COUNT' : 8, 'SOURCE': 'Sense', 'SAMPLING_RATE': 1000, 'WINDOW_SIZE': 200, 'WINDOW_STEP': 10, 'FFT_LENGTH': '32', 'ELECTRODE_GAIN' : 4, 'FL' : False},

                            'Calibration':  {'NUM_OF_TRIALS' : 1, 'CUE_DELAY': 2, 'CUE_DURATION': 3, 'CALIB_METHOD' : 'Onset Calibration', 'CALIB_ONSET_SCALAR' : 1.5, 'CUE_LIST' : ['REST'], 'ALL_CLASSES': ['REST']},

                            'Classification': { 'CLUSTER_TRAINING_SAMPLES' : 50, 'UPDATE_K' : 20, 'ONSET_DATA_CACHING' : False, 'RESCU': True, 'SEGMENTATION_PLOT': False, 'REGENERATE_CLASSIFIER': True,
                                                'CLASSIFIER': 'Spatial', 'FEATURE_FILTER': 'Linear', 'FILTER_LENGTH': 20, 'EXPONENTIAL_FILTER_SCALE': 0.25, 'BEZIER_PROJECTION_RATIO': 0.85,
                                                'EASRC_NEIGHBORHOOD' : 3, 'CONFIDENCE_REJECTION' : False,
                                                'ONSET_CLASSIFICATION' : False,'ONSET_THRESHOLD' : True , 'ONSET_SCALAR' : 1.3, 'THRESHOLD_RATIO': 0.5},

                            'Output Filtering': { 'OUTPUT_FILTER' : 'First Over Filter', 'UNIFORM_FILTER_SIZE' : 3, 'SMOOTHING_FILTER_LENGTH' : 1, 'INDIV_SENS' : False, 'GLOBAL_SENS' : 10 },
                            
                            'Proportional Control': { 'VELOCITY_RAMP_ENABLED' : True, 'VELOCITY_RAMP_INCREMENT' : 1, 'VELOCITY_RAMP_DECREMENT' : 2, 'VELOCITY_RAMP_SIZE' : 50,
                                                    'PROPORTIONAL_ENABLED' : True,'PROPORTIONAL_METHOD': 'Spatial' , 'PROP_SMOOTH': 20 },

                            'Signal Outputting': {'SHOW_CONTROL': True, 'SHOW_MAV': True, 'SHOW_REVIEWER': True,'SHOW_VELOCITY': True, 'SHOW_MYOTRAIN': False, 'PROSTHESIS': False,  'TERMINAL_DEVICE': 'ProETD'},

                            'Signal Level': {'Input': 'Pattern Recognition','Electrode': 0,'Movement': 'POWER','Hold Time': 0.5, 'Timeout': 10, 'FILTER_LENGTH': 40, 'Smoothing': 50, 'PROPORTIONAL_METHOD': 'Spatial', 'File': 'signallevel.csv','Trials': 1,'Rest Time': 3},

                            'Fitts Law' :   {'Movement Setting': 'Velocity','Trials': 1, 'Trial Time' : 15, 'Start Cue': 0, 'Hold Time': 0.5, 'DoF': {}, 'File': 'fittslaw.csv', 'CUE_LIST': ['REST'], 'GRIP_CUE_LIST': [], 'Method': '1'},

                            'Misc.': {'EXPERIMENT' : False, 'CARRY_UPDATES' : True, 'NUM_CUES' : 12, 'EXIT_BUTTON': False }
                }

    
    WIDGET_PROPERTIES = {  'General':      {'SUBJECT_NAME' :           {'type': 'integer', 'label': 'Subject Name', 'values': [], 'min': 1, 'max': 999, 'step': 1, 'visible': [], 'online' : False},
                                            'DEVICE_ELECTRODE_COUNT' : {'type': 'integer', 'label': 'Electrode Count', 'values': [8, 8], 'min': 2, 'max': 128, 'step': 1, 'visible': [], 'visible': ['General','SOURCE','Sense'], 'online' : False, 'dependency': ['General', 'SOURCE']},
                                            'SOURCE':                  {'type': 'dropdown', 'label': 'Source', 'values': ['Sense', 'Myo Band', 'HDEMG', 'SMG'], 'min': 0, 'max': 0, 'step': 0, 'visible': [], 'online' : False},
                                            'SAMPLING_RATE':           {'type': 'hidden', 'label': '', 'values': [1000, 200], 'min': 0, 'max': 0, 'step': 0, 'visible': [], 'online' : False, 'dependency': ['General', 'SOURCE']},                                            
                                            'WINDOW_SIZE':             {'type': 'integerslider', 'label': 'Window Size', 'values': [], 'min': 200, 'max': 1000, 'step': 100, 'visible': [], 'online' : True, 'dependency': ['General', 'SOURCE']},
                                            'WINDOW_STEP':             {'type': 'hidden', 'label': '', 'values': [10, 2], 'min': 0, 'max': 0, 'step': 0, 'visible': [], 'online' : False, 'dependency': ['General', 'SOURCE']},
                                            'FFT_LENGTH':              {'type': 'dropdown', 'label': 'FFT Length', 'values': ['32', '64', 'Custom Adapt'], 'min': 2, 'max': 1024, 'step': 1, 'visible': [], 'online' : False},
                                            'ELECTRODE_GAIN' :         {'type': 'integer', 'label': 'Electrode Gain', 'values': [], 'min': 1, 'max': 7, 'step': 1, 'visible': ['General','SOURCE', 'Sense'], 'online' : False},
                                            'LOAD' :                   {'type': 'button', 'label': 'LOAD PARAMETERS', 'values': [], 'min': 1, 'max': 7, 'step': 1, 'visible': [], 'online' : False},
                                            'DELETE' :                 {'type': 'button', 'label': 'DELETE UPDATE DATA', 'values': [], 'min': 1, 'max': 7, 'step': 1, 'visible': [], 'online' : False},
                                            'LAUNCH_RESCU':            {'type': 'button', 'label': 'LAUNCH RESCU', 'values': [], 'min': 0, 'max': 0, 'step': 0, 'visible': [], 'online': False} },
                                            #'LAUNCH_SEPREP':           {'type': 'button', 'label': 'LAUNCH SEPARABILITY / REPEATABILITY TEST', 'values': [], 'min': 0, 'max': 0, 'step': 0, 'visible': [], 'online': False} },

                            'Calibration':  {'NUM_OF_TRIALS' :     {'type': 'integer', 'label': 'Number of Trials', 'values': [], 'min': 1, 'max': 10, 'step': 1, 'visible': [], 'online' : False},
                                            'CUE_DELAY':           {'type': 'integer', 'label': 'Cue Delay', 'values': [], 'min': 1, 'max': 10, 'step': 1, 'visible': [], 'online' : False},
                                            'CUE_DURATION':        {'type': 'integer', 'label': 'Cue Duration', 'values': [], 'min': 1, 'max': 10, 'step': 1, 'visible': [], 'online' : False},
                                            'CALIB_METHOD' :       {'type': 'dropdown', 'label': 'Calibration Method', 'values': ['Onset Calibration', 'Cued Onset', 'Cue Trim'], 'min': 0, 'max': 0, 'step': 0, 'visible': [], 'online' : False},
                                            'CALIB_ONSET_SCALAR' : {'type': 'floatslider', 'label': 'Onset Scalar', 'values': [], 'min': 1.1, 'max': 10, 'step': 0.1, 'visible': [], 'online' : False},
                                            'SPLITTER':            {'type': 'splitter', 'label': 'SPLITTER'},
                                            'Calibrate':         {'type': 'button', 'label': 'CALIBRATE', 'values': [], 'min': 0, 'max': 0, 'step': 0, 'visible': [], 'online': False} },

                            'Classification': { 'ONSET_SCALAR' :           {'type': 'floatslider', 'label': 'Onset Scalar', 'values': [], 'min': 1.1, 'max': 10, 'step': 0.1, 'visible': ['Classification','CLASSIFIER',['EASRC']], 'online' : True},
                                                'THRESHOLD_RATIO' :        {'type': 'floatslider', 'label': 'Threshold ratio', 'values': [], 'min': 0, 'max': 1, 'step': 0.05, 'visible': ['Classification','CLASSIFIER', ['Spatial', 'Simultaneous']], 'online' : True},
                                                'CLUSTER_TRAINING_SAMPLES' : {'type': 'integer', 'label': 'Cluster Training Samples', 'values': [], 'min': 1, 'max': 10000, 'step': 1, 'visible': [], 'online' : False},
                                                'EASRC_NEIGHBORHOOD' :     {'type': 'integer', 'label': 'EASRC Neighborhood', 'values': [], 'min': 1, 'max': 10, 'step': 1, 'visible': ['Classification','CLASSIFIER','EASRC'], 'online' : True},
                                                'ONSET_CLASSIFICATION' :   {'type': 'checkbox', 'label': 'Onset Classification', 'values': [], 'min': 1, 'max': 999, 'step': 1, 'visible': [], 'online' : True},
                                                'ONSET_THRESHOLD' :        {'type': 'checkbox', 'label': 'Onset Threshold', 'values': [], 'min': 1, 'max': 999, 'step': 1, 'visible': [], 'online' : True},
                                                'SPLITTER':                {'type': 'splitter', 'label': 'SPLITTER'},
                                                'RESCU':                   {'type': 'checkbox', 'label': 'RESCU Functionality', 'values': [], 'min': 1, 'max': 999, 'step': 1, 'visible': [], 'online' : True},
                                                'SEGMENTATION_PLOT':       {'type': 'checkbox', 'label': 'Segmentation Plot', 'values': [], 'min': 1, 'max': 999, 'step': 1, 'visible': ['Classification','RESCU', True], 'online' : True},
                                                'REGENERATE_CLASSIFIER':   {'type': 'checkbox', 'label': 'Regenerate Classifier', 'values': [], 'min': 1, 'max': 999, 'step': 1, 'visible': ['Classification','RESCU', True], 'online' : True},
                                                'UPDATE_K' :               {'type': 'integerslider', 'label': 'Update Percentage', 'values': [], 'min': 1, 'max': 100, 'step': 5, 'visible': ['Classification','RESCU', True], 'online' : True},
                                                'ONSET_DATA_CACHING' :     {'type': 'checkbox', 'label': 'Onset Data Caching', 'values': [], 'min': 1, 'max': 999, 'step': 1, 'visible': ['Classification','RESCU', True], 'online' : True},
                                                'CONFIDENCE_REJECTION' :   {'type': 'checkbox', 'label': 'Confidence Rejection', 'values': [], 'min': 1, 'max': 999, 'step': 1, 'visible': ['Classification','RESCU', True], 'online' : True},
                                                'SPLITTER':                {'type': 'splitter', 'label': 'SPLITTER'},
                                                'CLASSIFIER':              {'type': 'dropdown', 'label': 'Classifier', 'values': ['EASRC', 'Spatial', 'Simultaneous'], 'min': 0, 'max': 0, 'step': 0, 'visible': [], 'online' : True},
                                                'FEATURE_FILTER':          {'type': 'dropdown', 'label': 'Pre-classification Filter', 'values': ['Linear', 'Exponential', 'Bezier'], 'min': 0, 'max': 0, 'step': 0, 'visible': [], 'online' : True},
                                                'FILTER_LENGTH' :          {'type': 'integerslider', 'label': 'Filter Length', 'values': [], 'min': 1, 'max': 100, 'step': 5, 'visible': [], 'online' : True, 'dependency': ['Classification', 'FEATURE_FILTER']},
                                                'EXPONENTIAL_FILTER_SCALE':{'type': 'floatslider', 'label': 'Exponential Filter Scale', 'values': [], 'min': 0, 'max': 1, 'step': 0.05, 'visible': ['Classification', 'FEATURE_FILTER', 'Exponential'], 'online' : True},
                                                'BEZIER_PROJECTION_RATIO' :{'type': 'floatslider', 'label': 'Bezier Projection Ratio', 'values': [], 'min': 0, 'max': 1, 'step': 0.05, 'visible': ['Classification', 'FEATURE_FILTER', 'Bezier'], 'online' : True} },
                                                

                            'Output Filtering': {   'OUTPUT_FILTER' :          {'type': 'dropdown', 'label': 'Output Filter', 'values': ['First Over Filter', 'Uniform Filter'], 'min': 0, 'max': 0, 'step': 0, 'visible': [], 'online' : True},
                                                    'UNIFORM_FILTER_SIZE' :    {'type': 'integerslider', 'label': 'Uniform Filter Size', 'values': [], 'min': 1, 'max': 100, 'step': 5, 'visible': ['Output Filtering','OUTPUT_FILTER', 'Uniform Filter'], 'online' : True},
                                                    'SMOOTHING_FILTER_LENGTH': {'type': 'integerslider', 'label': 'Output Smoothing Length', 'values': [], 'min': 1, 'max': 50, 'step': 1, 'visible': ['Output Filtering','OUTPUT_FILTER', 'First Over Filter'], 'online' : True},
                                                    'INDIV_SENS' :             {'type': 'checkbox', 'label': 'Individual Sensitivity', 'values': [], 'min': 1, 'max': 999, 'step': 1, 'visible': [], 'online' : True},
                                                    'GLOBAL_SENS' :            {'type': 'integerslider', 'label': 'Global Sensitivity', 'values': [], 'min': 5, 'max': 50, 'step': 1, 'visible': ['Output Filtering', 'INDIV_SENS', False], 'online' : True},
                                                    'SPLITTER':                {'type': 'splitter', 'label': 'SPLITTER'} },
                            
                            'Proportional Control': { 'VELOCITY_RAMP_ENABLED' :     {'type': 'checkbox', 'label': 'Velocity Ramp', 'values': [], 'min': 1, 'max': 999, 'step': 1, 'visible': [], 'online' : True},
                                                    'PROPORTIONAL_ENABLED' :        {'type': 'checkbox', 'label': 'Proportional Conrol', 'values': [], 'min': 1, 'max': 999, 'step': 1, 'visible': [], 'online' : True},
                                                    'PROPORTIONAL_METHOD' :         {'type': 'dropdown', 'label': 'Proportional Method', 'values': ['Spatial', 'Amplitude'], 'min': 0, 'max': 0, 'step': 0, 'visible': [], 'online' : True},
                                                    'PROP_SMOOTH':                  {'type': 'integerslider', 'label': 'Proportional Smoothing', 'values': [], 'min': 5, 'max': 100, 'step': 5, 'visible': ['Proportional Control','PROPORTIONAL_ENABLED', True], 'online' : True} },

                            'Signal Outputting': {  'SHOW_CONTROL':  {'type': 'checkbox', 'label': 'Show Control Outputs', 'values': [], 'min': 1, 'max': 999, 'step': 1, 'visible': [], 'online' : True},
                                                    'SHOW_MAV':      {'type': 'checkbox', 'label': 'Show MAV Signals', 'values': [], 'min': 1, 'max': 999, 'step': 1, 'visible': [], 'online' : True},
                                                    'SHOW_REVIEWER': {'type': 'checkbox', 'label': 'Show Reviewer', 'values': [], 'min': 1, 'max': 999, 'step': 1, 'visible': [], 'online' : True},
                                                    'SHOW_VELOCITY': {'type': 'checkbox', 'label': 'Show Velocity Controls', 'values': [], 'min': 1, 'max': 999, 'step': 1, 'visible': [], 'online' : True},
                                                    'SHOW_MYOTRAIN': {'type': 'checkbox', 'label': 'Show MyoTrain', 'values': [], 'min': 1, 'max': 999, 'step': 1, 'visible': [], 'online' : True},
                                                    'SPLITTER':      {'type': 'splitter', 'label': 'SPLITTER'},
                                                    'PROSTHESIS':       {'type': 'checkbox', 'label': 'Prosthesis Output', 'values': [], 'min': 1, 'max': 999, 'step': 1, 'visible': [], 'online' : True},
                                                    'TERMINAL_DEVICE':  {'type': 'dropdown', 'label': 'Terminal Device', 'values': ['ProETD', 'BeBionic', 'Taska'], 'min': 1, 'max': 999, 'step': 1, 'visible': ['Signal Outputting','PROSTHESIS', True], 'online' : False} },

                            'Signal Level': {   'Input': {'type': 'dropdown', 'label': 'Movement Input', 'values': ['Single Electrode', 'Pattern Recognition'], 'min': 0, 'max': 0, 'step': 0, 'visible': [], 'online':True},
                                                'Electrode': {'type': 'integer', 'label': 'Electrode ID', 'values': [],'min':0,'max':7,'step': 1,'visible': ['Signal Level','Input','Single Electrode'],'online':True},
                                                'Movement': {'type': 'dropdown', 'label': 'Movement', 'values': ['OPEN', 'POWER', 'PALM UP', 'PALM DOWN'], 'min': 0, 'max': 0, 'step': 0, 'visible': [], 'online': True},
                                                'Trials' : {'type' : 'integer', 'label': 'Trials', 'values': [], 'min': 1, 'max': 10, 'step': 1, 'visible': [], 'online': True},
                                                'Hold Time': {'type': 'floatslider', 'label': 'Hold Time (seconds)', 'values': [], 'min': 0.1, 'max': 2, 'step': 0.1, 'visible': [], 'online': True},
                                                'Timeout' : {'type': 'integerslider', 'label': 'Timeout (seconds)', 'values': [], 'min': 0, 'max': 20, 'step': 1, 'visible': [], 'online': True},
                                                'Rest Time': {'type': 'integerslider', 'label': 'Rest Time (seconds)', 'values': [], 'min': 0, 'max': 10, 'step': 1, 'visible': [], 'online': True},
                                                'PROPORTIONAL_METHOD' : {'type': 'dropdown', 'label': 'Proportional Method', 'values': ['Spatial', 'Amplitude'], 'min': 0, 'max': 0, 'step': 0, 'visible': [], 'online' : True},
                                                'FILTER_LENGTH' : {'type': 'integerslider', 'label': 'Filter Length', 'values': [], 'min': 1, 'max': 100, 'step': 5, 'visible': [], 'online' : True},#, 'dependency': ['Classification', 'FEATURE_FILTER']},
                                                'Smoothing' : {'type': 'integerslider','label': 'Smoothing Level', 'values': [], 'min': 1, 'max': 100, 'step':1, 'visible': [], 'online': True},
                                                'File' : {'type': 'dropdown', 'label': 'File Name', 'values': ['signallevel.csv'], 'min': 0, 'max': 0, 'step': 0, 'visible': [], 'online': True},
                                                'Targets' : {'type': 'button', 'label': 'CHOOSE TARGETS', 'values': [], 'min':0,'max':0,'step':0,'visible':[],'online': True},
                                                'CalibrateSL': {'type': 'button', 'label': 'CALIBRATE', 'values': [], 'min': 0, 'max': 0, 'step': 0, 'visible': [], 'online': True},
                                                'SignalLevel' : {'type': 'button', 'label': 'RUN SIGNAL LEVEL TEST', 'values': [], 'min': 1, 'max': 7, 'step': 1, 'visible': [], 'online' : False} },

                            'Fitts Law': {   'Movement Setting' : {'type': 'dropdown', 'label': 'Movement Setting', 'values': ['Proportional','Velocity'], 'min': 0, 'max': 0, 'step': 0, 'visible': [], 'online': True},
                                             'Trials': {'type': 'integer', 'label': 'Repetitions', 'values': [],'min':1,'max':100,'step': 1,'visible': [],'online':True},
                                             'Start Cue': {'type': 'integer', 'label': 'Start Cue', 'values': [],'min':0,'max':100,'step': 1,'visible': [],'online':True},
                                             'Trial Time': {'type': 'integerslider', 'label': 'Trial Time (seconds)', 'values': [], 'min': 10, 'max': 60, 'step': 1, 'visible': [], 'online': True},
                                             'Hold Time': {'type': 'floatslider', 'label': 'Hold Time (seconds)', 'values': [], 'min': 0.1, 'max': 3, 'step': 0.1, 'visible': [], 'online': True},
                                             'Method' :       {'type': 'dropdown', 'label': 'Test Method', 'values': ['1', '2'], 'min': 0, 'max': 0, 'step': 0, 'visible': [], 'online' : True},
                                             'DoF' : {},
                                             'File' : {'type': 'dropdown', 'label': 'File Name', 'values': ['fittslaw.csv'], 'min': 0, 'max': 0, 'step': 0, 'visible': [], 'online': True},
                                             'TargetsFL' : {'type': 'button', 'label': 'CHOOSE TARGETS', 'values': [], 'min':0,'max':0,'step':0,'visible':[],'online': True},
                                             'CalibrateFL': {'type': 'button', 'label': 'CALIBRATE', 'values': [], 'min': 0, 'max': 0, 'step': 0, 'visible': [], 'online': True},
                                             'FittsLaw' : {'type': 'button', 'label': 'RUN FITTS LAW TEST', 'values': [], 'min': 1, 'max': 7, 'step': 1, 'visible': [], 'online' : True}
                                        },


                            'Misc.': {  'EXPERIMENT' :    {'type': 'checkbox', 'label': 'Experiment', 'values': [], 'min': 1, 'max': 999, 'step': 1, 'visible': [], 'online' : False},
                                        'CARRY_UPDATES' : {'type': 'checkbox', 'label': 'Carry Updates', 'values': [], 'min': 1, 'max': 999, 'step': 1, 'visible': [], 'online' : False},
                                        'NUM_CUES' :      {'type': 'integer', 'label': 'Number of Cues in Experiment', 'values': [], 'min': 1, 'max': 100, 'step': 1, 'visible': [], 'online' : False} }
                }
    
    
    for i in range(len(GLOBAL_CLASSES_LIST)):
        
        if i > 0:
            
            if GLOBAL_CLASSES_LIST[i] in ['OPEN', 'POWER', 'PALM DOWN', 'PALM UP']:
                PARAMETERS['Calibration'][GLOBAL_CLASSES_LIST[i]] = True
            else:
                PARAMETERS['Calibration'][GLOBAL_CLASSES_LIST[i]] = False
            
            PARAMETERS['Output Filtering'][GLOBAL_CLASSES_LIST[i].replace(" ", "_") + '_SENS'] = 10
            
            PARAMETERS['Proportional Control'][GLOBAL_CLASSES_LIST[i].replace(" ", "_") + '_VEL_LOW'] = [0, 0]
            PARAMETERS['Proportional Control'][GLOBAL_CLASSES_LIST[i].replace(" ", "_") + '_VEL_HIGH'] = [0, 1]

            WIDGET_PROPERTIES['Calibration'][GLOBAL_CLASSES_LIST[i]] = {'type': 'checkbox', 'label': ' '.join(elem.capitalize() for elem in GLOBAL_CLASSES_LIST[i].split()) + (' Grip' if GLOBAL_CLASSES_LIST[i] == 'KEY' else ''), 'values': [], 'min': 1, 'max': 999, 'step': 1, 'visible': [], 'online' : False, 'movement' : True}
        
        else:
            
            PARAMETERS['Output Filtering'][GLOBAL_CLASSES_LIST[i].replace(" ", "_") + '_SENS'] = 1


    tmp_dict = {}
    for key, value in enumerate(WIDGET_PROPERTIES['Calibration'].keys()):
        tmp_dict[value] = WIDGET_PROPERTIES['Calibration'][value]
        if value == 'SPLITTER':
            for i in range(len(GLOBAL_CLASSES_LIST)):
                if i > 0:
                    tmp_dict[GLOBAL_CLASSES_LIST[i]] = {'type': 'checkbox', 'label': ' '.join(elem.capitalize() for elem in GLOBAL_CLASSES_LIST[i].split()) + (' Grip' if GLOBAL_CLASSES_LIST[i] == 'KEY' else ''), 'values': [], 'min': 1, 'max': 999, 'step': 1, 'visible': [], 'online' : False, 'movement' : True}
    
    WIDGET_PROPERTIES['Calibration'] = tmp_dict


    tmp_dict = {}
    for key, value in enumerate(WIDGET_PROPERTIES['Output Filtering'].keys()):
        tmp_dict[value] = WIDGET_PROPERTIES['Output Filtering'][value]
        if value == 'INDIV_SENS':
            tmp_dict[GLOBAL_CLASSES_LIST[0].replace(" ", "_") + '_SENS'] = {'type': 'integerslider', 'label': ' '.join(elem.capitalize() for elem in GLOBAL_CLASSES_LIST[0].split()) + ' Sensitivity', 'values': [], 'min': 1, 'max': 50, 'step': 1, 'visible': [], 'online' : True}
        if value == 'GLOBAL_SENS':
            for i in range(len(GLOBAL_CLASSES_LIST)):
                if i > 0:
                    tmp_dict[GLOBAL_CLASSES_LIST[i].replace(" ", "_") + '_SENS'] = {'type': 'integerslider', 'label': ' '.join(elem.capitalize() for elem in GLOBAL_CLASSES_LIST[i].split()) + (' Grip' if GLOBAL_CLASSES_LIST[i] == 'KEY' else '') + ' Sensitivity', 'values': [], 'min': 5, 'max': 50, 'step': 1, 'visible': ['Output Filtering', 'INDIV_SENS', True], 'online' : True}
    
    WIDGET_PROPERTIES['Output Filtering'] = tmp_dict

    
    for i in range(len(GLOBAL_TRANSIENT_GROUPS_LIST)):

        PARAMETERS['Output Filtering']['TRANSIENT_' + GLOBAL_TRANSIENT_GROUPS_LIST[i].replace(" ", "_")] = False
        WIDGET_PROPERTIES['Output Filtering']['TRANSIENT_' + GLOBAL_TRANSIENT_GROUPS_LIST[i].replace(" ", "_")] = {'type': 'checkbox', 'label': ' '.join(elem.capitalize() for elem in GLOBAL_TRANSIENT_GROUPS_LIST[i].replace("_", " ").split()) + ' Transient', 'values': [], 'min': 1, 'max': 999, 'step': 1, 'visible': ['Calibration',GLOBAL_TRANSIENT_CALIBRATION_CLASSES_DICTIONARY[GLOBAL_TRANSIENT_GROUPS_LIST[i]], True], 'online' : True}


    class MainWindow(QWidget):
        def fill_DoF(self):
            for i in range(4):
                InitGUI.WIDGET_PROPERTIES['Fitts Law']['DoF'][i] = {    'MOVEMENT_TYPE' : {'type': 'dropdown', 'label': str(i) + ' DoF', 'values': ['Off','Hand','Wrist','Elbow'], 'min': 0, 'max': 0, 'step': 0, 'visible': [], 'online': True},
                                                                        'POWER' : {'type': 'checkbox', 'label': 'Power', 'values': [], 'min': 1, 'max': 999, 'step': 1, 'visible': ['Fitts Law','DoF', i, 'MOVEMENT_TYPE','Hand'], 'online' : True, 'movement' : True},
                                                                        'TRIPOD' : {'type': 'checkbox', 'label': 'Tripod', 'values': [], 'min': 1, 'max': 999, 'step': 1, 'visible': ['Fitts Law','DoF', i, 'MOVEMENT_TYPE','Hand'], 'online' : True, 'movement' : True},
                                                                        'KEY' : {'type': 'checkbox', 'label': 'Key', 'values': [], 'min': 1, 'max': 999, 'step': 1, 'visible': ['Fitts Law','DoF', i, 'MOVEMENT_TYPE','Hand'], 'online' : True, 'movement' : True},
                                                                        'INDEX' : {'type': 'checkbox', 'label': 'Index', 'values': [], 'min': 1, 'max': 999, 'step': 1, 'visible': ['Fitts Law','DoF', i, 'MOVEMENT_TYPE','Hand'], 'online' : True, 'movement' : True},
                                                                        'PINCH' : {'type': 'checkbox', 'label': 'Pinch', 'values': [], 'min': 1, 'max': 999, 'step': 1, 'visible': ['Fitts Law','DoF', i, 'MOVEMENT_TYPE','Hand'], 'online' : True, 'movement' : True},
                                                                        'WRIST_DOF' : {'type' : 'dropdown', 'label': 'Wrist DoF', 'values': ['ROTATION','FLEXION'], 'min': 0, 'max': 0, 'step': 0, 'visible': ['Fitts Law','DoF', i, 'MOVEMENT_TYPE','Wrist'], 'online': True},
                                                                        'ELBOW' : {'type' : 'dropdown', 'label': 'Elbow Mov', 'values': ['ELBOW BEND','ELBOW EXTEND'], 'min': 0, 'max': 0, 'step': 0, 'visible': ['Fitts Law','DoF', i, 'MOVEMENT_TYPE','Elbow'], 'online': True}
                                                                    }
                InitGUI.PARAMETERS['Fitts Law']['DoF'][i] = {           'MOVEMENT_TYPE' : 'Off',
                                                                        'POWER' : False,
                                                                        'TRIPOD' : False,
                                                                        'KEY' : False,
                                                                        'INDEX' : False,
                                                                        'PINCH' : False,
                                                                        'WRIST_DOF' : 'ROTATION',
                                                                        'WRIST_MOV' : 'PALM DOWN',
                                                                        'ELBOW' : 'ELBOW BEND'
                                                                    }

        def __init__(self, online, preload):
            super().__init__()
            self.first = True
            self.online = online
            self._changed = False
            self._last_changed_param = []

            if self.online:
                subject = preload
                with open( f"Data\Subject_{subject}\RESCU_Protocol_Subject_{subject}.pkl", 'rb' ) as f:
                    Protocol_Data = pickle.load(f)

                self._parameters = Protocol_Data
            else:
                self._parameters = InitGUI.PARAMETERS
            self.fill_DoF()
            self._widget_properties = InitGUI.WIDGET_PROPERTIES

            self.redrawingInProgress = False
            # set up window
            self.setGeometry( 1065, 470, 650, 400 )
            self.setWindowTitle( "Initialization GUI" )
            self.resize(650, 400)

            self.plotGUI()
        
        def clearLayout(self, layout):
            for i in reversed(range(layout.count())):
                item = layout.itemAt(i)

                if isinstance(item, QWidgetItem):
                    item.widget().setParent(None)
                elif isinstance(item, QSpacerItem):
                    pass
                else:
                    self.clearLayout(item.layout())

        def plotGUI(self):

            # Create a top-level layout
            self.framelayout = QVBoxLayout()
            self.setLayout(self.framelayout)
            # Create the tab widget with two tabs
            tabs = QTabWidget()
            
            self.widgets = {}
            self.labels = {}
            self.layouts = {}
            self.tabs_available = []
            for key, value in self._parameters.items():
                self.widgets[key] = {}
                self.labels[key] = {}
                self.layouts[key] = {}
                if self.TabUI(key):
                    self.tabs_available.append(key)
                    tabs.addTab(self.currentTab, key)

            self.framelayout.addWidget(tabs)
            self._on_load_button()

        def draw_widgets(self, key, widget_value, parameters_value, labels_value, widget_data_value, key_stack):
            try: 
                if widget_value['type'] == 'textfield':
                    pass
                elif widget_value['type'] == 'integer':
                    temp_widget = QSpinBox()
                    temp_widget.setMinimum(widget_value['min'])
                    temp_widget.setMaximum(widget_value['max'])
                    temp_widget.setValue(int(parameters_value))
                    temp_widget.valueChanged.connect( lambda *args, keys=key_stack : self._on_change_callback(keys) )
                    if self.isVisible(widget_value):
                        self.layouts[key_stack[0]].addRow(widget_value['label'] + ': ', temp_widget )
                        self.hasWidgets = True
                        
                elif widget_value['type'] == 'dropdown':
                    temp_widget = QComboBox()

                    list_to_use = copy.deepcopy(widget_value['values'])
                    if key_stack[0] == 'Fitts Law' and type(key_stack[-2]) == int:
                        hand_act = 0
                        elbow_act = 0 
                        wrist_act = 0                        
                        for i in range(4):
                            if self._parameters[key_stack[0]][key_stack[1]][i]['MOVEMENT_TYPE'] != 'Off':
                                if self._parameters[key_stack[0]][key_stack[1]][i]['MOVEMENT_TYPE'] == 'Wrist':
                                    wrist_act += 1
                                elif self._parameters[key_stack[0]][key_stack[1]][i]['MOVEMENT_TYPE'] == 'Hand':
                                    hand_act += 1
                                elif self._parameters[key_stack[0]][key_stack[1]][i]['MOVEMENT_TYPE'] == 'Elbow':
                                    elbow_act += 1
                                    
                        if hand_act > 0: 
                            try: list_to_use.remove('Hand') 
                            except: pass
                        if wrist_act > 1: 
                            try: list_to_use.remove('Wrist') 
                            except: pass
                        if elbow_act > 0: 
                            try: list_to_use.remove('Elbow') 
                            except: pass

                    temp_widget.addItems(list_to_use)
                    temp_widget.setCurrentText(parameters_value)
                    temp_widget.currentIndexChanged.connect( lambda *args, keys=key_stack : self._on_change_callback(keys) )
                    if self.isVisible(widget_value):
                        self.layouts[key_stack[0]].addRow(widget_value['label'] + ': ', temp_widget )
                        self.hasWidgets = True
                        
                elif widget_value['type'] == 'checkbox':
                    temp_widget = QCheckBox()
                    if parameters_value != temp_widget.isChecked(): temp_widget.toggle()
                    temp_widget.stateChanged.connect( lambda *args, keys=key_stack : self._on_change_callback(keys) )
                    if self.isVisible(widget_value):
                        self.layouts[key_stack[0]].addRow(widget_value['label'], temp_widget )
                        self.hasWidgets = True

                elif widget_value['type'] == 'floatslider':
                    temp_widget = DoubleSlider()
                    temp_widget.setMinimum(widget_value['min'])
                    temp_widget.setMaximum(widget_value['max'])
                    temp_widget.setSingleStep(widget_value['step'])
                    temp_widget.setValue(parameters_value)
                    temp_widget.valueChanged.connect( lambda *args, keys=key_stack : self._on_change_callback(keys) )
                    temp_labels =  QLabel( widget_value['label'] + '  ' + str(parameters_value) +': ' )
                    if self.isVisible(widget_value):
                        self.layouts[key_stack[0]].addRow(temp_labels, temp_widget )
                        self.hasWidgets = True

                elif widget_value['type'] == 'integerslider':
                    temp_widget = QSlider(Qt.Horizontal)
                    temp_widget.setMinimum(widget_value['min'])
                    temp_widget.setMaximum(widget_value['max'])
                    temp_widget.setSingleStep(widget_value['step'])
                    temp_widget.setTickPosition(QSlider.TicksBothSides)
                    temp_widget.setValue(parameters_value)
                    temp_widget.valueChanged.connect( lambda *args, keys=key_stack : self._on_change_callback(keys) )
                    temp_labels =  QLabel( widget_value['label'] + '  ' + str(parameters_value) +': ' )
                    if self.isVisible(widget_value):
                        self.layouts[key_stack[0]].addRow(temp_labels, temp_widget )
                        self.hasWidgets = True


                elif widget_value['type'] == 'button':
                    temp_widget = QPushButton( widget_value['label'] )
                    if key == 'LOAD':
                        temp_widget.clicked.connect( self._on_load_button )
                    elif key == 'DELETE':
                        temp_widget.clicked.connect( self._on_delete_button )
                    elif key == 'Calibrate':
                        temp_widget.clicked.connect( self.calibrate )
                    elif key == 'FittsLaw':
                        temp_widget.clicked.connect( self.fittslawtask )
                    elif key == 'SignalLevel':
                        temp_widget.clicked.connect( self.signalleveltask )
                    elif key == 'CalibrateSL':
                        temp_widget.clicked.connect( self.calibrateSL )
                    elif key == 'CalibrateFL':
                        temp_widget.clicked.connect( self._check_errors )
                    elif key == 'Targets':
                        temp_widget.clicked.connect( self.choosetarget )
                    elif key == 'TargetsFL':
                        temp_widget.clicked.connect( self.choosetargetFL )
                    elif key == 'LAUNCH_RESCU':
                        temp_widget.clicked.connect( self.launch_rescu )
                    elif key == 'LAUNCH_SEPREP':
                        temp_widget.clicked.connect( self.launch_seprep )

                    if self.isVisible(widget_value):
                        
                        self.layouts[key_stack[0]].addRow( temp_widget )
                        self.hasWidgets = True

                elif widget_value['type'] == 'splitter' and False:
                    separator = QFrame()
                    separator.setFrameShape(QFrame.HLine)
                    separator.setFrameShadow(QFrame.Sunken)
                    #separator.setSizePolicy(QSizePolicy.Minimum,QSizePolicy.MinimumExpanding)
                    separator.setLineWidth(4)
                    separator.setMidLineWidth(2)
                    separator.setStyle
                    self.layouts[key_stack[0]].addRow(separator)
                                
                try:
                    temp_dict = self._widget_properties[widget_value['dependency'][0]][widget_value['dependency'][1]]
                    temp_param = self._parameters[widget_value['dependency'][0]][widget_value['dependency'][1]]
                    for idx, i in enumerate(widget_value['dependency']):
                        if idx > 1 and idx < len(widget_value['dependency'])-1:
                            temp_dict = temp_dict[i]
                            temp_param = temp_param[i]
                    parameters_value = widget_value['values'][temp_dict['values'].index( temp_param ) ]
                except:
                    pass

                try:
                    return [parameters_value, temp_widget, temp_labels]
                except:
                    try:
                        return [parameters_value, temp_widget]
                    except:
                        return [parameters_value]
            except KeyError:
                
                for key, value in widget_value.items():
                    temp_key_stack = copy.deepcopy(key_stack)
                    temp_key_stack.append(key)
                    labels_value[key] = {}
                    widget_data_value[key] = {}
                    data = self.draw_widgets( key, value, parameters_value[key], labels_value[key], widget_data_value[key], temp_key_stack )

                    parameters_value[key] = data[0]
                    try:
                        widget_data_value[key] = data[1]
                    except:
                        pass
                    try:
                        labels_value[key] = data[2]
                    except:
                        pass

                return [parameters_value, widget_data_value, labels_value]

        def TabUI(self, tabkey):
            """Create the General page UI."""
            Tab = QWidget()
            self.layouts[tabkey] = QFormLayout()
            self.currentTab = Tab
            self.hasWidgets = False
            
            for key, value in self._widget_properties[tabkey].items():

                key_stack = [tabkey, key]
                self.labels[tabkey][key] = {}
                self.widgets[tabkey][key] = {}
                
                try:
                    data = self.draw_widgets( key, value, self._parameters[tabkey][key], self.labels[tabkey][key], self.widgets[tabkey][key], key_stack)
                except KeyError:
                    data = self.draw_widgets( key, value, [], self.labels[tabkey], self.widgets[tabkey], key_stack)

                self._parameters[tabkey][key] = data[0]
                try:
                    self.widgets[tabkey][key] = data[1]
                except:
                    pass
                try:
                    self.labels[tabkey][key] = data[2]
                except:
                    pass


            self.widgets[tabkey]['SAVE'] =QPushButton( 'SAVE PARAMETERS' )
            self.widgets[tabkey]['SAVE'].clicked.connect( self._on_save_button )
            self.layouts[tabkey].addRow( self.widgets[tabkey]['SAVE'] )

            self.widgets[tabkey]['EXIT_BUTTON'] =QPushButton( 'CLOSE' )
            self.widgets[tabkey]['EXIT_BUTTON'].clicked.connect( self._on_exit_button )
            self.layouts[tabkey].addRow( self.widgets[tabkey]['EXIT_BUTTON'] )

            Tab.setLayout(self.layouts[tabkey])
            return self.hasWidgets

        def redraw_widgets(self, key, widget_value, parameters_value, labels_value, widget_data_value, key_stack):
            # print( widget_value )
            try:

                if widget_value['type'] == 'textfield':
                    pass
                elif widget_value['type'] == 'integer':
                    widget_data_value.setValue(int(parameters_value))
                    if self.isVisible(widget_value):
                        self.layouts[key_stack[0]].addRow(widget_value['label'] + ': ', widget_data_value )
                        
                elif widget_value['type'] == 'dropdown':
                    if key_stack[0] == 'Fitts Law' and type(key_stack[-2]) == int:   
                        
                        widget_data_value.clear()
                        list_to_use = copy.deepcopy(widget_value['values'])
                        if key_stack[0] == 'Fitts Law' and type(key_stack[-2]) == int:
                            hand_act = 0
                            elbow_act = 0 
                            wrist_act = 0
                            wrist_dof = []                        
                            for i in range(4):
                                if self._parameters[key_stack[0]][key_stack[1]][i]['MOVEMENT_TYPE'] != 'Off':
                                    if self._parameters[key_stack[0]][key_stack[1]][i]['MOVEMENT_TYPE'] == 'Wrist':
                                        '''if wrist_act > 0:
                                            wrist_dof.append(self._parameters[key_stack[0]][key_stack[1]][i]['WRIST_DOF'])'''
                                        wrist_act += 1
                                        
                                    elif self._parameters[key_stack[0]][key_stack[1]][i]['MOVEMENT_TYPE'] == 'Hand':
                                        hand_act += 1
                                    elif self._parameters[key_stack[0]][key_stack[1]][i]['MOVEMENT_TYPE'] == 'Elbow':
                                        elbow_act += 1
                                        
                                if hand_act == 1 and parameters_value != 'Hand': 
                                    try: list_to_use.remove('Hand') 
                                    except: pass
                                if wrist_act == 2 and parameters_value != 'Wrist': 
                                    try: list_to_use.remove('Wrist') 
                                    except: pass
                                if elbow_act == 1 and parameters_value != 'Elbow': 
                                    try: list_to_use.remove('Elbow') 
                                    except: pass
                            '''if key_stack[-1] == 'WRIST_DOF':
                                for i in wrist_dof:
                                    try: list_to_use.remove(i) 
                                    except: pass'''

                            widget_data_value.addItems(list_to_use)
                    widget_data_value.setCurrentText(parameters_value)

                    if self.isVisible(widget_value):
                        self.layouts[key_stack[0]].addRow(widget_value['label'] + ': ', widget_data_value )
                        
                elif widget_value['type'] == 'checkbox':
                    if parameters_value != widget_data_value.isChecked(): widget_data_value.toggle()
                    if self.isVisible(widget_value):
                        self.layouts[key_stack[0]].addRow(widget_value['label'], widget_data_value )

                elif widget_value['type'] == 'floatslider':
                    widget_data_value.setValue(parameters_value)
                    labels_value =  QLabel( widget_value['label'] + '  ' + str(parameters_value) +': ' )
                    if self.isVisible(widget_value):
                        self.layouts[key_stack[0]].addRow(labels_value, widget_data_value )

                elif widget_value['type'] == 'integerslider':
                    widget_data_value.setValue(parameters_value)
                    labels_value =  QLabel( widget_value['label'] + '  ' + str(parameters_value) +': ' )
                    if self.isVisible(widget_value):
                        self.layouts[key_stack[0]].addRow(labels_value, widget_data_value )

                elif widget_value['type'] == 'button':
                    if self.isVisible(widget_value):
                        self.layouts[key_stack[0]].addRow( widget_data_value )

                elif widget_value['type'] == 'splitter' and False:                    
                    separator = QFrame()
                    separator.setFrameShape(QFrame.HLine)
                    separator.setFrameShadow(QFrame.Sunken)
                    #separator.setSizePolicy(QSizePolicy.Minimum,QSizePolicy.MinimumExpanding)
                    separator.setLineWidth(4)
                    separator.setMidLineWidth(2)
                    separator.setStyle
                    self.layouts[tabkey].addRow(separator)

                try:
                    temp_dict = self._widget_properties[widget_value['dependency'][0]][widget_value['dependency'][1]]
                    temp_param = self._parameters[widget_value['dependency'][0]][widget_value['dependency'][1]]
                    for idx, i in enumerate(widget_value['dependency']):
                        if idx > 1 and idx < len(widget_value['dependency'])-1:
                            temp_dict = temp_dict[i]
                            temp_param = temp_param[i]
                    parameters_value = widget_value['values'][temp_dict['values'].index( temp_param ) ]
                except:
                    pass

                try:
                    return [parameters_value, widget_data_value, labels_value]
                except:
                    try:
                        return [parameters_value, widget_data_value]
                    except:
                        return [parameters_value]

            except KeyError:

                for key, value in widget_value.items():
                    #print( '\n', key, '-----', value )
                    #if key == 'POWER' or key == 'PINCH':
                       # print('')
                    temp_key_stack = copy.deepcopy(key_stack)
                    temp_key_stack.append(key)
                    data = self.redraw_widgets( key, value, parameters_value[key], labels_value[key], widget_data_value[key], temp_key_stack )

                    parameters_value[key] = data[0]
                    try:
                        widget_data_value[key] = data[1]
                    except:
                        pass
                    try:
                        labels_value[key] = data[2]
                    except:
                        pass

                return [parameters_value, widget_data_value, labels_value]
            
            except AttributeError:
                subject = self.widgets['General']['SUBJECT_NAME'].text()
                folder = '.\Data\Subject_%d' % int(subject)
                filename = 'RESCU_Protocol_Subject_%d.pkl' % int(subject)

                PREVIOUS_FILE = os.path.join( folder, filename )
                if os.path.isfile(PREVIOUS_FILE):
                    os.remove(PREVIOUS_FILE)

            except:
                pass

        def TabUIRefill(self, tabkey):
            """Create the General page UI."""
            for key, value in self._widget_properties[tabkey].items():
                
                key_stack = [tabkey, key]
                
                try:
                    data = self.redraw_widgets( key, value, self._parameters[tabkey][key], self.labels[tabkey][key], self.widgets[tabkey][key], key_stack)
                except KeyError as k:
                    data = self.redraw_widgets( key, value, [], self.labels[tabkey], self.widgets[tabkey], key_stack)

                try:
                    self._parameters[tabkey][key] = data[0]
                except:
                    pass
                try:
                    self.widgets[tabkey][key] = data[1]
                except:
                    pass
                try:
                    self.labels[tabkey][key] = data[2]
                except:
                    pass

            self.layouts[tabkey].addRow( self.widgets[tabkey]['SAVE'] )

            self.layouts[tabkey].addRow( self.widgets[tabkey]['EXIT_BUTTON'] )

        def isVisible(self, value):
            
            if len(value['visible']) > 0:
                temp_dict = self._parameters[value['visible'][0]][value['visible'][1]]
                for idx, i in enumerate(value['visible']):
                    if idx > 1 and idx < len(value['visible'])-1:
                        temp_dict = temp_dict[i]
                try:
                    return ( temp_dict in value['visible'][-1]) and ((self.online and value['online']) or not self.online)
                except:
                    return ( temp_dict == value['visible'][-1]) and ((self.online and value['online']) or not self.online)
            else:
                return ((self.online and value['online']) or not self.online)


        def TabClear(self, tabkey):
            layout = self.layouts[tabkey]
            self.clearLayout(layout)

        def TabUpdate(self, tabkey):
            self.redrawingInProgress = True
            if tabkey in self.tabs_available:
                self.TabClear(tabkey)
                self.TabUIRefill(tabkey)
            self.redrawingInProgress = False

        def UpdateAll(self):
            for key in self._widget_properties:
                self.TabUpdate(key)

        def dict_recursion(self, dictionary, val, keys):

            if len(keys) == 1:
                dictionary[keys[0]] = val
                return dictionary
            else:
                key = keys[0]
                keys = keys[1:]
                out_dict = self.dict_recursion( dictionary[key], val, keys)
                dictionary[key] = out_dict
                return dictionary

        @pyqtSlot()
        def _on_change_callback(self, keys):
            if not self.redrawingInProgress:
                temp_dict = self._widget_properties[keys[0]][keys[1]]
                temp_widget = self.widgets[keys[0]][keys[1]]
                temp_label = self.labels[keys[0]][keys[1]]
                for idx, i in enumerate(keys):
                    if idx > 1:
                        temp_dict = temp_dict[i]
                        temp_widget = temp_widget[i]
                        temp_label = temp_label[i]
                try:
                    tabkey = keys[0]
                    key = keys[1]
                    type = temp_dict['type']
                    if type == 'integer' or type == 'textfield':
                        value = int(temp_widget.text())
                    elif type == 'dropdown':
                        value = temp_widget.currentText()
                    elif type == 'checkbox':
                        value = temp_widget.isChecked()
                    elif type == 'floatslider' or type == 'integerslider':
                        value = temp_widget.value()
                        i, j = self.layouts[tabkey].getWidgetPosition(temp_label)
                        label_item = self.layouts[tabkey].itemAt(i, j)
                        label_widget = label_item.widget()
                        label_widget.setText( temp_dict['label'] + '  ' + str(value) +': ' )

                    try:
                        self._parameters[keys[0]][keys[1]] = self.dict_recursion( self._parameters[keys[0]][keys[1]], value, keys[2:])
                    except:
                        self._parameters[keys[0]][keys[1]] = value

                    if type != 'floatslider' and type != 'integerslider':
                        self.UpdateAll()
                    
                    self._changed = True
                    self._last_changed_param = keys

                except Exception as e:
                    print(e)

        @pyqtSlot()
        def launch_rescu(self):
            self._on_save_button()
            self.close()
            from Rescu import RESCU
            RESCU(subject = self._parameters['General']['SUBJECT_NAME'], run_mode = 0, source = False)
            #subprocess.call([sys.executable, '.\code\Rescu.py', str(self._parameters['General']['SUBJECT_NAME'])])

        @pyqtSlot()
        def launch_seprep(self):
            self._on_save_button()
            self.close()
            from Rescu import RESCU
            RESCU(subject = self._parameters['General']['SUBJECT_NAME'], run_mode = 2, source = False)
            #subprocess.call([sys.executable, '.\code\Rescu.py', str(self._parameters['General']['SUBJECT_NAME']), '2'])

        @pyqtSlot()
        def calibrate(self):
            self._on_save_button()
            self.close()
            subprocess.call([sys.executable, '.\code\Calibration.py', str(self._parameters['General']['SUBJECT_NAME'])])

        @pyqtSlot()
        def fittslawtask(self):
            
            self._parameters['General']['FL'] = True
            self._on_save_button()
            if not self.online:
                self.close()
                from Rescu import RESCU
                RESCU(subject = self._parameters['General']['SUBJECT_NAME'], run_mode = 1, source = False)
                #subprocess.call([sys.executable, '.\code\Rescu.py', str(self._parameters['General']['SUBJECT_NAME']), '1'])
            else:
                self._changed = True
                self._last_changed_param = ['General', 'FL']

        @pyqtSlot()
        def signalleveltask(self):
            self._on_save_button()
            subprocess.call([sys.executable, '.\code\SignalLevel.py', str(self._parameters['General']['SUBJECT_NAME'])])

        @pyqtSlot()
        def calibrateSL(self):
            self._on_save_button()
            subprocess.call([sys.executable, '.\code\CalibrateSignalLevel.py', str(self._parameters['General']['SUBJECT_NAME'])])

        @pyqtSlot()
        def calibrateFL(self):
            self._parameters['General']['FL'] = True
            self._on_save_button()
            subprocess.call([sys.executable, '.\code\Calibration.py', str(self._parameters['General']['SUBJECT_NAME'])])

        @pyqtSlot()
        def choosetarget(self):
            self._on_save_button()
            subprocess.call([sys.executable, '.\code\ChooseTargets.py', str(self._parameters['General']['SUBJECT_NAME'])])

        @pyqtSlot()
        def choosetargetFL(self):
            self._parameters['General']['FL'] = True
            self._on_save_button()
            subprocess.call([sys.executable, '.\code\ChooseTargets.py', str(self._parameters['General']['SUBJECT_NAME'])])

        @pyqtSlot()
        def errormessage(self):
            subprocess.call([sys.executable, '.\code\ErrorMessage.py', str(self._parameters['General']['SUBJECT_NAME'])])
            
        @pyqtSlot()
        def _on_save_button(self):
            # write parameters to text file
            if self.online == False:
                subject = self.widgets['General']['SUBJECT_NAME'].text()
                if subject != '':

                    classes = []
                    gripclasses = ['OPEN']
                    for i in range(4):
                        classes, gripclasses = self._gather_FL_classes(i, classes, gripclasses)                    
                    
                    sorted_classes = ['REST']
                    if self._parameters['Fitts Law']['DoF'][0]['MOVEMENT_TYPE'] == 'Hand' or self._parameters['Fitts Law']['DoF'][1]['MOVEMENT_TYPE'] == 'Hand' or self._parameters['Fitts Law']['DoF'][2]['MOVEMENT_TYPE'] == 'Hand' or self._parameters['Fitts Law']['DoF'][3]['MOVEMENT_TYPE'] == 'Hand':
                        for i in gripclasses:
                            sorted_classes.append(i)
                        for i in classes:
                            sorted_classes.append(i)
                    classes = sorted_classes
                    self._parameters['Fitts Law']['CUE_LIST'] = classes
                    self._parameters['Fitts Law']['GRIP_CUE_LIST'] = gripclasses

                    folder = '.\Data\Subject_%d' % int(subject)
                    filename = 'RESCU_Protocol_Subject_%d.pkl' % int(subject)
                    if not path.exists(folder):
                        os.mkdir(folder)
                    filepath = folder + '/' + filename
                    if not path.exists(filepath):
                        f = open(filepath, "w")
                        f.write("")
                        f.close()

                    classes = ['REST']
                    all_classes = ['REST']

                    for key, i in self._widget_properties['Calibration'].items():
                        try:
                            if i['movement'] and self._parameters['Calibration'][key]:
                                classes.append(key)
                            if i['movement']:
                                all_classes.append(key)
                        except:
                            pass

                    self._parameters['Calibration']['CUE_LIST'] = classes
                    self._parameters['Calibration']['ALL_CLASSES'] = all_classes
            else:
                subject = self._parameters['General']['SUBJECT_NAME']

            folder = '.\Data\Subject_%d' % int(subject)
            filename = 'RESCU_Protocol_Subject_%d.pkl' % int(subject)
            if not path.exists(folder):
                os.mkdir(folder)
            filepath = folder + '/' + filename
            if not path.exists(filepath):
                f = open(filepath, "w")
                f.write("")
                f.close()

            # TODO call this with the local public function
            with open( f"Data\Subject_{subject}\RESCU_Protocol_Subject_{subject}.pkl", 'wb' ) as f:
                pickle.dump(self._parameters, f)

        def _gather_FL_classes(self, DoF, classes, gripclasses):
                                        
            if self._parameters['Fitts Law']['DoF'][DoF]['MOVEMENT_TYPE'] != 'Off':
                if self._parameters['Fitts Law']['DoF'][DoF]['MOVEMENT_TYPE'] == 'Hand':
                    for key, i in self._widget_properties['Fitts Law']['DoF'][DoF].items():
                        try:
                            if i['movement'] and self._parameters['Fitts Law']['DoF'][DoF][key]:
                                gripclasses.append(key)
                        except:
                            pass
                # TODO restructure these below
                if self._parameters['Fitts Law']['DoF'][DoF]['MOVEMENT_TYPE'] == 'Wrist':
                    if self._parameters['Fitts Law']['DoF'][DoF]['WRIST_DOF'] == 'ROTATION':
                        classes.append('PALM DOWN')
                        classes.append('PALM UP')
                    elif self._parameters['Fitts Law']['DoF'][DoF]['WRIST_DOF'] == 'FLEXION':
                        classes.append('FLEXION')
                        classes.append('EXTENSION')
                elif self._parameters['Fitts Law']['DoF'][DoF]['MOVEMENT_TYPE'] == 'Elbow':
                    classes.append('ELBOW BEND')
                    classes.append('ELBOW EXTEND')

            return classes, gripclasses

        @pyqtSlot()
        def _check_errors(self):
            self._on_save_button()
            if len(self._parameters['Fitts Law']['CUE_LIST']) != len(set(self._parameters['Fitts Law']['CUE_LIST'])):
                self.errormessage()
            else:
                self.calibrateFL()

        @pyqtSlot()
        def _on_exit_button(self):
            self._parameters['General']['EXIT_BUTTON'] = True
            self._changed = True
            self._last_changed_param = ['General','EXIT_BUTTON']

        @pyqtSlot()
        def _on_load_button(self): 
            subject = self.widgets['General']['SUBJECT_NAME'].text()
            
            try:
                # TODO call this with the local public function
                with open( f"Data\Subject_{subject}\RESCU_Protocol_Subject_{subject}.pkl", 'rb' ) as f:
                    Protocol_Data = pickle.load(f)

                self._parameters = Protocol_Data
                self._parameters['General']['FL'] = False
            except:
                print( 'No data for selected subject.' )
            self.UpdateAll()

        @pyqtSlot()
        def _on_delete_button(self): 
            practice_session = 1
            subject = self.widgets['General']['SUBJECT_NAME'].text()
            FILE_DIRECTORY = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), 'data', 'Subject_%s' % subject )
            print( 'Deleting previous session data...', end = '', flush = True)
            while True:
                PREVIOUS_FILE = os.path.join( FILE_DIRECTORY, 'Subject_%s_Practice_%s.pkl' % ( subject, practice_session ) )
                if os.path.isfile(PREVIOUS_FILE):
                    
                    practice_session += 1
                    os.remove(PREVIOUS_FILE)
                else:
                    print( 'Done')
                    break

        @property
        def parameters(self):
            return self._parameters

        @property
        def last_changed_param(self):
            return self._last_changed_param

        @property
        def changed(self):
            return self._changed

        def set_changed(self):
            self._changed = False

        def set_parameters(self, parameters):
            self._parameters = parameters
            

    def __init__(self, left = 1370, top = 630, width = 1280, height = 720, online = False, preload = None, visualize = True):
        self._online = online
        self._preload = preload
        self.visualize = visualize
        
        self._param_buffer = mp.Queue( maxsize = 1 )
        self._change_buffer = mp.Queue( maxsize = 1 )
        self._external_param_buffer = mp.Queue( maxsize = 1 )
        self._exit_event = mp.Event()

        self._process = mp.Process( target = self._eventloop )

        if self.visualize:
            self._process.start()

    def _eventloop( self ):
        """
        Main GUI eventloop to be run in a separate process
        """
        app = QApplication( sys.argv )
        window = InitGUI.MainWindow( online = self._online, preload = self._preload )

        # launch the polling thread
        if self.visualize:
            thread = th.Thread( target = self._poll, args = ( window, ) )
            thread.start()

            window.show()

        try:
            app.exec_()
        finally:
            # print( 'exec done!' )
            self._cleanup()
            # print( 'process done!' )

    def _poll(self, window):
        """
        Polls the passed MainWindow object for its current parameters and stores them in the buffer
        """

        while not self._exit_event.is_set():
            
            change = window._last_changed_param
            if window.changed is True :
                self.change
                self.parameters
                self._change_buffer.put( change, timeout = 1e-3 )
                self._param_buffer.put( window.parameters, timeout = 1e-3 )
                
                window.set_changed()

            try:
                external_param = self._external_param_buffer.get( timeout = 1e-3 )
                if external_param is not None:
                    window.set_parameters( external_param )
                    #window.UpdateAll()
            except:
                pass
        window.close()
        # print( 'thread done!' )

    def _cleanup(self):
        self._exit_event.set()

        # empty buffer so polling thread can join
        empty = False
        while not empty:
            try:
                self._change_buffer.get( timeout = 1e-3 )
            except queue.Empty:
                empty = True

    def close( self ):
        self._exit_event.set()

        # empty buffer so polling thread can join
        empty = False
        while not empty:
            try:
                self._change_buffer.get( timeout = 1e-3 )
            except queue.Empty:
                empty = True
        self._process.join()

    def load_subject_parameters( self, subject ):
        FILE_DIRECTORY = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), 'data', 'Subject_%s' % subject )
        PARAM_FILE = os.path.join( FILE_DIRECTORY, 'RESCU_Protocol_Subject_%s.pkl' % subject )
        Parameters = self.load_data_from_file( PARAM_FILE )
        return Parameters
    
    def load_data_from_file( self, path ):
        try:
            with open( path, 'rb' ) as file:
                data = pickle.load( file )
                return data
        except:
            print( 'error reading data from ', path )
            return None
    
    def save_subject_parameters( self, subject, parameters ):
        folder = '.\Data\Subject_%d' % int(subject)
        filename = 'RESCU_Protocol_Subject_%d.pkl' % int(subject)
        if not path.exists(folder):
            os.mkdir(folder)
        filepath = folder + '/' + filename
        if not path.exists(filepath):
            f = open(filepath, "w")
            f.write("")
            f.close()
        self.save_data_to_file( f"Data\Subject_{subject}\RESCU_Protocol_Subject_{subject}.pkl", parameters )
    
    def save_data_to_file( self, path, data ):
        with open( path, 'wb' ) as f:
            pickle.dump( data, f )

    def set_parameters( self, parameters ):
        self._external_param_buffer.put( parameters, timeout = 1e-3 )

    @property
    def parameters(self):
        """
        Return the newest GUI parameters (if they've been updated since the last time this function was called)
        """
        if self._param_buffer.qsize():
            try:
                self.current_parameters = self._param_buffer.get( timeout = 1e-3 )
                return self.current_parameters
            except queue.Empty:
                pass
        try:
            return self.current_parameters
        except:
            return None

    @property
    def change(self):
        """
        Return the newest GUI parameters (if they've been updated since the last time this function was called)
        """
        if self._change_buffer.qsize():
            try:
                params = self._change_buffer.get( timeout = 1e-3 )
                return params
            except queue.Empty:
                pass
        return None

    @property
    def running(self):
        return not self._exit_event.is_set()

if __name__ == "__main__":
    
    init = InitGUI( online=False, preload = 1 )

    '''with open( f"Data\Subject_{1}\RESCU_Protocol_Subject_{1}.pkl", 'rb' ) as f:
        Protocol_Data = pickle.load(f)

    all_classes = ['OPEN','POWER','PALM_UP','PALM_DOWN','TRIPOD', 'KEY', 'INDEX']
    all_vel = ['OPEN_VEL','POWER_VEL','PALM_UP_VEL','PALM_DOWN_VEL','TRIPOD_VEL', 'KEY_VEL', 'INDEX_VEL']
    Protocol_Data['CUE_LIST'].remove('rest')'''

    while init.running:
        change = init.change
        if change is not None:
            param = init.parameters
            #print(param)
            if change == ['General', 'EXIT_BUTTON']:
                init.close()
            '''param['General']['DEVICE_ELECTRODE_COUNT'] = 69
            print(param)
            init.set_parameters(param)'''
        pass
        '''if p is not None:
            print(p)'''
