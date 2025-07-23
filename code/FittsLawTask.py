from genericpath import exists
import multiprocessing as mp
from multiprocessing import pool
import queue
from re import L
import time
import os
import fnmatch

import matplotlib
import matplotlib.image as mpimg
matplotlib.use("QT5Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle
import numpy as np
import scipy

from matplotlib.widgets import Button
import math
import csv
import sys
import pickle
import random
import copy

from OnsetThreshold import Onset

from TimeDomainFilter import TimeDomainFilter
from FourierTransformFilter import FourierTransformFilter
from SenseController import SenseController
from MyoArmband import MyoArmband

from TrainingDataGeneration import TrainingDataGenerator
from RunClassifier import RunClassifier
from FeatureSmoothing import Smoothing

from datetime import datetime

sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ))

from os import path

from local_addresses import address

from CueVisualizer import CueVisualizer

from effective_mov import effective

#set file path for Parameters and get Sense address
SUBJECT = 2 # this needs to be changed to do FittsLawTaskResults still
DAY = 1 # this need to be changed before every day of test

if len(sys.argv) > 1:
    #print( 'SUBJECT:', str(sys.argv[1]) )
    SUBJECT = sys.argv[1]

CUE_FLDR = os.path.join( 'cues' )
PARAM_FILE = os.path.join('data', 'Subject_%s' % SUBJECT, 'RESCU_Protocol_Subject_%s.pkl' % SUBJECT)



with open( PARAM_FILE, 'rb' ) as pkl:
    Parameters = pickle.load( pkl )


EMG_SCALING_FACTOR = 10000.0

cue_count = ( Parameters['General']['SAMPLING_RATE'] * Parameters['Calibration']['CUE_DURATION'] - Parameters['General']['WINDOW_SIZE'] ) / Parameters['General']['WINDOW_STEP']

local_surce_info = address()

class FittsLawTask:

    RESULTS = {}
    
    def __init__(self, subject = 1, classes = ('rest','open','power','pronate','supinate','tripod') ):

        self.subject = subject

        CUE_FLDR = os.path.join( 'cues' )

        PARAM_FILE = os.path.join('data', 'Subject_%s' % SUBJECT, 'RESCU_Protocol_Subject_%s.pkl' % SUBJECT )

        with open( PARAM_FILE, 'rb' ) as pkl:
            Parameters = pickle.load( pkl )

        self._classes = classes
        self.show_raw = True

        self._queue = mp.Queue()
        self._queue_out = mp.Queue()
        self._inner_queue = mp.Queue()
        self._exit_event = mp.Event()
        self._run_event = mp.Event()
        self._plotter = mp.Process(target = self._plot)

        self.current_omega = 0

        self.runcountrim = 0
        self.runcountcir = 0
        self.runcountx = 0
        self.runcounty = 0

        self.method = 0

        self.rimcomplete = False
        self.circlecomplete = False
        self.xcomplete = False
        self.ycomplete = False

        self.axis_side = [0,0,0,0]

        self.zero = False
        self.one = False
        self.two = False
        self.three = False

        self.next_start = True

        self.parentdict = {}

        self.targetgrip = 'none'

        self.start_time = 0
        self.end_time = 0

        self.circle_default_radius = (0.5 + 0.075)/2

        self.date_time = ''

        self.griplist = []
        non_grips = ['REST', 'PALM UP', 'PALM DOWN', 'FLEXION', 'EXTENSION', 'ELBOW BEND', 'ELBOW EXTEND']

        if 'OPEN' in self._classes:
            for i in self._classes:
                if i not in non_grips:
                    self.griplist.append(i)


        self.targetlist = []
        if Parameters['Fitts Law']['DoF'][0]['MOVEMENT_TYPE'] != 'Off':
            self.targetlist.append('circle')
        if Parameters['Fitts Law']['DoF'][1]['MOVEMENT_TYPE'] != 'Off':
            self.targetlist.append('rim')
        if Parameters['Fitts Law']['DoF'][2]['MOVEMENT_TYPE'] != 'Off':
            self.targetlist.append('x')
        if Parameters['Fitts Law']['DoF'][3]['MOVEMENT_TYPE'] != 'Off':
            self.targetlist.append('y')
        
        inputloc = './Input/fittslawtargets.pkl'
        try:
            with open( inputloc, 'rb' ) as f:
                self.DoF_mov_list = pickle.load(f)

            with open( 'fittslawrandom.txt', 'r') as f:
                idx = []
                for line in f.readlines():
                    idx.append( int( line[:-1] ) )
            self.idx = idx


            with open( 'fittslawpositions.txt', 'r') as f:
                pos = []
                for line in f.readlines():
                    pos.append( int( line[:-1] ) )
            self.position_order = pos

            with open( 'fittslawmethods.txt', 'r') as f:
                method = []
                for line in f.readlines():
                    method.append( int( line[:-1] ) )
            self.method_order = method

        except:
            print("File containing targets does not exist")
            exit()

        for i in range( 4 ):
            if len( self.DoF_mov_list[i]['Movements'] ):        # only shuffle selected DoFs
                print(self.DoF_mov_list[i]['Movements'])
                self.DoF_mov_list[i]['Movements']       = [ self.DoF_mov_list[i]['Movements'][j] for j in self.idx ]
                print(self.DoF_mov_list[i]['Movements'], '\n')

                PATH1 = os.path.join('data', 'Subject_%s' % SUBJECT, 'movement_hand')
                isexist = os.path.exists(PATH1)
                if not isexist:
                    os.makedirs(PATH1)
                with open(os.path.join(PATH1, 'file_%s.txt' % DAY), "w") as f:
                    for s in self.DoF_mov_list[0]['Movements']:
                        f.write(str(s) + "\n")

                PATH2 = os.path.join('data', 'Subject_%s' % SUBJECT, 'movement_wrist')
                isexist = os.path.exists(PATH2)
                if not isexist:
                    os.makedirs(PATH2)
                with open(os.path.join(PATH2, 'file_%s.txt' % DAY), "w") as f:
                    for s in self.DoF_mov_list[1]['Movements']:
                        f.write(str(s) + "\n")

                self.DoF_mov_list[i]['Target Position'] = [ self.DoF_mov_list[i]['Target Position'][j] for j in self.idx ]
                self.DoF_mov_list[i]['Target Range']    = [ self.DoF_mov_list[i]['Target Range'][j] for j in self.idx ]
                self.DoF_mov_list[i]['Target ID']       = [ self.DoF_mov_list[i]['Target ID'][j] for j in self.idx ]

                print( '\n', self.idx, len( self.DoF_mov_list[i]['Target ID'] ) )


        for i in range(4):
            if len(self.DoF_mov_list[i]['Movements']) != 0:
                self.mov_per_trial = len(self.DoF_mov_list[i]['Movements'])/Parameters['Fitts Law']['Trials']
        self.cue = Parameters['Fitts Law']['Start Cue']

        self.position_cue = CueVisualizer( cue_path = os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ), "position" ) )


        self._plotter.start()

    def add_inner_queue(self, x, y, rad):
        try:
            self._inner_queue.put([x,y,rad],timeout=1e-3)
        except queue.Full:
            pass

    def _check_target(self):
        for i in self.targetlist:
            if i == 'rim':
                if self.rimcomplete == True:
                    self.zero = True
            elif 'rim' not in self.targetlist:
                self.zero = True
            elif 'rim' in self.targetlist and self.rimcomplete == False:
                self.zero = False
            if i == 'circle':
                if self.circlecomplete == True:
                    self.one = True
            elif 'circle' not in self.targetlist:
                self.one = True
            elif 'circle' in self.targetlist and self.circlecomplete == False:
                self.one = False
            if i == 'x':
                if self.xcomplete == True:
                    self.two = True
            elif 'x' not in self.targetlist:
                self.two = True
            elif 'x' in self.targetlist and self.xcomplete == False:
                self.two = False
            if i == 'y':
                if self.ycomplete == True:
                    self.three = True
            elif 'y' not in self.targetlist:
                self.three = True
            elif 'y' in self.targetlist and self.ycomplete == False:
                self.three = False
        if self.zero == True and self.one == True and self.two == True and self.three == True:
            self._next_target()

    def _next_target(self):
        self.RESULTS[self.cue] =     {
            'METHOD'                :   None,
            'CUE_COMPLETION_TIMES'  :   -1,
            'COMPLETION_RESULTS'    :   False,
            'OVERSHOOT_RESULTS'     :   {
                '0_DOF'  :   {},
                '1_DOF'  :   {},
                '2_DOF'  :   {},
                '3_DOF'  :   {}
            },
            'PROPORTIONAL_VALUES'   :   [],
            'CLASSIFICATION_VALUES' :   [],
            'FEATURE_VALUES'        :   [],
            'PATH'                  :   {
                'OPTIMAL_PATH'  :   {
                    '0_DOF'  :   0,
                    '1_DOF'  :   0,
                    '2_DOF'  :   0,
                    '3_DOF'  :   0
                },
                'TRUE_PATH'  :   {
                    '0_DOF'  :   0,
                    '1_DOF'  :   0,
                    '2_DOF'  :   0,
                    '3_DOF'  :   0
                }
            }
        }
        '''self.method += 1

        self._queue_out.put(str(self.method%2+1), timeout=1e-3)'''

        # print( "Method_%d" % ( self.methods[self.cue] + 1 )
        # print(self.cue, len(self.method_order))

        if self.cue > 0:
            self.RESULTS[self.cue-1]['METHOD'] = self.method_order[self.cue-1]
            self._queue_out.put( "Method_%d" % ( self.method_order[self.cue-1]), timeout = 1e-3 )
            
        if self.cue > 0 and self.cue < Parameters['Fitts Law']['Trials']*self.mov_per_trial:
            print( "Target Position: %d" % ( self.position_order[self.cue] ) )

            print( "%02d" % ( self.position_order[self.cue] ) )
            self.position_cue.publish( "%02d" % ( self.position_order[self.cue] ) )


        # Parameters['Fitts Law']['Method'] = str( self.method_order[self.cue] )
        # print(Parameters['Fitts Law']['Method']  )
        # self.RESULTS[self.cue]['METHOD'] = self.methods[self.cue]

        try:
            self.target_radius = self.DoF_mov_list[0]['Target Position'][self.cue]
            self.target_radius_offset = self.DoF_mov_list[0]['Target Range'][self.cue]

            if self.target_radius + self.target_radius_offset < self.circle_default_radius:
                self.RESULTS[self.cue]['PATH']['OPTIMAL_PATH']['0_DOF'] = self.circle_default_radius - (self.target_radius + self.target_radius_offset)
            else:
                self.RESULTS[self.cue]['PATH']['OPTIMAL_PATH']['0_DOF'] = (self.target_radius - self.target_radius_offset) - self.circle_default_radius

            self.rim_target_starting_angle = self.DoF_mov_list[1]['Target Position'][self.cue]
            self.rim_target_tolerance = self.DoF_mov_list[1]['Target Range'][self.cue]

            self.RESULTS[self.cue]['PATH']['OPTIMAL_PATH']['1_DOF'] = copy.deepcopy(abs(self.rim_target_starting_angle)-self.rim_target_tolerance)

            self.target_centroid_x = self.DoF_mov_list[2]['Target Position'][self.cue]
            self.xaxis_target_range = self.DoF_mov_list[2]['Target Range'][self.cue]

            if self.target_centroid_x + self.xaxis_target_range < self.x_axis:
                self.RESULTS[self.cue]['PATH']['OPTIMAL_PATH']['2_DOF'] = self.x_axis - (self.target_centroid_x + self.xaxis_target_range)
            else:
                self.RESULTS[self.cue]['PATH']['OPTIMAL_PATH']['2_DOF'] = (self.target_centroid_x - self.xaxis_target_range) - self.x_axis

            self.target_centroid_y = self.DoF_mov_list[3]['Target Position'][self.cue]
            self.yaxis_target_range = self.DoF_mov_list[3]['Target Range'][self.cue]

            if self.target_centroid_y + self.yaxis_target_range < self.y_axis:
                self.RESULTS[self.cue]['PATH']['OPTIMAL_PATH']['3_DOF'] = self.y_axis - (self.target_centroid_y + self.yaxis_target_range)
            else:
                self.RESULTS[self.cue]['PATH']['OPTIMAL_PATH']['3_DOF'] = (self.target_centroid_y - self.yaxis_target_range) - self.y_axis
        except: pass

        # print("self.cue", self.cue, 'self.mov_per_trial', self.mov_per_trial)

        folder = 'Data\Subject_%d' % int(SUBJECT)
        filename = 'FLT_results_%s.pkl' % self.date_time
        if not path.exists(folder):
            os.mkdir(folder)
        filepath = folder + '/' + filename
        if not path.exists(filepath):
            f = open(filepath, "w")
            f.write("")
            f.close()
        
        if (time.perf_counter()-self.trialstart) >= Parameters['Fitts Law']['Trial Time']:
            self.RESULTS[self.cue-1]['CUE_COMPLETION_TIMES'] = Parameters['Fitts Law']['Trial Time']
            self.RESULTS[self.cue-1]['COMPLETION_RESULTS'] = False
            # print(self.RESULTS[self.cue-1]['OVERSHOOT_RESULTS'])

            with open( f"Data\Subject_{SUBJECT}\FLT_results_{self.date_time}.pkl", 'wb' ) as f:
                pickle.dump(self.RESULTS, f)

            try:
                self.rim_target_plot.remove()
                #self.rim_arc.remove()
            except: pass
            self.rimcomplete = False
            try:
                self.inner_target_circle_plot.remove()
                self.outer_target_circle_plot.remove()
            except: pass
            self.circlecomplete = False
            try:
                self.xaxis_target.remove()
            except: pass
            self.xcomplete = False
            try:
                self.yaxis_target.remove()
            except: pass
            self.ycomplete = False
            self.endtext = self.fig.text(0.5,0.7,'Out of Time! \nPress Enter to Restart',size=30,ha='center',va='center',bbox=dict(boxstyle = 'round',color='lightsteelblue',alpha=0.3))
            plt.pause(0.001)
            plt.waitforbuttonpress()
            self._restart()

        elif self.cue > 0 and self.cue%self.mov_per_trial == 0:
            self.end_time = time.time()
            self.RESULTS[self.cue-1]['CUE_COMPLETION_TIMES'] = self.end_time - self.start_time
            self.RESULTS[self.cue-1]['COMPLETION_RESULTS'] = True
            # print(self.RESULTS[self.cue-1]['OVERSHOOT_RESULTS'])

            with open( f"Data\Subject_{SUBJECT}\FLT_results_{self.date_time}.pkl", 'wb' ) as f:
                pickle.dump(self.RESULTS, f)
            
            if self.cue == Parameters['Fitts Law']['Trials']*self.mov_per_trial:

                try:
                    self.im.remove()
                except: pass
                try:
                    self.circle_plot.remove()
                except: pass
                try:
                    self.rim_plot[0].remove()
                except: pass
                try:
                    self.xaxis_indicator_bottom_plot[0].remove()
                    self.xaxis_indicator_top_plot[0].remove()
                except: pass
                try:
                    self.yaxis_indicator_right_plot[0].remove()
                    self.yaxis_indicator_left_plot[0].remove()
                except: pass
                # print('trials complete')
                self.finishedtext = self.fig.text(0.5,0.5,'Trials Completed! \nPress ENTER to Close',size=30,ha='center',va='center',bbox=dict(boxstyle = 'round',color='lightsteelblue',alpha=0.3))
                plt.pause(0.001)
                plt.waitforbuttonpress()
                plt.close()
                
            else:

                # print('trial complete')
                self.finishedtext = self.fig.text(0.5,0.9,'Trial Completed! \nPress ENTER to Continue',size=20,ha='center',va='center',bbox=dict(boxstyle = 'round',color='lightsteelblue',alpha=0.3))
                plt.pause(0.001)
                plt.waitforbuttonpress()
                self.next_start = True
                self.finishedtext.remove()
                try:
                    self.rim_target_plot.remove()
                    #self.rim_arc.remove()
                except: pass
                self.rimcomplete = False
                try:
                    self.inner_target_circle_plot.remove()
                    self.outer_target_circle_plot.remove()
                except: pass
                self.circlecomplete = False
                try:
                    self.xaxis_target.remove()
                except: pass
                self.xcomplete = False
                try:
                    self.yaxis_target.remove()
                except: pass
                self._restart()
        else:
            
            self.xcomplete = False
            self.ycomplete = False
            self.rimcomplete = False
            self.circlecomplete = False
            self.activey = False
            self.activex = False
            self.activecir = False
            self.activerim = False
            
            try:
                for i in self.targetlist:
                    if i == 'rim':
                        self.rim_target_plot.remove()
                        #self.rim_arc.remove()
                    elif i == 'circle':
                        self.inner_target_circle_plot.remove()
                        self.outer_target_circle_plot.remove()
                    elif i == 'x':
                        self.xaxis_target.remove()
                    elif i == 'y':
                        self.yaxis_target.remove()
                
                if self.cue > 0:

                    self.end_time = time.time()
                    self.RESULTS[self.cue-1]['CUE_COMPLETION_TIMES'] = self.end_time - self.start_time
                    self.RESULTS[self.cue-1]['COMPLETION_RESULTS'] = True

                    with open( f"Data\Subject_{SUBJECT}\FLT_results_{self.date_time}.pkl", 'wb' ) as f:
                        pickle.dump(self.RESULTS, f)

                    self.finishedtext = self.fig.text(0.5,0.9,'Task Completed! \nPress ENTER to Continue',size=20,ha='center',va='center',bbox=dict(boxstyle = 'round',color='lightsteelblue',alpha=0.3))
                    plt.pause(0.001)
                    plt.waitforbuttonpress()
                    self.finishedtext.remove()
                    self.next_start = True

                    self._generate_targets()
            except:
                pass

            self.xaxis_indicator_top_plot[0].set_color('b')
            self.xaxis_indicator_bottom_plot[0].set_color('b')
            self.yaxis_indicator_left_plot[0].set_color('y')
            self.yaxis_indicator_right_plot[0].set_color('y')

            if self.next_start == True:
                self.next_start = False
                self.x_axis = 0.5
                self.y_axis = 0.5
                self.rim_ratio = 0
                self.circle_radius = copy.deepcopy(self.circle_default_radius)

                self.previous_x_axis = copy.deepcopy(self.x_axis)
                self.previous_y_axis = copy.deepcopy(self.y_axis)
                self.previous_omega = 0
                self.previous_circle_radius = copy.deepcopy(self.circle_radius)

            if Parameters['Fitts Law']['DoF'][0]['MOVEMENT_TYPE'] != 'Off':  
                self._draw_circle()
            if Parameters['Fitts Law']['DoF'][1]['MOVEMENT_TYPE'] != 'Off':
                self._draw_rim()
            if Parameters['Fitts Law']['DoF'][2]['MOVEMENT_TYPE'] != 'Off':
                self._draw_x()
            if Parameters['Fitts Law']['DoF'][3]['MOVEMENT_TYPE'] != 'Off':
                self._draw_y()
            self.trialstart = time.perf_counter()

        self.cue += 1
        # print('cue incremented')

    # TODO the dof-axis pairings don't seem to make sens regarding Hand gestures
    def _generate_targets(self):
        
        self.start_time = time.time()
        # print( "" )
        print( "Trial", self.cue, "\n")
        # print( "Target Position: %d" % ( self.position_order[self.cue] ) )

        self.target_radius = self.DoF_mov_list[0]['Target Position'][self.cue]
        self.target_radius_offset = self.DoF_mov_list[0]['Target Range'][self.cue]

        self.rim_target_starting_angle = self.DoF_mov_list[1]['Target Position'][self.cue]
        self.rim_target_tolerance = self.DoF_mov_list[1]['Target Range'][self.cue]

        self.target_centroid_x = self.DoF_mov_list[2]['Target Position'][self.cue]
        self.xaxis_target_range = self.DoF_mov_list[2]['Target Range'][self.cue]

        self.target_centroid_y = self.DoF_mov_list[3]['Target Position'][self.cue]
        self.yaxis_target_range = self.DoF_mov_list[3]['Target Range'][self.cue]

        for i in range(4):
            if Parameters['Fitts Law']['DoF'][i]['MOVEMENT_TYPE'] == 'Hand':  
                self.targetgrip = self.DoF_mov_list[i]['Movements'][self.cue]

    def _draw_circle(self):
        # draw target circle
        self.outer_target_circle_plot = self.cax.add_patch(Circle(  (self.x_axis, self.y_axis), 
                                                                    self.target_radius+self.target_radius_offset, 
                                                                    linewidth = 0,
                                                                    color = 'r',
                                                                    zorder = 1,
                                                                    alpha = 0.3))

        self.inner_target_circle_plot = self.cax.add_patch(Circle(  (self.x_axis, self.y_axis), 
                                                                    self.target_radius-self.target_radius_offset, 
                                                                    linewidth = 0,
                                                                    color = 'w',
                                                                    zorder = 2))

    def _draw_rim(self):
        # draw target rim on circle
        self.rim_target_tolerance_range = [(360-self.rim_target_starting_angle)%360, (360-self.rim_target_starting_angle+self.rim_target_tolerance)%360]
        
        self.rim_target_plot = self.cax.add_patch(Arc(   (self.x_axis, self.y_axis), 
                                                    self.circle_radius, self.circle_radius,
                                                    theta1 = 90-self.rim_target_tolerance_range[1],
                                                    theta2 = 90-self.rim_target_tolerance_range[0],
                                                    linewidth = 12, 
                                                    alpha = 0.3, color = 'r', zorder = 8 ))

        # moving rim arc
        '''self.rim_arc = self.cax.add_patch(Arc(   (self.x_axis, self.y_axis), 
                                                    self.circle_radius, self.circle_radius,0, 
                                                    theta1 = -self.rim_target_tolerance_range[1], linewidth = 4, 
                                                    alpha = 0.3, color = 'r', zorder = 8 ))'''

    def _draw_x(self):
        self.xaxis_target = self.cax.fill_betweenx(  np.linspace(self.cax.get_ylim()[0], self.cax.get_ylim()[1], 2),
                                                self.target_centroid_x-self.xaxis_target_range, 
                                                self.target_centroid_x+self.xaxis_target_range,
                                                linewidth = 0, 
                                                alpha = 0.3,
                                                zorder = 3,
                                                color = 'b' )

    def _draw_y(self):
        self.yaxis_target = self.cax.fill_between(   np.linspace(self.cax.get_xlim()[0], self.cax.get_xlim()[1], 2),
                                                    self.target_centroid_y-self.yaxis_target_range, 
                                                    self.target_centroid_y+self.yaxis_target_range,
                                                    linewidth = 0, 
                                                    alpha = 0.3,
                                                    zorder = 4,
                                                    color = 'y' )
            
    # DRAW AND CHECK TARGET RIM ON CIRCLE
    def _update_target_rim(self, movement):
        if self.runcountrim == 0:
            self.activerim = False

        self.rim_target_plot.center = [self.x_axis,self.y_axis]
        self.rim_target_plot.height = self.circle_radius*2 + 0.03
        self.rim_target_plot.width = self.circle_radius*2 + 0.03

        if (self.current_omega) > (self.rim_target_tolerance_range[0]) and (self.current_omega) < (self.rim_target_tolerance_range[1]): 
            if self.activerim == False:
                self.startrim = time.perf_counter()
            self.activerim = True 
            self.rim_target_plot.set_color('g')
            #self.rim_arc.set_color('g')
        else:
            if self.activerim:
                try:
                    self.RESULTS[self.cue-1]['OVERSHOOT_RESULTS']['1_DOF'][movement] += 1
                except:
                    try:
                        self.RESULTS[self.cue-1]['OVERSHOOT_RESULTS']['1_DOF'][movement] = 1
                    except: pass
            self.rim_target_plot.set_color('r')
            #self.rim_arc.set_color('r')
            self.activerim = False
            self.rimcomplete = False

        if self.activerim == True:
            if (time.perf_counter() - self.startrim) > Parameters['Fitts Law']['Hold Time']:
                self.rimcomplete = True
                self._check_target()

        self.runcountrim = 1
    
    # DRAW REALTIME RIM ON CIRCLE
    def _update_rim(self):
        self.current_omega = self.max_omega * self.rim_ratio
        start_x = self.x_axis + (self.circle_radius+0.005) * math.sin(math.radians(self.current_omega))
        start_y = self.y_axis + (self.circle_radius+0.005) * math.cos(math.radians(self.current_omega))
        end_x = self.x_axis + (self.circle_radius + self.rim_length) * math.sin(math.radians(self.current_omega))
        end_y = self.y_axis + (self.circle_radius + self.rim_length) * math.cos(math.radians(self.current_omega))
        rim_x = np.linspace(start_x, end_x, 10)
        rim_y = np.linspace(start_y, end_y, 10)
        self.rim_plot[0].set_ydata(rim_y)
        self.rim_plot[0].set_xdata(rim_x)

        '''previous_Arc_length = (math.pi*self.previous_circle_radius) * (self.previous_omega/360)
        Arc_length = (math.pi*self.circle_radius) * (self.current_omega/360)'''
        previous_Arc_length = (self.previous_omega)
        Arc_length = (self.current_omega)

        if abs(previous_Arc_length-Arc_length) > 5:
            arc_diff = abs(abs(previous_Arc_length-Arc_length)-360)
        else:
            arc_diff = abs(previous_Arc_length-Arc_length)

        self.RESULTS[self.cue-1]['PATH']['TRUE_PATH']['1_DOF'] += arc_diff

        self.previous_omega = copy.deepcopy(self.current_omega)
        self.previous_circle_radius = copy.deepcopy(self.circle_radius)
        
    # DRAW AND CHECK TARGET CIRCLE
    def _update_target_circle(self, movement):
        if self.runcountcir == 0:
            self.activecir = False

        self.outer_target_circle_plot.center = (self.x_axis, self.y_axis)
        self.inner_target_circle_plot.center = (self.x_axis, self.y_axis)
        
        if self.init_targets:  
            self.outer_target_circle_plot.radius = self.target_radius+self.target_radius_offset
            self.inner_target_circle_plot.radius = self.target_radius-self.target_radius_offset
        
        in_range = (self.inner_target_circle_plot.radius) <= self.circle_radius and self.circle_radius <= (self.outer_target_circle_plot.radius)

        if in_range:
            if self.activecir == False:
                self.startcir = time.perf_counter()
            self.activecir = True
            self.outer_target_circle_plot.set_color('g')
        else:
            if self.activecir:
                try:
                    self.RESULTS[self.cue-1]['OVERSHOOT_RESULTS']['0_DOF'][movement] += 1
                except:
                    try:
                        self.RESULTS[self.cue-1]['OVERSHOOT_RESULTS']['0_DOF'][movement] = 1
                    except: pass

            self.outer_target_circle_plot.set_color('r')
            self.activecir = False
            self.circlecomplete = False

        if self.activecir == True:
            if (time.perf_counter() - self.startcir) > Parameters['Fitts Law']['Hold Time']:
                self.circlecomplete = True
                self._check_target()

        self.runcountcir = 1

    # DRAW REALTIME CIRCLE
    def _update_circle(self):
        self.circle_plot.center = (self.x_axis, self.y_axis)
        self.circle_plot.radius = self.circle_radius

        self.RESULTS[self.cue-1]['PATH']['TRUE_PATH']['0_DOF'] += abs(self.previous_circle_radius-self.circle_radius)
        self.previous_circle_radius = copy.deepcopy(self.circle_radius)



    # DRAW AND CHECK TARGET RANGE ON XAXIS
    def _update_target_xaxis(self, movement):
        if self.runcountx == 0:
            self.activex = False

        if self.init_targets:
            self.xaxis_target.remove()
            self.xaxis_target = self.cax.fill_betweenx( np.linspace(self.cax.get_ylim()[0], self.cax.get_ylim()[1], 2),
                                                        self.target_centroid_x-self.xaxis_target_range, 
                                                        self.target_centroid_x+self.xaxis_target_range,
                                                        linewidth = 0, 
                                                        alpha = 0.3,
                                                        zorder = 3,
                                                        color = 'b' )
        
        if self.x_axis >= self.target_centroid_x-self.xaxis_target_range and self.x_axis <= self.target_centroid_x+self.xaxis_target_range and self.xaxis_target._facecolors[0][2]: 
            if self.activex == False:
                self.startx = time.perf_counter()
            self.activex = True
            self.xaxis_target.set_color('g')
            self.xaxis_indicator_top_plot[0].set_color('g')
            self.xaxis_indicator_bottom_plot[0].set_color('g')
        elif (self.x_axis < self.target_centroid_x-self.xaxis_target_range or self.x_axis > self.target_centroid_x+self.xaxis_target_range) and not self.xaxis_target._facecolors[0][0]:
            if self.activex:
                try:
                    self.RESULTS[self.cue-1]['OVERSHOOT_RESULTS']['2_DOF'][movement] += 1
                except:
                    try:
                        self.RESULTS[self.cue-1]['OVERSHOOT_RESULTS']['2_DOF'][movement] = 1
                    except: pass
            
            self.xaxis_target.set_color('b')
            self.xaxis_indicator_top_plot[0].set_color('b')
            self.xaxis_indicator_bottom_plot[0].set_color('b')
            self.activex = False
            self.xcomplete = False
        
        if self.activex == True:
            if (time.perf_counter()-self.startx) > Parameters['Fitts Law']['Hold Time']:
                    self.xcomplete = True
                    self._check_target()

        self.runcountx = 1

    # DRAW AND CHECK TARGET RANGE ON YAXIS
    def _update_target_yaxis(self, movement):
        if self.runcounty == 0:
            self.activey = False

        if self.init_targets:
            self.yaxis_target.remove()
            self.yaxis_target = self.cax.fill_between(  np.linspace(self.cax.get_xlim()[0], self.cax.get_xlim()[1], 2),
                                                        self.target_centroid_y-self.yaxis_target_range, 
                                                        self.target_centroid_y+self.yaxis_target_range,
                                                        linewidth = 0, 
                                                        alpha = 0.3,
                                                        zorder = 4,
                                                        color = 'y' )

        if self.y_axis >= self.target_centroid_y-self.yaxis_target_range and self.y_axis <= self.target_centroid_y+self.yaxis_target_range and self.yaxis_target._facecolors[0][0]: 
            if self.activey == False:
                self.starty= time.perf_counter()
            self.activey = True 
            self.yaxis_target.set_color('g')
            self.yaxis_indicator_left_plot[0].set_color('g')
            self.yaxis_indicator_right_plot[0].set_color('g')
        elif (self.y_axis < self.target_centroid_y-self.yaxis_target_range or self.y_axis > self.target_centroid_y+self.yaxis_target_range) and not self.yaxis_target._facecolors[0][0]:
            if self.activey:
                try:
                    self.RESULTS[self.cue-1]['OVERSHOOT_RESULTS']['3_DOF'][movement] += 1
                except:
                    self.RESULTS[self.cue-1]['OVERSHOOT_RESULTS']['3_DOF'][movement] = 1
            
            self.yaxis_target.set_color('y')
            self.yaxis_indicator_left_plot[0].set_color('y')
            self.yaxis_indicator_right_plot[0].set_color('y')
            self.activey = False
            self.ycomplete = False
        
        if self.activey == True:
            if (time.perf_counter()-self.starty) > Parameters['Fitts Law']['Hold Time']:
                self.ycomplete = True
                self._check_target()

        self.runcounty = 1

    def _update_image(self,movement):
        try:
            self.target_text.remove()
        except:
            pass
        self.target_text = self.fig.text((0.7+self.x_axis)/2.4, self.y_axis, self.targetgrip,ha='center', verticalalignment='center',fontsize = 40*self.circle_radius/0.5, transform=self.cax.transAxes)
        '''self.current_image = mpimg.imread( os.path.join( self.CUE_FLDR, self.targetgrip.lower() + '.jpg') )
        im_rad = self.circle_radius*2/3
        self.im = self.cax.imshow(self.current_image, extent=[self.x_axis-im_rad,self.x_axis+im_rad, self.y_axis-im_rad, self.y_axis+im_rad], zorder = 6, alpha = 0.7 )
        patch = Circle((self.x_axis, self.y_axis), radius=im_rad, transform=self.cax.transData)
        self.im.set_clip_path(patch)'''
        self.movtext.remove()
        if type(movement) == list:
            mov_str = ', '.join( movement )
        else:
            mov_str = movement
        self.movtext = self.fig.text(0.9,0.1, mov_str,ha='center',fontsize = 20,bbox=dict(boxstyle = 'round',color='lightsteelblue',alpha=0.3))

    def _update_axis_indicators(self):
        #x top
        xax_t_x = np.linspace(self.x_axis, self.x_axis, 2)
        xax_t_y = np.linspace(self.y_axis+self.circle_radius*self.ax_ind_start, self.y_axis+self.circle_radius*0.98, 2)
        self.xaxis_indicator_top_plot[0].set_xdata(xax_t_x)
        self.xaxis_indicator_top_plot[0].set_ydata(xax_t_y)
        #x bottom
        xax_b_x = np.linspace(self.x_axis, self.x_axis, 2)
        xax_b_y = np.linspace(self.y_axis-self.circle_radius*self.ax_ind_start, self.y_axis-self.circle_radius*0.98, 2)
        self.xaxis_indicator_bottom_plot[0].set_xdata(xax_b_x)
        self.xaxis_indicator_bottom_plot[0].set_ydata(xax_b_y)
        
        #y left
        yax_l_x = np.linspace(self.x_axis-self.circle_radius*self.ax_ind_start, self.x_axis-self.circle_radius*0.98, 2)
        yax_l_y = np.linspace(self.y_axis, self.y_axis, 2)
        self.yaxis_indicator_left_plot[0].set_xdata(yax_l_x)
        self.yaxis_indicator_left_plot[0].set_ydata(yax_l_y)
        #y right
        yax_r_x = np.linspace(self.x_axis+self.circle_radius*self.ax_ind_start, self.x_axis+self.circle_radius*0.98, 2)
        yax_r_y = np.linspace(self.y_axis, self.y_axis, 2)
        self.yaxis_indicator_right_plot[0].set_xdata(yax_r_x)
        self.yaxis_indicator_right_plot[0].set_ydata(yax_r_y)

    def _restart(self):
        try:
            self.endtext.remove()
        except:
            pass
        if 'OPEN' in Parameters['Fitts Law']['CUE_LIST']:
            self.griplist = copy.deepcopy(Parameters['Fitts Law']['GRIP_CUE_LIST'])
        else:
            self.griplist = []
        self._generate_targets()
        self.xaxis_indicator_top_plot[0].set_color('b')
        self.xaxis_indicator_bottom_plot[0].set_color('b')
        self.yaxis_indicator_left_plot[0].set_color('y')
        self.yaxis_indicator_right_plot[0].set_color('y')
        self.x_axis = 0.5
        self.y_axis = 0.5
        self.rim_ratio = 0
        self.circle_radius = copy.deepcopy(self.circle_default_radius)
        
        self.previous_x_axis = copy.deepcopy(self.x_axis)
        self.previous_y_axis = copy.deepcopy(self.y_axis)
        self.previous_omega = 0
        self.previous_circle_radius = copy.deepcopy(self.circle_radius)

        self.trialstart = time.perf_counter()
        if Parameters['Fitts Law']['DoF'][0]['MOVEMENT_TYPE'] != 'Off':  
            self._draw_circle()
        if Parameters['Fitts Law']['DoF'][1]['MOVEMENT_TYPE'] != 'Off':
            self._draw_rim()
        if Parameters['Fitts Law']['DoF'][2]['MOVEMENT_TYPE'] != 'Off':
            self._draw_x()
        if Parameters['Fitts Law']['DoF'][3]['MOVEMENT_TYPE'] != 'Off':
            self._draw_y()
        self.activey = False
        self.activex = False
        self.activecir = False
        self.activerim = False
        self.ycomplete = False
        self.xcomplete = False
        self.rimcomplete = False
        self.circlecomplete = False
    
    def _timer(self,movement):
        if (time.perf_counter()-self.trialstart) >= Parameters['Fitts Law']['Trial Time']:
            self._next_target()

    def _velocity_step(self, step, movement , axis, DoF, step_scalar, max_range, min_range):
        
        if Parameters['Fitts Law']['DoF'][DoF]['MOVEMENT_TYPE'] == 'Hand':
            if self.targetgrip == 'OPEN':
                if movement == 'OPEN':
                    step = step*step_scalar 
                    if axis < max_range:   
                        axis += step
                elif movement in self.griplist:
                    step = step*step_scalar
                    if axis > min_range:
                        axis -= step
            else:
                if movement == 'OPEN':
                    step = step*step_scalar 
                    if axis < max_range:   
                        axis += step
                elif movement == self.targetgrip:
                    step = step*step_scalar
                    if axis > min_range:
                        axis -= step
        elif Parameters['Fitts Law']['DoF'][DoF]['MOVEMENT_TYPE'] == 'Wrist':
            if Parameters['Fitts Law']['DoF'][DoF]['WRIST_DOF'] == 'ROTATION':
                if movement == 'PALM UP':
                    step = step*step_scalar
                    axis += step
                    if axis >= max_range:
                        axis -= 1
                elif movement == 'PALM DOWN':
                    step = step*step_scalar
                    axis -= step
                    if axis < min_range:
                        axis += 1
            elif Parameters['Fitts Law']['DoF'][DoF]['WRIST_DOF'] == 'FLEXION':  
                if movement == 'EXTENSION':
                    step = step*step_scalar
                    if axis < max_range:   
                        axis += step
                elif movement == 'FLEXION':
                    step = step*step_scalar
                    if axis > min_range:
                        axis -= step
        elif Parameters['Fitts Law']['DoF'][DoF]['MOVEMENT_TYPE'] == 'Elbow':
            if movement == 'ELBOW EXTEND':
                step = step*step_scalar
                if axis < max_range:   
                    axis += step
            elif movement == 'ELBOW BEND':
                step = step*step_scalar
                if axis > min_range:
                    axis -= step

        return axis
    
    def _proportional_step(self, prop, movement, axis, DoF, step_scalar, max_range, min_range):

        midpoint = (max_range+min_range)/2
        rad = (max_range-min_range)/2
        if DoF == 1:
            midpoint = min_range

        if movement != 'REST':
            if Parameters['Fitts Law']['DoF'][DoF]['MOVEMENT_TYPE'] == 'Hand':
                if self.targetgrip == 'OPEN':
                    if movement == 'OPEN':
                        axis = midpoint + rad*prop
                    elif movement in self.griplist:
                        axis = midpoint - rad*prop
                elif movement in self.griplist:
                    if movement == 'OPEN':
                        axis = midpoint + rad*prop
                    elif movement == self.targetgrip:
                        axis = midpoint - rad*prop
                        
            elif Parameters['Fitts Law']['DoF'][DoF]['MOVEMENT_TYPE'] == 'Wrist':
                if Parameters['Fitts Law']['DoF'][DoF]['WRIST_DOF'] == 'ROTATION':
                    if movement == 'PALM DOWN': 
                        axis = midpoint - rad*prop
                        if axis < min_range:
                            axis += 1
                    elif movement == 'PALM UP':
                        axis = midpoint + rad*prop
                        if axis >= max_range:
                            axis -= 1
                elif Parameters['Fitts Law']['DoF'][DoF]['WRIST_DOF'] == 'FLEXION':
                    if movement == 'FLEXION': 
                        axis = midpoint - rad*prop
                    elif movement == 'EXTENSION':
                        axis = midpoint + rad*prop

            elif Parameters['Fitts Law']['DoF'][DoF]['MOVEMENT_TYPE'] == 'Elbow':
                if movement == 'ELBOW EXTEND':
                    axis = midpoint - rad*prop
                elif movement == 'ELBOW BEND':
                    axis = midpoint + rad*prop

        if axis > max_range:
            axis = max_range
        if axis < min_range:
            axis = min_range
            
        return axis

    # Collective function call to run axis modules
    def _update_primitives(self, prop, movement):
        self._timer(movement)
        if Parameters['Fitts Law']['Movement Setting'] == 'Velocity':
            step = prop*0.01+0.002
            if Parameters['Fitts Law']['DoF'][0]['MOVEMENT_TYPE'] != 'Off':
                self.circle_radius = self._velocity_step(step = step, movement = movement, axis=self.circle_radius, DoF=0, step_scalar = 0.75, max_range=0.5, min_range=0.075)

            if Parameters['Fitts Law']['DoF'][1]['MOVEMENT_TYPE'] != 'Off':
                self.rim_ratio = self._velocity_step(step = step, movement = movement, axis=self.rim_ratio, DoF=1, step_scalar = 1, max_range=1, min_range=0)

            if Parameters['Fitts Law']['DoF'][2]['MOVEMENT_TYPE'] != 'Off':
                self.x_axis = self._velocity_step(step = step, movement = movement, axis=self.x_axis, DoF=2, step_scalar = 1.25, max_range=1, min_range=0)
                
            if Parameters['Fitts Law']['DoF'][3]['MOVEMENT_TYPE'] != 'Off':
                self.y_axis = self._velocity_step(step = step, movement = movement, axis=self.y_axis, DoF=3, step_scalar = 1.25, max_range=1, min_range=0)

        elif Parameters['Fitts Law']['Movement Setting'] == 'Proportional':
            if Parameters['Fitts Law']['DoF'][0]['MOVEMENT_TYPE'] != 'Off':
                self.circle_radius = self._proportional_step(prop=prop, movement=movement , axis=self.circle_radius, DoF=0, step_scalar = 0.75, max_range=0.5, min_range=0.075)
            if Parameters['Fitts Law']['DoF'][1]['MOVEMENT_TYPE'] != 'Off':
                self.rim_ratio = self._proportional_step(prop=prop, movement=movement , axis=self.rim_ratio, DoF=1, step_scalar = 1, max_range=1, min_range=0)
                
            if Parameters['Fitts Law']['DoF'][2]['MOVEMENT_TYPE'] != 'Off':
                self.x_axis = self._proportional_step(prop=prop, movement=movement , axis=self.x_axis, DoF=2, step_scalar = 1.25, max_range=1, min_range=0)

            if Parameters['Fitts Law']['DoF'][3]['MOVEMENT_TYPE'] != 'Off':
                self.y_axis = self._proportional_step(prop=prop, movement=movement , axis=self.y_axis, DoF=3, step_scalar = 1.25, max_range=1, min_range=0)

    def _execute_updates(self, movement):
        self._update_circle()
        if Parameters['Fitts Law']['DoF'][0]['MOVEMENT_TYPE'] != 'Off':
            self._update_target_circle(movement)
        if Parameters['Fitts Law']['DoF'][1]['MOVEMENT_TYPE'] != 'Off':
            self._update_rim()
            self._update_target_rim(movement)
        if Parameters['Fitts Law']['DoF'][2]['MOVEMENT_TYPE'] != 'Off':
            self._update_target_xaxis(movement)
        if Parameters['Fitts Law']['DoF'][3]['MOVEMENT_TYPE'] != 'Off':
            self._update_target_yaxis(movement)
        self._update_image(movement)
        self._update_axis_indicators()
        self.init_targets = False        

    def _fittslawplot(self, event):

        self.RESULTS['Trial Time'] = Parameters['Fitts Law']['Trial Time']        
        self.RESULTS['DoF_mov_list'] = self.DoF_mov_list
        # self.RESULTS['METHOD'] = int(Parameters['Fitts Law']['Method']), 
        
        self.RESULTS['METHOD'] = self.method_order[self.cue]
        Parameters['Fitts Law']['Method'] = str( self.method_order[self.cue] )
        
        self.date_time = datetime.now().strftime("%m%d%Y%H%M%S")

        self._run_event.set()
        self.clear_axes()

        self.init_targets = True

        self.x_axis = 0.5
        self.y_axis = 0.5

        grid_rows = 9
        grid_columns = 18

        # create GUI layout
        gs = self.fig.add_gridspec( grid_rows, grid_columns )
        self.cax = self.fig.add_subplot(gs[ :, :])
        self.cax.set_aspect(1)
        self.cax.axis('off')
        self.cax.set_xlim([-0.7,1.7])
        self.cax.set_ylim([-0.7,1.7])

        self.circle_radius = copy.deepcopy(self.circle_default_radius)

        self._generate_targets()

        # draw moving circle
        self.circle_plot = self.cax.add_patch(Circle(   (self.x_axis, self.y_axis), 
                                                        self.circle_radius,
                                                        fill = False, 
                                                        linewidth = 2,
                                                        color = 'k',
                                                        zorder = 10))

        self.movtext = self.fig.text(0.9,0.2,'Movement',ha='center',fontsize=20,bbox=dict(boxstyle = 'round',color='lightsteelblue',alpha=0.3))                                               


        if Parameters['Fitts Law']['DoF'][0]['MOVEMENT_TYPE'] != 'Off':

            self._draw_circle()
        
        if Parameters['Fitts Law']['DoF'][1]['MOVEMENT_TYPE'] != 'Off':

            # draw moving rim on circle
            self.rim_length = 0.025
            self.rim_ratio = 0
            self.max_omega = 360
            rim_x = np.linspace(self.x_axis, self.x_axis, 10)
            rim_y = np.linspace(self.y_axis+self.circle_radius, self.y_axis+self.circle_radius+self.rim_length, 10)
            self.rim_plot = self.cax.plot(rim_x, rim_y, linewidth = 3, zorder = 9, color = 'darkslateblue')   
            
            self._draw_rim()

        # draw target x vertical
        if Parameters['Fitts Law']['DoF'][2]['MOVEMENT_TYPE'] != 'Off':
            self._draw_x()

        # draw target y vertical
        if Parameters['Fitts Law']['DoF'][3]['MOVEMENT_TYPE'] != 'Off':
            self._draw_y()

        # draw static axis indicator crosshair
        #x top
        self.ax_ind_start = 2/3
        xax_t_x = np.linspace(self.x_axis, self.x_axis, 2)
        xax_t_y = np.linspace(self.y_axis+self.circle_radius*self.ax_ind_start, self.y_axis+self.circle_radius*0.98, 2)
        self.xaxis_indicator_top_plot = self.cax.plot(xax_t_x, xax_t_y, linewidth = 3, zorder = 6, color = 'b', alpha = 0.75)
        #x bottom
        xax_b_x = np.linspace(self.x_axis, self.x_axis, 2)
        xax_b_y = np.linspace(self.y_axis-self.circle_radius*self.ax_ind_start, self.y_axis-self.circle_radius*0.98, 2)
        self.xaxis_indicator_bottom_plot = self.cax.plot(xax_b_x, xax_b_y, linewidth = 3, zorder = 6, color = 'b', alpha = 0.75)
        #y left
        yax_l_x = np.linspace(self.x_axis-self.circle_radius*self.ax_ind_start, self.x_axis-self.circle_radius*0.98, 2)
        yax_l_y = np.linspace(self.y_axis, self.y_axis, 2)
        self.yaxis_indicator_left_plot = self.cax.plot(yax_l_x, yax_l_y, linewidth = 3, zorder = 6, color = 'y', alpha = 0.75)
        #y right
        yax_r_x = np.linspace(self.x_axis+self.circle_radius*self.ax_ind_start, self.x_axis+self.circle_radius*0.98, 2)
        yax_r_y = np.linspace(self.y_axis, self.y_axis, 2)
        self.yaxis_indicator_right_plot = self.cax.plot(yax_r_x, yax_r_y, linewidth = 3, zorder = 6, color = 'y', alpha = 0.75)

        self.CUE_FLDR = os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ), 'cues', 'FittsLawCues' )

        #self.current_image = mpimg.imread( os.path.join( self.CUE_FLDR, self.targetgrip.lower() + '.jpg') )

        if self.cue == 0:
            self.trialstart = time.perf_counter()

        self._next_target()

        while not self._exit_event.is_set():
            
            data = []
                    # pull samples from queue
            while self._queue.qsize() > 0:
                ctrl = self._queue.get()
                data = ctrl
                self.RESULTS[self.cue-1]['PROPORTIONAL_VALUES'].append(data[0])
                self.RESULTS[self.cue-1]['CLASSIFICATION_VALUES'].append(data[1])
                self.RESULTS[self.cue-1]['FEATURE_VALUES'].append(data[2])
            
            if data:
                if type(data[0]) == list:
                    for i, val in enumerate(data[0]):
                        self._update_primitives( val, data[1][i] )
                else:
                    self._update_primitives( data[0], data[1] ) 
                
                self._execute_updates(data[1])

            if not plt.fignum_exists(self.fig.number):
                self.close()
            plt.pause( 0.001 )
        
        plt.close( self.fig )

    def clear_axes(self):
        try:
            self.submitbtnax.cla() 
            self.submitbtnax.axis('off')
        except:
            pass

        for i in range(len(self.prompt)):
            try:
                self.prompt[i].remove()
            except:
                pass
        
    def initialize(self, reinit = False):
        try:
            self.clear_axes()
        except:
            pass

        self.mode = Parameters['Fitts Law']['Movement Setting']
        self.prompt = []
        for i in range(9):
            self.prompt.append([])

        dof_iter = 0
        height = [0.82, 0.76, 0.67, 0.61, 0.52, 0.46, 0.37, 0.31]
        self.prompt[0] = self.fig.text(0.3,0.9,'Movement Setting: %s' % self.mode,size=20,va='center',bbox=dict(boxstyle = 'round',color='lightsteelblue',alpha=0.3))
        for i in range(1,9,2):
            self.prompt[i] = self.fig.text(0.3,height[i - 1],str(dof_iter) + ' DoF: %s' % Parameters['Fitts Law']['DoF'][dof_iter]['MOVEMENT_TYPE'],size=20,va='center',bbox=dict(boxstyle = 'round',color='lightsteelblue',alpha=0.3))
            if Parameters['Fitts Law']['DoF'][dof_iter]['MOVEMENT_TYPE'] == 'Hand':
                self.prompt[i+1] = self.fig.text(0.3,height[i],'Movement: %s' % Parameters['Fitts Law']['GRIP_CUE_LIST'],size=20,va='center',bbox=dict(boxstyle = 'round',color='lightsteelblue',alpha=0.3))
            elif Parameters['Fitts Law']['DoF'][dof_iter]['MOVEMENT_TYPE'] == 'Wrist':
                self.prompt[i+1] = self.fig.text(0.3,height[i],'Movement: %s' % Parameters['Fitts Law']['DoF'][dof_iter]['WRIST_DOF'],size=20,va='center',bbox=dict(boxstyle = 'round',color='lightsteelblue',alpha=0.3))
            elif Parameters['Fitts Law']['DoF'][dof_iter]['MOVEMENT_TYPE'] == 'Elbow':
                self.prompt[i+1] = self.fig.text(0.3,height[i],'Movement: ELBOW',size=20,va='center',bbox=dict(boxstyle = 'round',color='lightsteelblue',alpha=0.3))
            dof_iter += 1
        
        #create a button and make it respond to click
        self.submitbtnax = plt.axes([0.29, 0.15, 0.25, 0.1])
        self.button = Button(self.submitbtnax, 'Confirm and Start')
        self.button.on_clicked( self._fittslawplot )

        print( "\nTarget Position: %d" % ( self.position_order[self.cue] ) )
        self.position_cue.publish( "%02d" % ( self.position_order[self.cue] ) )


    def dof_callback(self, ind):
        self.widgets[ind]['active'] = self.widgets[ind]['checkbox'].get_status()[0]
        self.initialize( reinit = True)

    def joint_callback(self, ind):
        self.widgets[ind]['joint']['joint'] = self.widgets[ind]['joint']['radio'].value_selected
        self.initialize( reinit = True)

    def _plot(self):
        self.fig = plt.figure(figsize = ( 6, 6 ), tight_layout = 3)
        matplotlib.rcParams['font.size'] = 7
        mngr = plt.get_current_fig_manager()

        self.fig.add_subplot(111)
        plt.xlim(-2,2)
        plt.ylim(-2,2)

        mngr.window.setGeometry(100,100,900,900)

        plt.axis('off')

        self.initialize()

        plt.show()

    def add(self,ctrl):
        try:
            self._queue.put(ctrl,timeout=1e-3)
        except queue.Full:
            pass
    
    @property
    def is_running(self):
        return self._run_event.is_set()

    @property
    def state( self ):
        """
        Returns
        -------
        tuple [numpy.ndarray (n_samples, n_features), list (classes)]
            A newly labeled segment of features
        """
        if self._queue_out.qsize() > 0:
            try:
                return self._queue_out.get( timeout = 1e-3 )
            except self._queue_out.Empty:
                return None

    def close(self):
        self._exit_event.set()
        while self._queue.qsize() > 0:
            try:
                self._queue.get(timeout = 1e-3)
            except queue.Empty:
                pass
        #self._plotter.join()
#
#
#
#
#
#
#
#
#
#

class FittsLawTaskResults:

    RESULTS = {}
    MOVE_LIST = {}
    TYPE_LIST = {}
    TOT_LIST = {}


    def __init__( self, subject = 1 ):

        self.subject = subject
        self.num_trials = 0

        #self._plotter = mp.Process(target = self._plot)

        #TODO
        self.active_dofs = [0, 1]

        self.palette = ['darkslateblue',
                        'firebrick',
                        'forestgreen',
                        'goldenrod',
                        'darkorchid',
                        'lightseagreen',
                        'plum',
                        'lightcoral',
                        'peru',
                        'lightpink',
                        'tan' ]


        for file in os.listdir(f'.\data\Subject_{subject}'):
            delete_next = False
            if fnmatch.fnmatch(file, 'FLT_results*.pkl'):
                with open( f'.\data\Subject_{subject}\{file}', 'rb' ) as pkl:
                    try:
                        self.RESULTS[self.num_trials] = pickle.load( pkl )
                        self.num_trials += 1
                        #print(self.num_trials)
                    except EOFError:
                        delete_next = True
                
                if delete_next:
                    os.remove(f'.\data\Subject_{subject}\{file}')


                # Reading the movements from file
                PATH = os.path.join('data', 'Subject_%s' % SUBJECT, 'movement_hand', 'file_%s.txt' % self.num_trials)
                my_file = open(PATH, "r")
                movement_pattern = my_file.read()
                movement_list = movement_pattern.split("\n")
                ALL_MOVE = {'OPEN': [], 'POWER': [], 'TRIPOD': [], 'KEY': [], 'PINCH': [], 'INDEX': []}
                TOT_MOVE = {'OPEN': 0, 'POWER': 0, 'TRIPOD': 0, 'KEY': 0, 'PINCH': 0, 'INDEX': 0}
                for IDX in range(0, len(movement_list)-1):
                    ALL_MOVE[movement_list[IDX]].append(IDX)
                    TOT_MOVE[movement_list[IDX]] += 1
                self.TYPE_LIST[self.num_trials] = ALL_MOVE
                self.TOT_LIST[self.num_trials] = TOT_MOVE
                self.MOVE_LIST[self.num_trials] = movement_list
        print('TYPE_LIST')
        print(self.TYPE_LIST)
        print('TOT_LIST')
        print(self.TOT_LIST)
        print('MOVE_LIST')
        print(self.MOVE_LIST)

        #TODO
        self.methods = []
        
        tmpCount = 0
        tmpRESULTS = {}
        for session in self.RESULTS.keys():
            for subkey in self.RESULTS[session].keys():
                if type( subkey ) is int:
                    task = subkey
                    if self.RESULTS[session][task]['METHOD'] is not None:
                        if ( tmpCount + self.RESULTS[session][task]['METHOD'] ) not in tmpRESULTS.keys():
                            tmpRESULTS.update( { tmpCount + self.RESULTS[session][task]['METHOD'] : {} } )  # add current session, method 0
                            
                            tmpRESULTS[tmpCount + self.RESULTS[session][task]['METHOD']].update( { 'Trial Time' : self.RESULTS[session]['Trial Time'] } )
                            tmpRESULTS[tmpCount + self.RESULTS[session][task]['METHOD']].update( { 'DoF_mov_list' : self.RESULTS[session]['DoF_mov_list'] } )
                            tmpRESULTS[tmpCount + self.RESULTS[session][task]['METHOD']].update( { 'METHOD' : ( self.RESULTS[session][task]['METHOD'], ) } )
                        
                        tmpRESULTS[tmpCount + self.RESULTS[session][task]['METHOD']].update( { len( tmpRESULTS[tmpCount + self.RESULTS[session][task]['METHOD']] ) - 3 : self.RESULTS[session][task] } ) # subtract 3 because of first 3 keys [Trial Time, DoF_mov_list, METHOD ]

                            # tmpRESULTS.update( { self.RESULTS[session][task]['METHOD'] : {} } )

                        #     tmpRESULTS[self.RESULTS[session][task]['METHOD']].update( { 'Trial Time' : self.RESULTS[0]['Trial Time'] } )
                        #     tmpRESULTS[self.RESULTS[session][task]['METHOD']].update( { 'DoF_mov_list' : self.RESULTS[0]['DoF_mov_list'] } )
                        #     tmpRESULTS[self.RESULTS[session][task]['METHOD']].update( { 'METHOD' : ( self.RESULTS[session][task]['METHOD'], ) } )
                        
                        # tmpRESULTS[ self.RESULTS[session][task]['METHOD'] ].update( { len( tmpRESULTS[ self.RESULTS[session][task]['METHOD'] ] ) - 3 : self.RESULTS[session][task] } ) # subtract 3 because of first 3 keys [Trial Time, DoF_mov_list, METHOD ]
            tmpCount = max( tmpRESULTS.keys() ) + 1

        self.RESULTS = tmpRESULTS
        self.num_trials = tmpCount

        for key in self.RESULTS.keys():
            sub_keys = list(self.RESULTS[key].keys())
            #print(self.RESULTS[key]['DoF_mov_list'][0]['Movements'])
            self.num_cues = len(self.RESULTS[key]['DoF_mov_list'][0]['Movements'])
            if self.RESULTS[key]['METHOD'][0] not in self.methods:
                self.methods.append(self.RESULTS[key]['METHOD'][0])

            for i in sub_keys:
                try:
                    if len(self.RESULTS[key][i]['FEATURE_VALUES']) == 0 or i > len(self.RESULTS[key]['DoF_mov_list'][0]['Movements']):
                        del self.RESULTS[key][i]
                except: pass

        #self._plotter.start()
        self._plot()

    def _plot(self):
        self.plotThroughput()
        self.plotAverage()
        # pass

    def plotIndividual(self):

        rows = 3

        fig, axs = plt.subplots( figsize=(16,10), dpi= 80, nrows= rows, ncols= self.num_trials+1, tight_layout = 6)
        #fig.suptitle(f'FLT Subject {self.subject} Results', fontweight='bold')

        for results_key, trial in enumerate(self.RESULTS.values()):

            # Completion rate

            num_cues = len(trial)
            unsuccessful = 0

            for cue in trial.keys():
                if type(cue) == int:
                    unsuccessful += 1 if not trial[cue]['COMPLETION_RESULTS'] else 0



            success_rate = 100 - (unsuccessful/num_cues) * 100

            x_data = np.linspace( 0, 1, 1 )

            axs[ 0 , int(results_key)].vlines(x_data, ymin=0, ymax=success_rate, color=self.palette[0], alpha=0.7, linewidth=2)
            axs[ 0 , int(results_key)].scatter(x_data, success_rate, s=10, color=self.palette[0], alpha=0.8)

            axs[ 0 , int(results_key)].set(xlim=(-0.5,0.5), ylim=(0,105), ylabel='Percentage')

            # Throughput time
            trial_completion_time = []
            throughput = []
            for cue in trial.keys():
                if type(cue) == int:
                    trial_completion_time.append(trial[cue]['CUE_COMPLETION_TIMES'])
                    throughput.append( [trial['DoF_mov_list'][0]['Target ID'][cue]/trial_completion_time[-1], trial['DoF_mov_list'][1]['Target ID'][cue]/ trial_completion_time[-1]] )

            throughput = np.vstack(throughput)
            x_data = np.linspace( 0, throughput.shape[0]-1, throughput.shape[0] )

            axs[ 1 , int(results_key)].vlines(x_data-0.1, ymin=0, ymax=throughput[:,0], color=self.palette[0], alpha=0.7, linewidth=2)
            axs[ 1 , int(results_key)].scatter(x_data-0.1, throughput[:,0], s=10, color=self.palette[0], alpha=0.8, label = 'Hand')

            axs[ 1 , int(results_key)].vlines(x_data+0.1, ymin=0, ymax=throughput[:,1], color=self.palette[1], alpha=0.7, linewidth=2)
            axs[ 1 , int(results_key)].scatter(x_data+0.1, throughput[:,1], s=10, color=self.palette[1], alpha=0.8, label = 'Wrist')

            axs[ 1 , int(results_key)].set(xlim=(-0.5,throughput.shape[0]), ylim=(0,2), ylabel='Throughput (bit/s)')

            axs[ 1 , int(results_key)].legend()


            # Overshoot
            max_val = 0

            for key, movement in enumerate(Parameters['Fitts Law']['CUE_LIST']):
                if key == 0:
                    continue

                current_overshoot = []

                for cue in trial.keys():
                    if type(cue) == int:
                        templen = len(current_overshoot)
                        for dof in trial[cue]['OVERSHOOT_RESULTS'].values():
                            try:
                                current_overshoot.append(dof[movement])
                            except: pass
                        if templen == len(current_overshoot):
                            current_overshoot.append(0)

                offset = 0.25 - ( key * 0.5/len(Parameters['Fitts Law']['CUE_LIST']) )
                x_data = np.linspace( 0, len(current_overshoot)-1, len(current_overshoot) )

                axs[ 2 , int(results_key)].vlines(x_data+offset, ymin=0, ymax=current_overshoot, color=self.palette[key], alpha=0.7, linewidth=2)
                axs[ 2 , int(results_key)].scatter(x_data+offset, current_overshoot, s=10, color=self.palette[key], alpha=0.8, label = movement)

                max_val = max( [max(current_overshoot), max_val] )

            axs[ 2 , int(results_key)].set(xlim=(-0.5,len(current_overshoot)), ylim=(0,max_val+1), ylabel='Number of Overshots')

            axs[ 2 , int(results_key)].legend()


        plt.show()

    def plotAverage(self):
        rows = 4
        cols = len(self.methods)
        #print( self.methods )

        col_labels = ['Method {}'.format(col) for col in range(1, cols+1)]
        row_labels = ['{}'.format(row) for row in ['Completion rate', 'Throughput', 'Overshoot', 'Efficiency']]

        fig, axes = plt.subplots( figsize=(16,10), dpi= 80, nrows= rows, ncols= cols, tight_layout = 6)
        #fig.suptitle(f'FLT Subject {self.subject} Results', fontweight='bold')

        try:
            for ax, col in zip(axes[0], col_labels):
                ax.set_title(col)

            for ax, row in zip(axes[:,0], row_labels):
                ax.set_ylabel(row, rotation=0, size='large')
        except:
            for ax, row in zip(axes, row_labels):
                ax.set_ylabel(row, rotation=0, size='large')

        trials_in_methods = np.zeros(len(self.methods))

        success_rate = {}
        SEP_SUCCESS_RATE = {}


        throughput_avgs = {}
        throughput_sems = {}
        THROUGH_MOVE = {}
        THROUGH_MOVE_AVGS = {}
        cpt_avgs = {}
        iod_avgs = {}

        overshoot = {}
        overshoot_sems = {}
        optimal_overshoot = {}
        overshoot_percentage = {}
        overshoot_percentage_sems = {}
        overshoot_sep = {}
        overshoot_wrist = {}

        efficiency = {}
        efficiency_sems = {}
        efficiency_avgs = {}
        efficiency_avgs_sems = {}
        EFFICIENCY_MOVE = {}
        EFFICIENCY_MOVE_AVGS = {}



        for i in self.methods:

            success_rate[i] = []

            throughput_avgs[i] = []
            throughput_sems[i] = []
            cpt_avgs[i] = []
            iod_avgs[i] = []

            overshoot[i] = []
            overshoot_sems[i] = []
            optimal_overshoot[i] = []
            overshoot_percentage[i] = []
            overshoot_percentage_sems[i] = []

            efficiency[i] = []
            efficiency_sems[i] = []
            efficiency_avgs[i] = []
            efficiency_avgs_sems[i] = []


        [list_avgs, list_avgs_sep] = effective(self)

        for results_key, trial in enumerate(self.RESULTS.values()):


            key_list = list(self.TYPE_LIST[results_key + 1].keys())
            val_list = list(self.TYPE_LIST[results_key + 1].values())

            # Completion rate
            trials_in_methods[trial['METHOD'][0]-1] += 1

            num_cues = len(trial) - 3  # NOTE: subtract 3 from len for (Trial Time, DoF_mov_list, METHOD)
            unsuccessful = 0

            individual_success = {'OPEN': 0, 'POWER': 0, 'TRIPOD': 0, 'KEY': 0, 'PINCH': 0, 'INDEX': 0}
            individual_success_rate = {}


            #for cue in trial.values():
                #print(cue)


            for cue in trial.keys():
                if type(cue) == int:
                    unsuccessful += 1 if not trial[cue]['COMPLETION_RESULTS'] else 0

            success_rate[trial['METHOD'][0]].append(100 - (unsuccessful/num_cues) * 100)

            # Calculating success based on total and separate movements
            successful = 0
            for cue in trial.keys():
                if type(cue) == int:
                    successful += 1 if trial[cue]['COMPLETION_RESULTS'] else 0
                    if trial[cue]['COMPLETION_RESULTS'] != 0:
                        for check in self.TYPE_LIST[results_key + 1]:
                            if cue in self.TYPE_LIST[results_key + 1][check]:
                                individual_success[check] += 1
            for MOV in self.TOT_LIST[results_key +1]:
                if self.TOT_LIST[results_key +1][MOV] != 0:
                    rate = individual_success[MOV] / self.TOT_LIST[results_key +1][MOV]
                    individual_success_rate[MOV] = round(rate*100)
                SEP_SUCCESS_RATE[results_key +1] = individual_success_rate


            # Throughput time
            trial_completion_time = []
            idx_diff = []
            throughput = []
            individual_tp = {'OPEN': [], 'POWER': [], 'TRIPOD': [], 'KEY': [], 'PINCH': [], 'INDEX': []}
            individual_tp_avgs = {'OPEN': 0, 'POWER': 0, 'TRIPOD': 0, 'KEY': 0, 'PINCH': 0, 'INDEX': 0}

            for cue in trial.keys():
                if type(cue) == int:
                    trial_completion_time.append(trial[cue]['CUE_COMPLETION_TIMES'])
                    idx_diff.append( [trial['DoF_mov_list'][0]['Target ID'][cue], trial['DoF_mov_list'][1]['Target ID'][cue]])
                    if trial[cue]['COMPLETION_RESULTS']:
                        throughput.append( [idx_diff[-1][0]/trial_completion_time[-1], idx_diff[-1][1]/trial_completion_time[-1]] )
                    else: throughput.append( np.nan * np.ones( shape = ( len( self.active_dofs ), ) ) )

            cpt = np.vstack(trial_completion_time)
            iod = np.vstack(idx_diff)
            tp = np.vstack(throughput)


            for item_tp in range( tp.shape[0] ):
                for item_val in val_list:
                    if item_tp in item_val:
                        pos_tp = val_list.index(item_val)
                        individual_tp[key_list[pos_tp]].append(tp[item_tp])

            for things in individual_tp:
                pos1 = 0
                pos2 = 0
                for arrs in individual_tp[things]:
                    pos1 += arrs[0] if not np.isnan( arrs[0] ) else 0
                    pos2 += arrs[1] if not np.isnan( arrs[1] ) else 0
                # print( individual_tp[things], pos1, pos2 )
                if len( individual_tp[things] ):
                    denom = len([x for x in individual_tp[things] if not any(np.isnan(x))])
                    if denom > 0:
                        avgs1 = pos1 / denom
                        avgs2 = pos2 / denom
                        all_avgs = (avgs1 + avgs2) / 2
                    else:
                        all_avgs = 0
                    individual_tp_avgs[things] = all_avgs

            THROUGH_MOVE[results_key + 1] = individual_tp
            THROUGH_MOVE_AVGS[results_key + 1] = individual_tp_avgs

            throughput_sems[trial['METHOD'][0]].append( np.array( [scipy.stats.sem(tp[:,0]), scipy.stats.sem(tp[:,1])] ) )
            throughput_avgs[trial['METHOD'][0]].append(np.nanmean(tp, axis=0))
            cpt_avgs[trial['METHOD'][0]].append(np.nanmean(cpt, axis=0)) # Completion time average
            iod_avgs[trial['METHOD'][0]].append(np.nanmean(iod, axis=0)) # Index of difficulty average

            print('CPT: ', cpt_avgs)
            print('IOD: ', iod_avgs)


            # Overshoot
            max_val = 0
            trial_overshoot = []
            optimal_overshoot[trial['METHOD'][0]].append(num_cues/(len(Parameters['Fitts Law']['CUE_LIST'])-1))

            for key, movement in enumerate(Parameters['Fitts Law']['CUE_LIST']):
                if key == 0:
                    continue

                current_overshoot = 0

                for cue in trial.keys():
                    if type(cue) == int:
                        for dof in trial[cue]['OVERSHOOT_RESULTS'].values():
                            try:
                                current_overshoot+=dof[movement]
                                #print(cue)
                            except: pass

                trial_overshoot.append(current_overshoot)

            overshoot[trial['METHOD'][0]].append(np.hstack(trial_overshoot))
            overshoot_sems[trial['METHOD'][0]].append(scipy.stats.sem(overshoot[trial['METHOD'][0]][-1]))

            overshoot_percentage[trial['METHOD'][0]].append( np.hstack( 100 - (optimal_overshoot[trial['METHOD'][0]][-1]/(overshoot[trial['METHOD'][0]][-1]+optimal_overshoot[trial['METHOD'][0]][-1]))*100 ) )
            overshoot_percentage_sems[trial['METHOD'][0]].append(scipy.stats.sem(overshoot_percentage[trial['METHOD'][0]][-1]))
            #axs[ 2 ].set(xlim=(-0.5,len(current_overshoot)), ylim=(0,max_val+1), ylabel='Number of Overshots')

            # Efficiency
            trial_efficiency = []
            individual_efficiency = {'OPEN': [], 'POWER': [], 'TRIPOD': [], 'KEY': [], 'PINCH': [], 'INDEX': []}
            individual_efficiency_avgs = {'OPEN': 0, 'POWER': 0, 'TRIPOD': 0, 'KEY': 0, 'PINCH': 0, 'INDEX': 0}



            for cue in trial.keys():

                current_efficiency = []

                # NOTE: THIS WILL FAIL IF NO TRIALS ARE COMPLETED
                # if type(cue) == int and trial[cue]['COMPLETION_RESULTS']: # only calculate throughput if successfully completed, otherwise the math isn't guaranteed
                for dof in self.active_dofs:
                    try:
                        # if trial[cue]['PATH']['TRUE_PATH'][f'{dof}_DOF'] == 0:
                            # print('')
                        if type(cue) == int and trial[cue]['COMPLETION_RESULTS']:
                            current_efficiency.append(trial[cue]['PATH']['OPTIMAL_PATH'][f'{dof}_DOF'] / trial[cue]['PATH']['TRUE_PATH'][f'{dof}_DOF'])
                        else: current_efficiency.append( np.nan )
                    except:
                        pass

                trial_efficiency.append(np.hstack(current_efficiency))

                #print(np.hstack(current_efficiency))

                for item in val_list:
                    if cue in item:
                        pos = val_list.index(item)
                        individual_efficiency[key_list[pos]].append(np.hstack(current_efficiency))

            for things in individual_efficiency:
                pos1 = 0
                pos2 = 0
                for arrs in individual_efficiency[things]:
                    pos1 += arrs[0] if not np.isnan( arrs[0] ) else 0
                    pos2 += arrs[1] if not np.isnan( arrs[1] ) else 0
                if len( individual_efficiency[things] ):
                    denom = len( [ x for x in individual_efficiency[things] if not any( np.isnan( x ) ) ] )
                    if denom > 0:
                        avgs1 = pos1 / denom
                        avgs2 = pos2 / denom
                        all_avgs = (avgs1 + avgs2) / 2
                    else: all_avgs = 0
                    individual_efficiency_avgs[things] = all_avgs*100
            EFFICIENCY_MOVE[results_key + 1] = individual_efficiency
            EFFICIENCY_MOVE_AVGS[results_key + 1] = individual_efficiency_avgs

            efficiency[trial['METHOD'][0]].append(np.vstack(trial_efficiency))

            efficiency_sems[trial['METHOD'][0]].append(scipy.stats.sem(efficiency[trial['METHOD'][0]][-1]))
            efficiency_avgs[trial['METHOD'][0]].append(np.nanmean(efficiency[trial['METHOD'][0]][-1], axis=0))

        for i in self.methods:

            x_data = np.linspace( 1, int(trials_in_methods[i-1]), int(trials_in_methods[i-1]) )
            if results_key < 5:
                xlim_max = 5.5
            else:
                xlim_max = trials_in_methods[i-1] + 0.5

            xticks = ['']
            for j in range(int(trials_in_methods[i-1])):
                xticks.append(f'Trial {j+1}')
            xticks.append('')

            axs = []

            if len(self.methods) > 1:
                for j in range(4):
                    axs.append(axes[j, i-1])
            else:
                for j in range(4):
                    axs.append(axes[j])


            # Completion rate

            #axs[ 0 ].vlines(x_data, ymin=0, ymax=success_rate, color=self.palette[0], alpha=1, linewidth=1)
            axs[ 0 ].plot(x_data, success_rate[i], color='k', alpha=0.8)
            axs[ 0 ].scatter(x_data, success_rate[i], s=10, color='k', alpha=1)
            print("success rate")
            print(success_rate[i])

            axs[ 0 ].set(xlim=(0.5,xlim_max), ylim=(0,105), ylabel='Percentage')
            axs[ 0 ].set_ylabel('Percentage', rotation=90)
            #axs[ 0 ].set_title('Completion rate', fontweight = 'bold')
            axs[ 0 ].set_xticklabels(xticks)

            #x_data = np.linspace( 0, throughput.shape[0]-1, throughput.shape[0] )

            # Throughput time
            throughput_avgs_current = np.vstack(throughput_avgs[i]).T
            throughput_sems_current = np.vstack(throughput_sems[i]).T
            cpt_avgs_current = np.vstack(cpt_avgs[i]).T # Completion time current
            iod_avgs_current = np.vstack(iod_avgs[i]).T # Index of difficulty current

            axs[ 1 ].errorbar(x_data-0.05, throughput_avgs_current[0,:], throughput_sems_current[0,:] , color=self.palette[0], linestyle='None', marker='', capsize=3, alpha=0.6, linewidth=1)
            axs[ 1 ].scatter(x_data-0.05, throughput_avgs_current[0,:], s=10, color=self.palette[0], alpha=0.6, label = 'Hand')
            axs[ 1 ].plot(x_data-0.05, throughput_avgs_current[0,:], color=self.palette[0], alpha=0.3)

            axs[ 1 ].errorbar(x_data+0.05, throughput_avgs_current[1,:], throughput_sems_current[1,:] , color=self.palette[1], linestyle='None', marker='', capsize=3, alpha=0.6, linewidth=1)
            axs[ 1 ].scatter(x_data+0.05, throughput_avgs_current[1,:], s=10, color=self.palette[1], alpha=0.6, label = 'Wrist')
            axs[ 1 ].plot(x_data+0.05, throughput_avgs_current[1,:], color=self.palette[1], alpha=0.3)

            axs[ 1 ].errorbar(x_data, np.mean(throughput_avgs_current, axis=0), np.mean(throughput_sems_current, axis=0) , color='k', linestyle='None', marker='', capsize=3, alpha=1, linewidth=1)
            axs[ 1 ].scatter(x_data, np.mean(throughput_avgs_current, axis=0), s=10, color='k', alpha=0.8, label = 'Means')
            axs[ 1 ].plot(x_data, np.mean(throughput_avgs_current, axis=0), color='k', alpha=0.5)

            axs[ 1 ].set(xlim=(0.5,xlim_max), ylim=(0,2))
            axs[ 1 ].set_ylabel('Throughput\n(bit/s)', rotation=90)
            #axs[ 1 ].set_title('Throughput', fontweight = 'bold')
            axs[ 1 ].set_xticklabels(xticks)


            # Overshoot
            overshoot_percentage_current = np.vstack(overshoot_percentage[i])
            overshoot_current = np.vstack(overshoot[i])

            # Removing Palm down and Palm up
            for item in range(len(overshoot_current)):
                overshoot_move_current = {'OPEN': 0, 'POWER': 0, 'TRIPOD': 0, 'KEY': 0, 'PINCH': 0, 'INDEX': 0}
                overshoot_wrist_current = {"PALM DOWN" : 0, "PALM UP" : 0}
                new_item = overshoot_current[item].tolist()
                overshoot_wrist_current["PALM DOWN"] = new_item[-2]
                overshoot_wrist_current["PALM UP"] = new_item[-1]
                del new_item[-2:]
                move_name_list = list(overshoot_move_current)
                for items in range(len(new_item)):
                    move_name = move_name_list[items]
                    if new_item[items] > 0:
                        overshoot_move_current[move_name] = new_item[items]

                #print(overshoot_move_current)
                overshoot_sep[item + 1] = overshoot_move_current
                overshoot_wrist[item + 1] = overshoot_wrist_current


            for idx in range(overshoot_current.shape[1]):
                if idx%2 == 0:
                    offset = -0.05*np.ceil(idx/2)
                else:
                    offset = 0.05*np.ceil(idx/2)

                axs[ 2 ].scatter(x_data+offset, overshoot_current[:,idx], s=10, color=self.palette[idx], alpha=0.6, label = Parameters['Fitts Law']['CUE_LIST'][idx+1])
                axs[ 2 ].plot(x_data+offset, overshoot_current[:,idx], color=self.palette[idx], alpha=0.3)

            axs[ 2 ].errorbar(x_data, np.mean(overshoot_current, axis=1), overshoot_sems[i] , color='k', linestyle='None', marker='', capsize=3, alpha=1, linewidth=1)
            axs[ 2 ].scatter(x_data, np.mean(overshoot_current, axis=1), s=10, color='k', alpha=0.8, label = 'Means')
            axs[ 2 ].plot(x_data, np.mean(overshoot_current, axis=1), color='k', alpha=0.5)

            axs[ 2 ].set(xlim=(0.5,xlim_max), ylim=(0,overshoot_current.max()+1))
            #axs[ 2 ].set_title('Overshoot', fontweight = 'bold')
            axs[ 2 ].set_ylabel('Number of times\nYou Shot Over', rotation=90)

            # Efficiency
            efficiency_avgs_current = np.vstack(efficiency_avgs[i]).T*100
            efficiency_sems_current = np.vstack(efficiency_sems[i]).T*100

            axs[ 3 ].errorbar(x_data-0.05, efficiency_avgs_current[0,:], efficiency_sems_current[0,:] , color=self.palette[0], linestyle='None', marker='', capsize=3, alpha=0.6, linewidth=1)
            axs[ 3 ].scatter(x_data-0.05, efficiency_avgs_current[0,:], s=10, color=self.palette[0], alpha=0.6, label = 'Hand')
            axs[ 3 ].plot(x_data-0.05, efficiency_avgs_current[0,:], color=self.palette[0], alpha=0.3)

            axs[ 3 ].errorbar(x_data+0.05, efficiency_avgs_current[1,:], efficiency_sems_current[1,:] , color=self.palette[1], linestyle='None', marker='', capsize=3, alpha=0.6, linewidth=1)
            axs[ 3 ].scatter(x_data+0.05, efficiency_avgs_current[1,:], s=10, color=self.palette[1], alpha=0.6, label = 'Wrist')
            axs[ 3 ].plot(x_data+0.05, efficiency_avgs_current[1,:], color=self.palette[1], alpha=0.3)

            axs[ 3 ].errorbar(x_data, np.mean(efficiency_avgs_current, axis=0), np.mean(efficiency_sems_current, axis=0) , color='k', linestyle='None', marker='', capsize=3, alpha=1, linewidth=1)

            print("standard error of the mean")
            print(np.mean(efficiency_sems_current, axis = 0))
            axs[ 3 ].scatter(x_data, np.mean(efficiency_avgs_current, axis=0), s=10, color='k', alpha=0.8, label = 'Means')

            print("overshoot rate")
            overshoot_rate = np.mean(overshoot_current, axis=1)/successful*100
            print(overshoot_rate)

            axs[ 3 ].plot(x_data, np.mean(efficiency_avgs_current, axis=0), color='k', alpha=0.5)

            print("Efficiency")
            print(np.mean(efficiency_avgs_current, axis = 0))

            axs[ 3 ].set(xlim=(0.5,xlim_max), ylim=(0,105))
            axs[ 3 ].set_ylabel('Percentage', rotation=90)
            #axs[ 3 ].set_title('Efficiency', fontweight = 'bold')
            axs[ 3 ].set_xticklabels(xticks)

            axs[ 1 ].legend()
            axs[ 2 ].legend()
            axs[ 3 ].legend()


            PATH3 = os.path.join('data', 'Subject_%s' % SUBJECT)
            with open(os.path.join(PATH3, 'RESULT.txt'), "w") as f:
                f.write("TEST RESULTS\n")
                f.write("\n1. Completion Rate\n")
                f.write("- Total:\n")
                for j in range(len(success_rate[i])):
                    jj = j + 1
                    f.write("\tDAY %d: " % jj +  str(round(success_rate[i][j])) + '%\n')
                f.write("- Separate Movements:\n")
                for i in SEP_SUCCESS_RATE:
                    f.write("\tDAY %d: " % i +str(SEP_SUCCESS_RATE[i]) + '\n')
                f.write("\n2. Overshoot Number\n")
                f.write("- Total:\n")
                for j in range(len(overshoot_current)):
                    jj = j + 1
                    f.write("\tDAY %d: " % jj + str(np.sum(overshoot_current, axis=1)[j]) + '\n')
                f.write("- Separate Movements:\n")
                for i in overshoot_sep:
                    f.write("\tDAY %d: " % i + str(overshoot_sep[i]) + " " + str(overshoot_wrist[i]) + '\n')
                f.write("\n3. Throughput\n")
                f.write("- Total:\n")
                for j in range(len(np.mean(throughput_avgs_current, axis=0))):
                    jj = j + 1
                    f.write("\tDAY %d: " % jj + str(np.mean(throughput_avgs_current, axis=0)[j]) + '\n')
                f.write("- Separate Movements:\n")
                for i in THROUGH_MOVE_AVGS:
                    f.write("\tDAY %d: " % i + str(THROUGH_MOVE_AVGS[i]) + '\n')
                f.write("\n4. Efficiency\n")
                f.write("- Total:\n")
                for j in range(len(np.mean(efficiency_avgs_current, axis = 0))):
                    jj = j + 1
                    f.write("\tDAY %d: " % jj + str(np.mean(efficiency_avgs_current, axis=0)[j]) + '\n')
                f.write("- Separate Movements:\n")
                for i in EFFICIENCY_MOVE_AVGS:
                    f.write("\tDAY %d: " % i + str(EFFICIENCY_MOVE_AVGS[i]) + '\n')
                f.write("\n5. Effective Movement\n")
                f.write("- Total:\n")
                for j in range(len(list_avgs)):
                    jj = j + 1
                    f.write("\tDAY %d: " % jj + str(list_avgs[jj]) + '\n')
                f.write("- Separate Movements:\n")
                for i in list_avgs_sep:
                    f.write("\tDAY %d: " % i + str(list_avgs_sep[i]) + '\n')
                f.write("\n6. Completion Time:\n")
                for j in range(len(np.mean(cpt_avgs_current, axis=0))):
                    jj = j + 1
                    f.write("\tDAY %d: " % jj + str(np.mean(cpt_avgs_current, axis=0)[j]) + '\n')
                f.write("\n7. Index of Difficulty:\n")
                for j in range(len(np.mean(iod_avgs_current, axis=0))):
                    jj = j + 1
                    f.write("\tDAY %d: " % jj + str(np.mean(iod_avgs_current, axis=0)[j]) + '\n')


        plt.show()



    def plotThroughput(self):
        rows = self.num_trials
        cols = self.num_cues

        fig, axs = plt.subplots( figsize=(16,10), dpi= 80, nrows= rows, ncols= 3, tight_layout = 6)

        throughput_avgs = []
        throughput_sems = []

        min_ID = 1
        max_ID = 6

        for results_key, trial in self.RESULTS.items():

            # Throughput time
            trial_completion_time = []
            ID = []

            for cue in trial.keys():
                if type(cue) == int and trial[cue]['COMPLETION_RESULTS']: # only compute throughput for successful completions
                    trial_completion_time.append(trial[cue]['CUE_COMPLETION_TIMES'])
                    print(cue, trial['DoF_mov_list'][0]['Target ID'])
                    ID.append( [trial['DoF_mov_list'][0]['Target ID'][cue], trial['DoF_mov_list'][1]['Target ID'][cue]] )
            print( '\n' )
            if len( ID ):
                ID = np.vstack(ID)

                try:
                    current_0 = axs[results_key, 0]
                    current_1 = axs[results_key, 1]
                    current_2 = axs[results_key, 2]

                except:
                    current_0 = axs[0]
                    current_1 = axs[1]
                    current_2 = axs[2]

                current_0.scatter(ID[:,0], trial_completion_time, s=50, color=self.palette[0], alpha=1)
                current_1.scatter(ID[:,1], trial_completion_time, s=50, color=self.palette[0], alpha=1)
                current_2.scatter(np.mean(ID, axis=1), trial_completion_time, s=50, color=self.palette[0], alpha=1)

                x_data = np.linspace( min_ID, max_ID, max_ID+1 )
                m, b = np.polyfit(ID[:,0], trial_completion_time, 1)
                current_0.plot(x_data, m*(x_data) + b, color='k', alpha = 0.8)

                m, b = np.polyfit(ID[:,1], trial_completion_time, 1)
                current_1.plot(x_data, m*(x_data) + b, color='k', alpha = 0.8)

                m, b = np.polyfit(np.mean(ID, axis=1), trial_completion_time, 1)
                current_2.plot(x_data, m*(x_data) + b, color='k', alpha = 0.8)

                current_0.set(xlim=(min_ID,max_ID), ylim=(0,self.RESULTS[results_key]['Trial Time']), ylabel='Completion Time', xlabel='Index of Difficulty')
                current_1.set(xlim=(min_ID,max_ID), ylim=(0,self.RESULTS[results_key]['Trial Time']), ylabel='Completion Time', xlabel='Index of Difficulty')
                current_2.set(xlim=(min_ID,max_ID), ylim=(0,self.RESULTS[results_key]['Trial Time']), ylabel='Completion Time', xlabel='Index of Difficulty')

        plt.show()

if __name__ == '__main__':

    import time

    plotter = FittsLawTaskResults(SUBJECT)
    exit()
    SESSION = 1 # NOTE: This variable may need to be looked at later
    # FILE_DIRECTORY = os.path.join( os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) ), 'Data', 'Subject_%s' % SUBJECT )
    FILE_DIRECTORY = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), 'data', 'Subject_%s' % SUBJECT ) # if rescu.py is in main folder

    PREVIOUS_FILE = os.path.join( FILE_DIRECTORY, 'Subject_%s_Session_%s.pkl' % ( SUBJECT, SESSION - 1 ) )

    # download training data
    print( 'Importing training data...', end = '', flush = True)
    # check if longterm updating is active
    if Parameters['Misc.']['CARRY_UPDATES']:
        # check if updates have occured in previous sessions and load latest set
        practice_session = 1
        while True:
            PREVIOUS_FILE = os.path.join( FILE_DIRECTORY, 'Subject_%s_Practice_%s.pkl' % ( SUBJECT, practice_session ) )
            if os.path.isfile(PREVIOUS_FILE):
                practice_session += 1
            else:
                SAVE_FILE = os.path.join( FILE_DIRECTORY, 'Subject_%s_Practice_%s.pkl' % ( SUBJECT, practice_session ) )
                PREVIOUS_FILE = os.path.join( FILE_DIRECTORY, 'Subject_%s_Practice_%s.pkl' % ( SUBJECT, practice_session - 1 ) )
                if practice_session > 1:
                    print( 'Importing previous session data...', end = '', flush = True)
                break


    tdg = TrainingDataGenerator()
    # compute training features and labels
    Xtrain, ytrain = tdg.emg_pre_extracted( CLASSES = Parameters['Fitts Law']['CUE_LIST'],
                                            PREVIOUS_FILE = PREVIOUS_FILE,
                                            practice_session = practice_session,
                                            CARRY_UPDATES = Parameters['Misc.']['CARRY_UPDATES'],
                                            total_training_samples = Parameters['Classification']['CLUSTER_TRAINING_SAMPLES'],
                                            onset_threshold_enable = Parameters['Classification']['ONSET_THRESHOLD'],
                                            onset_scaler = Parameters['Calibration']['CALIB_ONSET_SCALAR'],
                                            gain = Parameters['General']['ELECTRODE_GAIN'],
                                            fft_length = Parameters['General']['FFT_LENGTH'],
                                            EMG_SCALING_FACTOR = EMG_SCALING_FACTOR,
                                            emg_window_size = Parameters['General']['WINDOW_SIZE'],
                                            emg_window_step = Parameters['General']['WINDOW_STEP'],
                                            CALIBRATION = Parameters['Calibration']['CALIB_METHOD'],
                                            FL = True,
                                            subject = SUBJECT)

    emg_window = []

    #create data filters
    print( 'Creating EMG feature filters...', end = '', flush = True )
    td5 = TimeDomainFilter()
    if Parameters['General']['FFT_LENGTH'] in ['Custom Adapt']:
        fft = FourierTransformFilter( fftlen = 64 )
    else:
        fft = FourierTransformFilter( fftlen = int(Parameters['General']['FFT_LENGTH']) )
    print( 'Done!' )

    # create EMG interface
    print( 'Creating device interface...', end = '', flush = True )
    if Parameters['General']['SOURCE'] == 'Sense':
        emg = SenseController( name = local_surce_info.sense_name,
                                    mac = local_surce_info.sense_mac,
                                    gain = Parameters['General']['ELECTRODE_GAIN'],
                                    num_electrodes = Parameters['General']['DEVICE_ELECTRODE_COUNT'],
                                    srate = Parameters['General']['SAMPLING_RATE'] )
    elif Parameters['General']['SOURCE'] == 'Myo Band':
        emg = MyoArmband( name = local_surce_info.myo_name, mac = local_surce_info.myo_mac )
    else:
        raise RuntimeError( 'Unsupported EMG interface provided' )
    print( 'Done!' )

    # initiate onset threshold
    print( 'Create onset threshold training data...', end = '', flush = True )
    if not Parameters['Classification']['ONSET_THRESHOLD']:
        onset_threshold_calculation = Onset( onset_scalar = 1.1 )
    else:
        onset_threshold_calculation = Onset( onset_scalar = Parameters['Classification']['ONSET_SCALAR'] )

    onset_threshold = onset_threshold_calculation.onset_threshold(  Xtrain = Xtrain,
                                                                    ytrain = ytrain,
                                                                    gain = Parameters['General']['ELECTRODE_GAIN'],
                                                                    classifier = Parameters['Classification']['CLASSIFIER'])
    print( 'Done!' )

    #create Smoothor
    print( 'Create Smoothor...', end = '', flush = True )
    smoothor = Smoothing( Parameters )
    print( 'Done!' )


    PROPORTIONAL_LOW = []
    PROPORTIONAL_HIGH = []
    PERCENTAGES_LOW = []
    PERCENTAGES_HIGH = []
    for cl in Parameters['Fitts Law']['CUE_LIST']:
        if cl != 'REST':
            PROPORTIONAL_LOW.append(Parameters['Proportional Control'][cl.replace(' ', '_')+'_VEL_LOW'][0])
            PROPORTIONAL_HIGH.append(Parameters['Proportional Control'][cl.replace(' ', '_')+'_VEL_HIGH'][0])
            PERCENTAGES_LOW.append(Parameters['Proportional Control'][cl.replace(' ', '_')+'_VEL_LOW'][1])
            PERCENTAGES_HIGH.append(Parameters['Proportional Control'][cl.replace(' ', '_')+'_VEL_HIGH'][1])

    # train classifier
    print( 'Training classifier...', end = '', flush = True )
    class_subdict_size = int( Parameters['Classification']['CLUSTER_TRAINING_SAMPLES'] )
    run_classifier = RunClassifier(   class_subdict_size = class_subdict_size,
                                            Xtrain = Xtrain,
                                            ytrain = ytrain,
                                            classes = Parameters['Fitts Law']['CUE_LIST'],
                                            perc_lo = [i for i in PERCENTAGES_LOW],
                                            perc_hi = [i for i in PERCENTAGES_HIGH],
                                            threshold_ratio = Parameters['Classification']['THRESHOLD_RATIO'] )
    print( 'Done!' )

    emg.run()

    FLT = FittsLawTask()

    while not FLT._exit_event.is_set():
        if FLT._run_event.is_set():
            data = emg.state
            # check for new data
            if data is not None:
                emg_window.append( data[:Parameters['General']['DEVICE_ELECTRODE_COUNT']].copy() )
                # if window full
                if len( emg_window ) == Parameters['General']['WINDOW_SIZE']:
                    # extract features
                    win = np.vstack( emg_window ) / EMG_SCALING_FACTOR
                    #feat = np.hstack( [ td5.filter( win ), fft.filterAdapt( win ) ] )
                    feat = np.hstack( [ td5.filter( win ), fft.filter( win ) ] )[:-8]

                    emg_window = emg_window[Parameters['General']['WINDOW_STEP']:]

                    CLASSIFIER = Parameters['Classification']['CLASSIFIER']
                    if CLASSIFIER == 'Simultaneous' or CLASSIFIER == 'Spatial' or CLASSIFIER == 'EASRC':
                        feat_filtered = smoothor.filter( feat )
                    else:
                        feat_filtered = feat

                    # classification (up to here)
                    pred = run_classifier.emg_classify(feat_filtered, onset_threshold, CLASSIFIER, Parameters['Classification']['ONSET_THRESHOLD'],True)
                    for i in range(len(Parameters['Fitts Law']['CUE_LIST'])):
                        if pred[0] == i:
                            movement = Parameters['Fitts Law']['CUE_LIST'][i]
                    try:
                        prop = pred[1][0][pred[0]]
                    except:
                        prop = 0

                    FLT.add(tuple([prop,movement, feat_filtered]))



    emg.close()

    pass