import copy
import multiprocessing as mp
import os
import pickle
import sys
import time
import traceback
import warnings
from collections import deque
import matplotlib
import numpy as np
from scipy.io import savemat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.transform import Rotation as R
import math
import copy

matplotlib.use( 'QT5Agg')
from matplotlib import cm
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button, RadioButtons, Slider, TextBox, CheckButtons
from matplotlib.patches import Rectangle
###patch start###
from mpl_toolkits.mplot3d.axis3d import Axis
from sklearn.decomposition import FastICA

if not hasattr(Axis, "_get_coord_info_old"):
    def _get_coord_info_new(self, renderer):
        mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, centers, deltas, tc, highs
    Axis._get_coord_info_old = Axis._get_coord_info  
    Axis._get_coord_info = _get_coord_info_new
###patch end##
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DownSampler import DownSampler
from FeatureSmoothing import Smoothing
from FourierTransformFilter import FourierTransformFilter
from MyoArmband import MyoArmband
from RunClassifier import RunClassifier
from SenseController import SenseController
from TimeDomainFilter import TimeDomainFilter
from TrainingDataGeneration import TrainingDataGenerator

from local_addresses import address
from global_parameters import global_parameters

from random import randint
import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D

local_source_info = address()
downsample = DownSampler()

warnings.simplefilter(action='ignore', category=FutureWarning)

class SepRepPlot:
    def __init__(   self, CLASSES, SUBJECT, Xtrain, ytrain, CLASSIFIER, ALL_CLASSES, CLUSTER_TRAINING_SAMPLES, 
                    realtime = False, Parameters = {}, perc_lo = [], perc_hi = [], 
                    threshold_ratio = 0.3, recalculate_RLDA = True, proj = [], stdscaler = [], classifier_obj = False):
    
        FILE_DIRECTORY = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), 'data', 'Subject_%s' % SUBJECT ) # if rescu.py is in main folder

        self.SUBJECT = SUBJECT
        self.realtime = realtime

        self.show_interface = not realtime

        self.new_train = True
        self.CLUSTER_TRAINING_SAMPLES = CLUSTER_TRAINING_SAMPLES
        
        self.CLASSES = CLASSES

        self.show_clusters = [True for _ in range(len(self.CLASSES))]
        self.visualize = [True for _ in range(len(self.CLASSES))]

        self.visualization_mode = 'Separability'
        self.run_repetitive_testing = False
        self.cue_length = 5
        self.num_reps = 3

        self.pause = False

        self.results_dictionary = {}

        self.num_classes = len( self.CLASSES )
        self.class_subdict_size = int( 50 )#self.CLUSTER_TRAINING_SAMPLES )

        self.new_data_buffer = deque()

        global_params = global_parameters()
        self.palette = global_params.palette

        self.all_classes = ALL_CLASSES
        
        self.colors = {}
        for key, cl in enumerate(self.all_classes):
            try:
                self.colors[cl] = self.palette[key]
            except:
                print( 'There are more classes than colors specified in the reviewer' )

        self.colors['new class'] = 'lightgray'

        self.warning_texts = ['Your new movement is too similar to rest, try adding a little more power',
                              'Your new movement is too similar to ',
                              'RECORDING']
        self.selected_texts = [False, False, False]
        
        self.feat_buffer = deque()

        # compute training features and labels
        self.Xtrain, self.ytrain = Xtrain, ytrain

        self.classifier = CLASSIFIER
        self.visualization_state = 'Clusters'   

        self.perc_lo = perc_lo
        self.perc_hi = perc_hi
        self.threshold_ratio = threshold_ratio     

        self.Parameters = Parameters
        if not self.realtime:
            self.smoothor = Smoothing( self.Parameters )

        self.data_scatter = []

        self.currentlyGatheringNewData = False

        self.recalculate_RLDA = recalculate_RLDA
        if self.recalculate_RLDA:
            if type(classifier_obj) == bool:
                self.stdscaler = MinMaxScaler()
                self.proj = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
                self.reinit_classifier()
            else:
                self.stdscaler = stdscaler
                self.proj = proj
                self.run_classifier = classifier_obj
                self.class_pairs = self.run_classifier.SpatialClassifier.class_pairs
                self.radius = self.run_classifier.SpatialClassifier.radius
                self.proj = self.run_classifier.SpatialClassifier.proj
                self.onset_threshold = self.threshold_ratio

        else:
            self.stdscaler = stdscaler
            self.proj = proj
            self.run_classifier = classifier_obj
            self.class_pairs = self.run_classifier.SpatialClassifier.class_pairs
            self.radius = self.run_classifier.SpatialClassifier.radius
            self.proj = self.run_classifier.SpatialClassifier.proj
            self.onset_threshold = self.threshold_ratio

        

        # multiprocessing variables
        self._run_multiproc()

    def reinit_classifier(self):
        self.run_classifier = RunClassifier(   class_subdict_size = self.class_subdict_size, 
                                                Xtrain = self.Xtrain, 
                                                ytrain = self.ytrain, 
                                                classes = self.CLASSES,
                                                perc_lo = self.perc_lo,
                                                perc_hi = self.perc_hi,
                                                threshold_ratio = self.threshold_ratio )

        self.class_pairs, self.radius = self.run_classifier.init_classifiers(   self.Xtrain, self.ytrain, self.CLASSES, 
                                                                                perc_lo = self.perc_lo,
                                                                                perc_hi = self.perc_hi,
                                                                                threshold_ratio = self.threshold_ratio)

        self.proj = self.run_classifier.SpatialClassifier.proj

        #self.threshold_radius = self.run_classifier.SpatialClassifier.threshold_radius
        self.onset_threshold = self.threshold_ratio

    def _get_class_color(self):

        current_colors = []
        for i in range(self.num_classes):
            current_colors.append(self.colors[self.CLASSES[i]])

        current_colors.append(self.colors['new class'])

        return current_colors

    def _run_multiproc(self):

        self._queue = mp.Queue()
        self._queue2 = mp.Queue()
        self._queue3 = mp.Queue()
        self._queue_segment = mp.Queue()
        self._queue_classifier = mp.Queue()

        self._exit_event = mp.Event()
        self._plotter = mp.Process( target = self._plot )

        # start plotting process
        self._plotter.start()

    def _plot(self):

        self.current_class = 0
        
        def draw_trail(Xmarker):
            self.trailX.append(Xmarker._offsets3d[0])
            self.trailY.append(Xmarker._offsets3d[1])
            self.trailZ.append(Xmarker._offsets3d[2])
            
            if self.trail_mode == 0:
                if self.prev_trail_mode == 1:
                    for i in range(self.trail_length):
                        self.trail[i].remove()
                elif self.prev_trail_mode == 2:
                    #self.dottrail.remove()
                    self.linetrail[0].remove()
            
            elif self.trail_mode == 1:
                if self.prev_trail_mode == 2:
                    #self.dottrail.remove()
                    self.linetrail[0].remove()
                if self.prev_trail_mode != 1:
                    self.trail = []
                    for i in range(self.trail_length):
                        self.trail.append(self.sax.scatter( 0, 0, 0, [i], marker='X', s=200, c='white', edgecolor = 'k', alpha = 0 ))
                else:
                    for i in range(self.trail_length - 1):
                        self.trail[self.trail_length - 1 - i]._offsets3d = self.trail[self.trail_length - 2 - i]._offsets3d
                        self.trail[self.trail_length - 1 - i]._sizes3d = self.trail[self.trail_length - 2 - i]._sizes3d
                        self.trail[self.trail_length - 1 - i]._facecolor3d = self.trail[self.trail_length - 2 - i]._facecolor3d
                        self.trail[self.trail_length - 1 - i].set_alpha(1 / (self.trail_length - 1 - i))
                    self.trail[0]._offsets3d = Xmarker._offsets3d
                    self.trail[0]._sizes3d = Xmarker._sizes3d
                    self.trail[0]._facecolor3d = Xmarker._facecolor3d
                    self.trail[0].set_alpha(1)

            elif self.trail_mode == 2:
                if self.prev_trail_mode == 1:
                    for i in range(self.trail_length):
                        self.trail[i].remove()
                if self.prev_trail_mode != 2:
                    #self.dottrail = self.sax.scatter(self.trailX[1:], self.trailY[1:], self.trailZ[1:], [0], c='tomato', alpha = 0.85, edgecolor = 'darkred' )
                    self.linetrail = self.sax.plot3D(np.hstack(self.trailX[1:]), np.hstack(self.trailY[1:]), np.hstack(self.trailZ[1:]), c='tomato', alpha = 0.7)
                else:
                    #self.dottrail._offsets3d = np.concatenate((self.dottrail._offsets3d, np.array((Xmarker._offsets3d[0], Xmarker._offsets3d[1], Xmarker._offsets3d[2]))), axis=1)
                    try:
                        self.linetrail[0]._verts3d = np.concatenate((self.linetrail[0]._verts3d, np.array((Xmarker._offsets3d[0], Xmarker._offsets3d[1], Xmarker._offsets3d[2]))), axis=1)
                    except: pass
            self.prev_trail_mode = self.trail_mode

        def projection():
            self.pause = True
            self.sax.clear()
            if self.num_classes > 3:

                if self.recalculate_RLDA:                    
                    Xtrain_scaled = self.stdscaler.fit_transform(self.Xtrain)
                    self.Xtrain_scatter = self.proj.fit_transform(Xtrain_scaled, self.ytrain)
                else:
                    Xtrain_scaled = self.stdscaler.transform(self.Xtrain)
                    self.Xtrain_scatter = self.proj.transform(Xtrain_scaled)
            else:
                self.Xtrain_scatter = self.ica_scatter.fit_transform(self.Xtrain[:,:8])

            if self.run_repetitive_testing:
                self.Xtrain_scatter = self.Xtrain_scatter[:,:3]
                for i in range(self.num_classes):
                    self.Xtrain_scatter[self.ytrain==i,:] -= self.cluster_mid_points[self.current_class-1][:3]
                    self.Xtrain_scatter[self.ytrain==i,:] = np.matmul(self.Xtrain_scatter[self.ytrain==i,:], self.line_rotation_matrices[self.current_class-1])

            self.trail_mode = 0
            self.prev_trail_mode = 0
            self.trailX = []
            self.trailY = []
            self.trailZ = []
            self.trail_length = 10
            self.dottrail_index = 0

            self.scat = []
            self.labels = []

            self.labels_starting_points = []
            self.label_colors = []
            for i in range(self.num_classes + 2):
                if i == self.num_classes:
                    self.scat.append(self.sax.scatter( 0, 0, 0, [i], marker='X', s=200, c='white', edgecolor = 'k' ))

                elif i == self.num_classes + 1:
                    self.scat.append(self.sax.scatter( 0, 0, 0, [i], c='darkgrey', alpha = 0, edgecolor = 'k' ))

                else:
                    temp_class = self.Xtrain_scatter[self.ytrain == i,:]
                    self.labels_starting_points.append( [np.mean(temp_class[:,0]), np.mean(temp_class[:,1]),  np.mean(temp_class[:,2])+np.linalg.norm(self.radius[i])*2] )

                    if self.show_clusters[i] or self.visualization_mode != 'Repeatability':

                        if self.visualization_state == "Clusters":
                            self.scat.append(self.sax.scatter( temp_class[:,0], temp_class[:,1], temp_class[:,2], self.ytrain[self.ytrain == i], c = self._get_class_color()[i] ))
                            self.label_colors.append((self.scat[-1]._facecolor3d[0][0], self.scat[-1]._facecolor3d[0][1], self.scat[-1]._facecolor3d[0][2]))# , temp_class[:,2].max() + (temp_class[:,2].max() - np.mean(temp_class[:,2])) / 3

                    else:
                        
                        if self.visualization_state == "Clusters":
                            self.scat.append(self.sax.scatter( temp_class[:,0], temp_class[:,1], temp_class[:,2], self.ytrain[self.ytrain == i], c = self._get_class_color()[i] ))
                            self.label_colors.append((self.scat[-1]._facecolor3d[0][0], self.scat[-1]._facecolor3d[0][1], self.scat[-1]._facecolor3d[0][2]))# , temp_class[:,2].max() + (temp_class[:,2].max() - np.mean(temp_class[:,2])) / 3

                    self.labels.append(self.sax.text( self.labels_starting_points[i][0], self.labels_starting_points[i][1],  self.labels_starting_points[i][2], self.CLASSES[i].upper(), None, size = 'x-large', ha='center', weight='semibold'))


            self.projection_lines = []
            self.mean_lines = []
            cluster_means = [np.mean( self.Xtrain_scatter[self.ytrain==0,:] , axis=0 )]
            for i in range(self.num_classes):
                if i != 0:
                    cluster_means.append( np.mean( self.Xtrain_scatter[self.ytrain==i,:] , axis=0 ) )
                    self.mean_lines.append( self.sax.plot((cluster_means[0][0], cluster_means[-1][0]), (cluster_means[0][1], cluster_means[-1][1]), (cluster_means[0][2], cluster_means[-1][2]), color='black', alpha = 0.5) )
                    self.projection_lines.append(self.sax.plot((0, 0), (0, 0), (0, 0), color='black', linestyle = ':'))

            if not self.run_repetitive_testing:
                self.centralized_mean_lines = []
                self.cluster_mid_points = []
                self.line_rotation_matrices = []
                for i in range(self.num_classes):
                    if i != 0:
                        self.cluster_mid_points.append((cluster_means[i] - cluster_means[0])/2)
                        pushed_to_origin_1 = -self.cluster_mid_points[-1]
                        pushed_to_origin_2 = cluster_means[i] - self.cluster_mid_points[-1] - cluster_means[0]

                        vx = pushed_to_origin_2[0] - pushed_to_origin_1[0]
                        vy = pushed_to_origin_2[1] - pushed_to_origin_1[1]
                        vz = pushed_to_origin_2[2] - pushed_to_origin_1[2] #0

                        angle_to_yz = math.asin(vx / math.sqrt( vx**2 + vy**2)) * (180 / math.pi)
                        angle_to_xy = math.asin(vz / math.sqrt( vx**2 + vy**2 + vz**2 )) * (180 / math.pi)

                        if self.cluster_mid_points[-1][1] < 0: 
                            rz = R.from_euler('z', angle_to_yz, degrees = True) 
                            rx = R.from_euler('x', 180-angle_to_xy, degrees = True)
                        else: 
                            rz = R.from_euler('z', -angle_to_yz, degrees = True)
                            rx = R.from_euler('x', angle_to_xy, degrees = True)
                        
                        r = np.matmul(rz.as_matrix(), rx.as_matrix())
                        self.line_rotation_matrices.append(r)

            self.mu = []
            for i in range(len(self.CLASSES)):
                self.mu.append(np.mean( self.Xtrain_scatter[self.ytrain==i,:] , axis=0 ))

            u = np.linspace(0, 2*np.pi, 15)
            v = np.linspace(0, np.pi, 15)
            coord1 = self.mu[0]
            x = self.onset_threshold * np.outer(np.cos(u), np.sin(v)) + coord1[0]
            y = self.onset_threshold * np.outer(np.sin(u), np.sin(v)) + coord1[1]
            z = self.onset_threshold * np.outer(np.ones(np.size(u)), np.cos(v)) + coord1[2]
            threshold_sphere = self.sax.plot_surface(x, y, z, color=self._get_class_color()[0], alpha=0.05, shade = True, linewidth = 0.05, edgecolors=[0.15,0.15,0.15])

            if self.visualization_state == "Spheres":
                u = np.linspace(0, 2*np.pi, 15)
                v = np.linspace(0, np.pi, 15)
                self.sphere = []
                for i in range(len(self.CLASSES)):
                    coord1 = np.mean( self.Xtrain_scatter[self.ytrain==i,:] , axis=0 )
                    try:
                        coord2 = self.radius[i] + coord1
                        x = np.linalg.norm(self.radius[i]) * np.outer(np.cos(u), np.sin(v)) + coord1[0]
                        y = np.linalg.norm(self.radius[i]) * np.outer(np.sin(u), np.sin(v)) + coord1[1]
                        z = np.linalg.norm(self.radius[i]) * np.outer(np.ones(np.size(u)), np.cos(v)) + coord1[2]
                    except:
                        x = np.linalg.norm(self.radius[i][:3]) * np.outer(np.cos(u), np.sin(v)) + coord1[0]
                        y = np.linalg.norm(self.radius[i][:3]) * np.outer(np.sin(u), np.sin(v)) + coord1[1]
                        z = np.linalg.norm(self.radius[i][:3]) * np.outer(np.ones(np.size(u)), np.cos(v)) + coord1[2]

                    self.sphere.append(self.sax.plot_surface(x, y, z, color=self._get_class_color()[i], alpha=0.4, shade = True, linewidth = 0.5, edgecolors=[0.15,0.15,0.15]))

                    try:
                        self.label_colors.append((self.sphere[-1]._facecolors3d[0][0], self.sphere[-1]._facecolors3d[0][1], self.sphere[-1]._facecolors3d[0][2]))
                    except:
                        self.label_colors.append((self.sphere[-1]._facecolor3d[0][0], self.sphere[-1]._facecolor3d[0][1], self.sphere[-1]._facecolor3d[0][2]))
                    #self.sax.plot((coord1[0], coord2[0]), (coord1[1], coord2[1]), (coord1[2], coord2[2]), color='black', alpha = 0.5)

            minmax = [ min([ min(self.Xtrain_scatter[:, 0]),min(self.Xtrain_scatter[:, 1]),min(self.Xtrain_scatter[:, 2]) ]),
                        max( [ max(self.Xtrain_scatter[:, 0]), max(self.Xtrain_scatter[:, 1]), max(self.Xtrain_scatter[:, 2]) ] ) ]

            self.xLim = [ -minmax[1], minmax[1] ]
            self.yLim = [ -minmax[1], minmax[1] ]
            self.zLim = [ -minmax[1], minmax[1] ]
            
            self.max_camera_distance = math.sqrt(minmax[1]**2*3)*2

            if self.run_repetitive_testing:
                self.sax.grid(False)
                self.sax.axis('off')
            else:
                self.sax.grid(True)
                self.sax.axis('on')

            self.sax.set_xlim(self.xLim)
            self.sax.set_ylim(self.yLim)
            self.sax.set_zlim(self.zLim)

            self.sax.xaxis._axinfo['tick'].update({'inward_factor': 0,
                                                'outward_factor': 0})
            self.sax.yaxis._axinfo['tick'].update({'inward_factor': 0,
                                                'outward_factor': 0})
            self.sax.zaxis._axinfo['tick'].update({'inward_factor': 0,
                                                'outward_factor': 0})

            self.sax.set_xticks(np.linspace(self.xLim[0],self.xLim[1],10))
            self.sax.set_yticks(np.linspace(self.yLim[0],self.yLim[1],10))
            self.sax.set_zticks(np.linspace(self.zLim[0],self.zLim[1],10))
            
            a = []
            for i in range(10):
                a.append('')
            self.sax.set_xticklabels(a)
            self.sax.set_yticklabels(a)
            self.sax.set_zticklabels(a)

            self.sax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            self.sax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            self.sax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

            self.sax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            self.sax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            self.sax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

            self.pause = False

        def progbar():
            self.tax.clear()
            percent = (len(self.new_data_buffer) / ( self.class_subdict_size ))
            num_eq = int( 17 * percent )
            num_spec = 17-num_eq
            progtxt = 'Data Buffer' + '[' + ( num_eq * '=' ) + ( num_spec * '   ' ) + ']   ' + str(int(100*percent)) + '%'
            # progressbar text
            self.prog_box = self.tax.text(0,0.5, progtxt, horizontalalignment='right', verticalalignment='center', transform=self.tax.transAxes, size='smaller')
            self.tax.xaxis.set_visible(False)
            self.tax.yaxis.set_visible(False)
    
        def textplot(close_class = ''):
            #bax.clear()
            self.scax.clear()
            self.text_box = []
            '''if not any(self.selected_texts):
                self.text_box = bax.text(0,0.5, 'Current patterns could be added as new movement', horizontalalignment='left', verticalalignment='center', size='medium', transform=bax.transAxes, c='forestgreen')
            elif self.selected_texts[0]:
                self.text_box = bax.text(0,0.5, self.warning_texts[0], horizontalalignment='left', verticalalignment='center', size='medium', c = 'firebrick')
            elif self.selected_texts[1]:
                self.text_box = bax.text(0,0.5, self.warning_texts[1], horizontalalignment='left', verticalalignment='center', size='medium', c = 'firebrick')
                self.text_box = bax.text(0,0.5, '                                                     ' + close_class, horizontalalignment='left', verticalalignment='center', size='medium', c = 'firebrick', fontweight = 'bold')
            elif self.selected_texts[2]:
                self.text_box = bax.text(0,0.5, self.warning_texts[2], horizontalalignment='left', verticalalignment='center', size='medium', c='forestgreen', fontweight = 'bold')'''

            self.scax.xaxis.set_visible(False)
            self.scax.yaxis.set_visible(False)
            remove_border(self.scax)

        def sepstatusplot():
            self.rax.clear()
            self.sepstatus_box = []
            vertical_offset = 0.211
            linecounter = 0
            target_class_sepscores = np.diag(self.run_classifier.SpatialClassifier.SeparabilityScore())
            # indicating the quality of classification (✓, x or -) for each class
            for key, val in enumerate(target_class_sepscores):
                if val >= 0 and val < 0.9:
                    self.sepstatus_box.append(self.rax.text(63,0.55 - key*vertical_offset, "x", horizontalalignment='center', verticalalignment='center', size='medium', transform=bax.transAxes, c='firebrick'))
                    linecounter += 1
                elif val >= 0.9 and val < 1:
                    self.sepstatus_box.append(self.rax.text(63,0.55 - key*vertical_offset, '–', horizontalalignment='center', verticalalignment='center', size='medium', transform=bax.transAxes, c='goldenrod'))
                    linecounter += 1
                elif val >= 1:
                    self.sepstatus_box.append(self.rax.text(63,0.55 - key*vertical_offset, '✓', horizontalalignment='center', verticalalignment='center', size='medium', transform=bax.transAxes, c='forestgreen'))
                    linecounter += 1
            # creating a rectangale to hide the checkboxes and their labels that are not needed at the bottom of the list using linecounter
            rect = Rectangle((0.0, -0.04 + linecounter * -0.077), 1.0, 1.0, angle=0.0, color='white', zorder=10)
            if self.visualization_mode == 'Separability' or self.run_repetitive_testing:
                # creating a rectangale to hide the checkboxes and only show their labels
                rect2 = Rectangle((0.0, 0.0), 0.1, 1.0, angle=0.0, color='white', zorder=10)
            elif self.visualization_mode == 'Repeatability':
                # creating a rectangale to hide the checkbox for REST
                rect2 = Rectangle((0.0, 0.9), 0.1, 1.0, angle=0.0, color='white', zorder=10)
            elif self.visualization_mode == 'Reviewer':
                # creating a rectangale to hide the checkbox for REST
                rect2 = Rectangle((0.0, 0.9), 0.1, 1.0, angle=0.0, color='white', zorder=10)
            self.rax.add_patch(rect)
            self.rax.add_patch(rect2)
            self.rax.xaxis.set_visible(False)
            self.rax.yaxis.set_visible(False)
            remove_border(self.rax)

        def remove_border(ax):
            for axis in [ 'top', 'bottom', 'left', 'right' ]:
                ax.spines[ axis ].set_visible( False )

        fig = plt.figure(figsize = ( 12, 9 ), tight_layout = 3)
        self.mngr = plt.get_current_fig_manager()
        geom = self.mngr.window.geometry()
        self.mngr.window.setWindowTitle( "Separability / Repeatability / Reviewer" )
        
        if self.realtime:
            self.mngr.window.setGeometry( 510, 470, 550, 550 )
            matplotlib.rcParams['font.size'] = 7
        else:
            self.mngr.window.setGeometry( 0, 0, 1024, 768 )
            matplotlib.rcParams['font.size'] = 10
        
        grid_rows = 9
        grid_columns = 12

        # create GUI layout
        gs = fig.add_gridspec( grid_rows, grid_columns )

        if self.show_interface:
            self.sax = fig.add_subplot(gs[ :, :3*grid_columns//4], projection = '3d')
        else:
            self.sax = fig.add_subplot(gs[ :, :], projection = '3d')

        projection()

        # Display text tips
        bax = plt.axes([0.25, 0.9, 0.01, 0.1])
        bax.xaxis.set_visible(False)
        bax.yaxis.set_visible(False)
        remove_border(bax)

        # Display scores
        self.scax = plt.axes([0.35, 0.85, 0.01, 0.1])
        self.scax.xaxis.set_visible(False)
        self.scax.yaxis.set_visible(False)
        remove_border(self.scax)

        # Display scores
        self.tcscax = plt.axes([0.35, 0.85, 0.01, 0.1])
        self.tcscax.xaxis.set_visible(False)
        self.tcscax.yaxis.set_visible(False)
        remove_border(self.tcscax)

        self.stax = plt.axes([0.72, 0.7, 0.2, 0.275])
        remove_border(self.stax)

        btn_x = 0.85
        btn_y = 0.9
        btn_x_offset = 0.14
        btn_y_offset = 0.025
        btn_y_offset_offset = 0.04

        def draw_class_checklist(draw = ''):
            
            if draw != 'Reviewer':
                def radio_callback( event ):
                    self.show_clusters[self.CLASSES.index(event)] = bool( (self.show_clusters[self.CLASSES.index(event)]+1)%2 )
                self.rax = plt.axes([0.72, 0.7, 0.2, 0.275])
                classes_active = []
                cl = []
                for i, val in enumerate(self.CLASSES):
                    # TODO: determine separability scores based on classifier return data
                    classes_active.append(self.show_clusters[i])
                    cl.append(val)
                for val in range(len(self.all_classes)-len(self.CLASSES)):
                    #classes_sepscores.append(-1)
                    classes_active.append(False)
                    cl.append('')
                sepstatusplot()
                self.radio = CheckButtons( self.rax, cl, actives=classes_active )

                self.rax_label =  self.all_classes[0]
                self.radio.on_clicked( radio_callback )
                remove_border(self.rax)
                try:
                    fig.delaxes(self.tax)
                except: pass
            elif draw == 'Reviewer':

                # generate radio buttons
                def radio_callback( event ):
                    label = self.radio.value_selected
                    self.rax_label = label

                self.rax.clear()
                self.rax = plt.axes([0.72, 0.7, 0.2, 0.275])
                if self.realtime:
                    self.radio = RadioButtons( self.rax, self.all_classes, activecolor = 'green')
                    #self.radio = RadioButtons( self.rax, self.CLASSES, activecolor = 'green')
                else:
                    self.radio = RadioButtons( self.rax, self.all_classes, activecolor = 'green')
                self.rax_label =  self.all_classes[0]
                self.radio.on_clicked( radio_callback )
                remove_border(self.rax)

            else:
                try:
                    fig.delaxes(self.rax)
                except: pass
        
        def draw_reviewer_buttons(draw = True):
            
            if draw:

                # add_class button
                def add_class_callback( event ):
                    if len(self.new_data_buffer) >= ( self.class_subdict_size ):
                        
                        if self.rax_label not in self.CLASSES:

                            if len(self.new_data_buffer) > self.class_subdict_size:
                                temp_class, Idx = downsample.uniform(np.vstack(self.new_data_buffer), self.class_subdict_size )
                            else:
                                temp_class = np.vstack(self.new_data_buffer)

                            class_label_idx = 0
                            for cl in self.all_classes:
                                
                                if cl in self.CLASSES:
                                    class_label_idx += 1
                                elif cl == self.rax_label:
                                    break 

                            temp_labels = class_label_idx * np.ones( ( self.class_subdict_size, ) )

                            self.Xtrain = np.concatenate( (self.Xtrain[self.ytrain < class_label_idx, :], temp_class, self.Xtrain[self.ytrain >= class_label_idx, :]), axis=0)
                            self.ytrain = np.concatenate( (self.ytrain[self.ytrain < class_label_idx], temp_labels, self.ytrain[self.ytrain >= class_label_idx] + 1 ), axis=0)



                            self.CLASSES.insert(class_label_idx, self.rax_label)

                            self.add3( Xtrain = self.Xtrain, ytrain = self.ytrain, classes = self.CLASSES )

                            self.num_classes = self.num_classes+1

                            self.show_clusters.append(True)
                            self.visualize.append(True)
                            
                            '''self.reinit_classifier()
                            
                            self.sax.clear()'''

                            self.new_data_buffer = deque()
                            self.current_bufflen = -1

                            self.new_train = True
                            
                            self.prev_pred = 100

                            self.pause = True
                        else:
                            print('already trained class, try overwriting or updating instead')

                    else:
                        print('Not enough datapoints')

                # ADD MOVEMENT BUTTON
                self.addax = plt.axes([btn_x, btn_y+btn_y_offset_offset, btn_x_offset, btn_y_offset])
                self.add_class = Button( self.addax, 'Add Movement', color = 'darkgray', hovercolor = 'lightgray' )            
                self.add_class.on_clicked( add_class_callback )

                for axis in [ 'top', 'bottom', 'left', 'right' ]:
                    self.addax.spines[ axis ].set_linewidth( 1.5 )

                def overwrite_class(percent):
                    class_index = self.CLASSES.index(self.rax_label)

                    if percent == 100:
                        if len(self.new_data_buffer) > self.class_subdict_size:
                            temp_class, Idx = downsample.uniform(np.vstack(self.new_data_buffer), self.class_subdict_size )
                        else:
                            temp_class = np.vstack(self.new_data_buffer)

                        self.Xtrain[self.ytrain == class_index,:] = temp_class
                        print(len(self.Xtrain[self.ytrain == class_index,:]), len(temp_class))

                    else:
                        temp_new_class, Idx = downsample.uniform(np.vstack(self.new_data_buffer), int(self.class_subdict_size * percent / 100) )
                        new_Xtrain = []
                        for i in range(self.num_classes):
                            class_temp = self.Xtrain[self.ytrain == i, :]
                            
                            if i == class_index:
                                temp_class, Idx = downsample.uniform(np.vstack(class_temp), self.class_subdict_size - int(self.class_subdict_size * percent / 100) )
                                temp_class = np.concatenate( (temp_class, temp_new_class), axis=0 )
                                new_Xtrain.append( temp_class )
                            else:
                                new_Xtrain.append( class_temp )
                    
                        self.Xtrain = np.vstack(new_Xtrain)
                    
                    return self.Xtrain

                # ow_class button
                def overwrite_class_callback( event ):
                    if len(self.new_data_buffer) >= ( self.class_subdict_size ):
                        
                        if self.rax_label in self.CLASSES:
                            self.Xtrain = overwrite_class(100)

                            self.add3( Xtrain = self.Xtrain, ytrain = self.ytrain, classes = self.CLASSES )
                            
                            '''self.reinit_classifier()
                            
                            self.sax.clear()

                            projection()'''

                            self.new_data_buffer = deque()
                            self.current_bufflen = -1

                            self.new_train = True
                            
                            self.prev_pred = 100

                            self.pause = True
                        else:
                            print('untrained class, try adding instead')

                    else:
                        print('Not enough datapoints')

                # OVERWRITE MOVEMENT BUTTON
                self.owax = plt.axes([btn_x, btn_y, btn_x_offset, btn_y_offset])
                self.ow_class = Button( self.owax, 'Overwrite Movement' )
                    
                self.ow_class.on_clicked( overwrite_class_callback )
                self.ow_class.color = 'darkgray'
                self.ow_class.hovercolor = 'lightgray'

                for axis in [ 'top', 'bottom', 'left', 'right' ]:
                    self.owax.spines[ axis ].set_linewidth( 1.5 )

                # reset buffer button
                def reset_buffer_callback( event ):
                    self.currentlyGatheringNewData = True
                    self.new_data_buffer = deque()
                    self.current_bufflen = -1
                    self.selected_texts = [False, False, False]
                    self.scat[-1]._offsets3d = (np.array([0]),np.array([0]),np.array([0]))

                self.rbax = plt.axes([btn_x, btn_y-btn_y_offset_offset*2, btn_x_offset, btn_y_offset])
                self.reset_buffer = Button( self.rbax, 'Gather New Data' )
                self.reset_buffer.on_clicked( reset_buffer_callback )
                self.reset_buffer.color = 'goldenrod'
                self.reset_buffer.hovercolor = 'gold'

                for axis in [ 'top', 'bottom', 'left', 'right' ]:
                    self.rbax.spines[ axis ].set_linewidth( 1.5 )

                # delete movement button
                def delete_movement_callback( event ):
                    class_index = self.CLASSES.index(self.rax_label)

                    self.Xtrain = np.delete( self.Xtrain, np.where(self.ytrain == class_index), 0 )
                    self.ytrain = np.delete( self.ytrain, np.where(self.ytrain == self.num_classes - 1), 0 )
                    self.num_classes -= 1

                    del self.show_clusters[self.CLASSES.index(self.rax_label)]
                    del self.visualize[self.CLASSES.index(self.rax_label)]
                    self.CLASSES.remove( self.rax_label )
            
                    self.add3( Xtrain = self.Xtrain, ytrain = self.ytrain, classes = self.CLASSES )

                    '''self.reinit_classifier()

                    self.sax.clear()
                    
                    projection()    '''        
                    
                    self.new_data_buffer = deque()
                    self.current_bufflen = -1

                    self.new_train = True

                    self.prev_pred = 100

                    self.pause = True
                    
                #delete movement
                self.dax = plt.axes([btn_x, btn_y-btn_y_offset_offset*3, btn_x_offset, btn_y_offset])
                self.delete_movement_class = Button( self.dax, 'Delete Movement' )
                self.delete_movement_class.on_clicked( delete_movement_callback )
                self.delete_movement_class.color = 'goldenrod'
                self.delete_movement_class.hovercolor = 'gold'

                for axis in [ 'top', 'bottom', 'left', 'right' ]:
                    self.dax.spines[ axis ].set_linewidth( 1.5 )

            else:
                try:
                    fig.delaxes(self.addax)
                    fig.delaxes(self.owax)
                    fig.delaxes(self.upax)
                    fig.delaxes(self.rbax)
                    fig.delaxes(self.dax)
                    self.tax.clear()
                except: pass

        def draw_progress_bar(draw = True):
            
            if draw:
                # progressbar text
                self.tax = plt.axes([0.99, 0.66, 0.25, 0.025])
                remove_border(self.tax)
                progbar()
            else:
                try:
                    fig.delaxes(self.tax)
                except: pass


        self.visualization_options = ['Run ghosting', 'Cluster fade', 'Crosshair perspective']
        self.visualization_options_dict = {}
        for i in self.visualization_options:
            self.visualization_options_dict[i] = True

        def draw_visualization_checklist(draw = True):
            # generate visualization option checklist buttons
            if draw:
                def visualization_checklist_callback( event ):
                    self.visualization_options_dict[event] = bool( (self.visualization_options_dict[event]+1)%2 )
                    try:
                        if event == 'Run ghosting':
                            self.ghost_alpha = 0 if not self.visualization_options_dict['Run ghosting'] else 1
                            for key, val in enumerate(self.ghost):
                                self.ghost[key].set_alpha(self.ghost_alpha)
                    except: pass

                self.visax = plt.axes([0.72, 0.25, 0.15, 0.15])
                self.visualization_checklist = CheckButtons( self.visax, self.visualization_options, actives=[True for i in self.visualization_options] )

                self.visualization_checklist.on_clicked( visualization_checklist_callback )
                remove_border(self.visax)
            else:
                try:
                    fig.delaxes(self.visax)
                except: pass

        # Separability button
        def sep_button_press():
            if self.visualization_mode != 'Separability':
                self.sep_button.color = 'goldenrod'
                self.sep_button.hovercolor = 'yellow'
                self.rev_button.color = 'forestgreen'
                self.rev_button.hovercolor = 'lightseagreen'

            self.visualization_mode = 'Separability'
            self.visualization_state = 'Clusters'
            self.run_repetitive_testing = False
            projection()
            draw_visualization_checklist(False)
            draw_class_checklist()
            draw_reviewer_buttons(False)
            draw_progress_bar(False)
            self.rotation = True
        
        def draw_seprep_interface(draw = True):

            if draw:
                # Separability button
                def separability_button_callback( event ):
                    sep_button_press()

                self.sepbax = plt.axes([0.7, 0.59, 0.12, 0.02])
                self.sep_button = Button( self.sepbax, 'Separability', color = 'goldenrod', hovercolor = 'yellow' )
                self.sep_button.on_clicked( separability_button_callback )

                for axis in [ 'top', 'bottom', 'left', 'right' ]:
                    self.sepbax.spines[ axis ].set_linewidth( 1.5 )

                # Reviewer button
                def reviewer_button_callback( event = 0 ):
                    if self.visualization_mode != 'Reviewer':
                        self.rev_button.color = 'goldenrod'
                        self.rev_button.hovercolor = 'yellow'
                        self.sep_button.color = 'forestgreen'
                        self.sep_button.hovercolor = 'lightseagreen'

                    self.visualization_mode = 'Reviewer'
                    self.visualization_state = 'Clusters'
                    self.run_repetitive_testing = False
                    projection()
                    draw_visualization_checklist(False)
                    draw_class_checklist('Reviewer')
                    draw_reviewer_buttons()
                    draw_progress_bar()

                self.revbax = plt.axes([0.7, 0.62, 0.12, 0.02])
                self.rev_button = Button( self.revbax, 'Reviewer' )
                self.rev_button.on_clicked( reviewer_button_callback )
                self.rev_button.color = 'forestgreen'
                self.rev_button.hovercolor = 'lightseagreen'

                for axis in [ 'top', 'bottom', 'left', 'right' ]:
                    self.revbax.spines[ axis ].set_linewidth( 1.5 )

                # exit button
                def exit_button_callback( event ):
                    self.close()

                self.ebax = plt.axes([0.87, 0.1, 0.12, 0.05])
                self.exit_button = Button( self.ebax, 'Exit' )
                self.exit_button.on_clicked( exit_button_callback )
                self.exit_button.color = 'firebrick'
                self.exit_button.hovercolor = 'indianred'

                for axis in [ 'top', 'bottom', 'left', 'right' ]:
                    self.ebax.spines[ axis ].set_linewidth( 1.5 )

                if self.realtime:
                    self.ebax.set_visible(False)

                # save button
                def save_button_callback( event ):
                    print( 'Saving data...', end = '', flush = True )
                    calibrate = np.zeros( ( 1, ), dtype = np.object )
                    calibrate[ 0 ] = {}
                    for key,i in enumerate(self.CLASSES):
                        calibrate[ 0 ].update( { i : self.Xtrain[self.ytrain==key,:] } )

                    savemat( 'calibrate.mat', mdict = { 'calibrate' : calibrate } )

                    PARAM_FILE = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), 
                        'data', 'Subject_%s' % SUBJECT, 'RESCU_Protocol_Subject_%s.pkl' % SUBJECT )

                    try:
                        with open( PARAM_FILE, 'rb' ) as pkl:
                            Parameters = pickle.load( pkl )

                        Parameters['Calibration']['CUE_LIST'] = ['REST']
                        for i in self.all_classes:
                            if i in self.CLASSES and i != 'REST':
                                Parameters['Calibration'][i.upper()] = True
                                Parameters['Calibration']['CUE_LIST'].append(i.upper())
                            elif i not in self.CLASSES and i != 'REST':
                                Parameters['Calibration'][i.upper()] = False

                        self.CLASSES = Parameters['Calibration']['CUE_LIST']

                        with open( PARAM_FILE, 'wb' ) as f:
                            pickle.dump(Parameters, f)

                    except:
                        print("Error reading parameters")


                    print( 'Done!' )

                self.sbax = plt.axes([0.74, 0.1, 0.12, 0.05])
                self.save_button = Button( self.sbax, 'Save' )
                self.save_button.on_clicked( save_button_callback )
                self.save_button.color = 'cornflowerblue'
                self.save_button.hovercolor = 'royalblue'

                for axis in [ 'top', 'bottom', 'left', 'right' ]:
                    self.sbax.spines[ axis ].set_linewidth( 1.5 )

                if self.realtime:
                    self.sbax.set_visible(False)
            
            else:
                try:
                    fig.delaxes(self.sepbax)
                except: pass      
                try:
                    fig.delaxes(self.repbax)
                except: pass      
                try:
                    fig.delaxes(self.revbax)
                except: pass      
                try:
                    fig.delaxes(self.ebax)
                except: pass      
                try:
                    fig.delaxes(self.sbax)
                except: pass  
                try:
                    fig.delaxes(self.krax)
                except: pass 
                try:
                    fig.delaxes(self.krax2)
                except: pass      

        if self.show_interface:
            draw_seprep_interface()
            draw_class_checklist('')
        else:
            draw_seprep_interface(False)
            draw_class_checklist('')
            try:
                fig.delaxes(self.rax)
            except: pass

        def clear_trail_button_callback( event ):
            self.trailX = []
            self.trailY = []
            self.trailZ = []
            self.trailX.append(self.scat[-2]._offsets3d[0])
            self.trailY.append(self.scat[-2]._offsets3d[1])
            self.trailZ.append(self.scat[-2]._offsets3d[2])

            self.linetrail[0]._verts3d = (self.scat[-2]._offsets3d[0], self.scat[-2]._offsets3d[1], self.scat[-2]._offsets3d[2])
        
        # switch interface button
        def switch_interface_button_callback( event ):
            self.sax.clear()
            try:
                fig.delaxes(self.sax)
            except: pass
            
            if self.show_interface:
                
                self.show_interface = False

                self.mngr.window.setGeometry( 510, 470, 550, 550 )
                matplotlib.rcParams['font.size'] = 7
                
                self.sax = fig.add_subplot(gs[ :, :], projection = '3d')

                self.visualize_button.label.set_text( 'Switch to Trainer' )
                
                if self.visualization_mode != 'Separability':
                    self.sep_button.color = 'goldenrod'
                    self.sep_button.hovercolor = 'yellow'
                    self.rev_button.color = 'forestgreen'
                    self.rev_button.hovercolor = 'lightseagreen'
                
                self.visualization_mode = 'Separability'
                self.visualization_state = 'Clusters'
                self.run_repetitive_testing = False
                projection()
                draw_visualization_checklist(False)
                draw_class_checklist()
                draw_reviewer_buttons(False)
                draw_progress_bar(False)
                self.rotation = True
                draw_seprep_interface(False)
                draw_class_checklist('')
                try:
                    fig.delaxes(self.rax)
                except: pass
                draw_visualization_buttons()
            else:
                self.show_interface = True

                self.mngr.window.setGeometry( 0, 0, 1024, 768 )
                matplotlib.rcParams['font.size'] = 10
                
                self.sax = fig.add_subplot(gs[ :, :3*grid_columns//4], projection = '3d')

                self.visualize_button.label.set_text( 'Switch to Overview' )
                
                projection()
                
                draw_seprep_interface()
                draw_class_checklist('')
                draw_visualization_buttons()
            
            self.sax.grid(True)
            self.sax.axis('on')

            self.sax.set_xlim(self.xLim)
            self.sax.set_ylim(self.yLim)
            self.sax.set_zlim(self.zLim)

            self.sax.xaxis._axinfo['tick'].update({'inward_factor': 0,
                                                'outward_factor': 0})
            self.sax.yaxis._axinfo['tick'].update({'inward_factor': 0,
                                                'outward_factor': 0})
            self.sax.zaxis._axinfo['tick'].update({'inward_factor': 0,
                                                'outward_factor': 0})

            self.sax.set_xticks(np.linspace(self.xLim[0],self.xLim[1],10))
            self.sax.set_yticks(np.linspace(self.yLim[0],self.yLim[1],10))
            self.sax.set_zticks(np.linspace(self.zLim[0],self.zLim[1],10))
            
            a = []
            for i in range(10):
                a.append('')
            self.sax.set_xticklabels(a)
            self.sax.set_yticklabels(a)
            self.sax.set_zticklabels(a)

            self.sax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            self.sax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            self.sax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

            self.sax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            self.sax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            self.sax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

        def draw_visualization_buttons(draw = True):
            
            try:
                fig.delaxes(self.swifax)
            except: pass
            try:
                fig.delaxes(self.vlax)
            except: pass
            try:
                fig.delaxes(self.vax)
            except: pass
            try:
                fig.delaxes(self.rotax)
            except: pass
            try:
                fig.delaxes(self.tratax)
            except: pass

            try:
                fig.delaxes(self.krax)
            except: pass

            try:
                fig.delaxes(self.krax2)
            except: pass

            try:
                fig.delaxes(self.kraxcont)
            except: pass

            try:
                fig.delaxes(self.kraxcont2)
            except: pass

            try:
                fig.delaxes(self.kraxcont3)
            except: pass

            try:
                fig.delaxes(self.kraxtrail)
            except: pass

            try:
                fig.delaxes(self.scax)
            except: pass

            try:
                fig.delaxes(self.tcscax)
            except: pass

            try:
                fig.delaxes(self.cltrax)
            except: pass

            # Display scores
            self.scax = plt.axes([0.35, 0.85, 0.01, 0.1])
            self.scax.xaxis.set_visible(False)
            self.scax.yaxis.set_visible(False)
            remove_border(self.scax)

            # Display scores
            self.tcscax = plt.axes([0.35, 0.85, 0.01, 0.1])
            self.tcscax.xaxis.set_visible(False)
            self.tcscax.yaxis.set_visible(False)
            remove_border(self.tcscax)

            self.krax = plt.axes([0.15, 0.4, 0.5, 0.1])
            self.krax.set_visible(False)
            self.krax2 = plt.axes([0.15, 0.15, 0.5, 0.1])
            self.krax2.set_visible(False)

            self.kraxcont = plt.axes([0.28, 0.41, 0.1, 0.025])

            def continue_button_callback( event ):
                self.add2(1)

            self.continue_button = Button( self.kraxcont, 'Continue' , color = 'mediumorchid', hovercolor = 'magenta')
            self.continue_button.on_clicked( continue_button_callback )

            for axis in [ 'top', 'bottom', 'left', 'right' ]:
                self.kraxcont.spines[ axis ].set_linewidth( 1.5 )     
            
            self.kraxcont.set_visible(False)

            self.kraxcont2 = plt.axes([0.345, 0.41, 0.1, 0.025])

            self.continue_button2 = Button( self.kraxcont2, 'Start', color = 'mediumorchid', hovercolor = 'magenta' )
            self.continue_button2.on_clicked( continue_button_callback )

            for axis in [ 'top', 'bottom', 'left', 'right' ]:
                self.kraxcont2.spines[ axis ].set_linewidth( 1.5 )     
            
            self.kraxcont2.set_visible(False)

            self.kraxcont3 = plt.axes([0.345, 0.16, 0.1, 0.025])

            self.continue_button3 = Button( self.kraxcont3, 'Continue', color = 'mediumorchid', hovercolor = 'magenta' )
            self.continue_button3.on_clicked( continue_button_callback )

            for axis in [ 'top', 'bottom', 'left', 'right' ]:
                self.kraxcont3.spines[ axis ].set_linewidth( 1.5 )     
            
            self.kraxcont3.set_visible(False)
            
            self.kraxtrail = plt.axes([0.41, 0.41, 0.1, 0.025])

            def view_trail_button_callback( event ):
                self.add2(2)

            self.view_trail_button = Button( self.kraxtrail, 'Review trail', color = 'mediumorchid', hovercolor = 'magenta')
            self.view_trail_button.on_clicked( view_trail_button_callback )

            for axis in [ 'top', 'bottom', 'left', 'right' ]:
                self.kraxtrail.spines[ axis ].set_linewidth( 1.5 )     
            
            self.kraxtrail.set_visible(False)


            self.swifax = plt.axes([0.20, 0.02, btn_x_offset, btn_y_offset])
            if self.show_interface:
                self.switch_interface_button = Button( self.swifax, 'Switch to Overview', color = 'mediumorchid', hovercolor = 'magenta' )
            else:
                self.switch_interface_button = Button( self.swifax, 'Switch to Trainer', color = 'mediumorchid', hovercolor = 'magenta' )
            self.switch_interface_button.on_clicked( switch_interface_button_callback )

            for axis in [ 'top', 'bottom', 'left', 'right' ]:
                self.swifax.spines[ axis ].set_linewidth( 1.5 )      
            
            self.vlax = plt.axes([0.04, 0.1, 0.1, 0.01])
            self.vlax.text(0,0.5, 'Visualization options', horizontalalignment='left', verticalalignment='center', size='medium', transform=self.vlax.transAxes, c='black', fontweight = 'bold')
            self.vlax.xaxis.set_visible(False)
            self.vlax.yaxis.set_visible(False)
            remove_border(self.vlax)

            self.cltrax = plt.axes([0.36, 0.02, btn_x_offset, btn_y_offset])
            self.clear_trail_button = Button( self.cltrax, 'Clear Trail', color = 'mediumorchid', hovercolor = 'magenta' )
            self.clear_trail_button.on_clicked( clear_trail_button_callback )

            for axis in [ 'top', 'bottom', 'left', 'right' ]:
                self.cltrax.spines[ axis ].set_linewidth( 1.5 )      

            # visualization mode button
            def visualize_button_callback( event ):
                if self.visualization_state == 'Spheres':
                    self.visualization_state = 'Clusters'
                    self.visualize_button.label.set_text( 'Show Spheres' )
                else:
                    self.visualization_state = 'Spheres'
                    self.visualize_button.label.set_text( 'Show Clusters' )
                projection()

            self.vax = plt.axes([0.04, 0.06, btn_x_offset, btn_y_offset])
            if self.visualization_state == 'Clusters':
                self.visualize_button = Button( self.vax, 'Show Spheres', color = 'mediumorchid', hovercolor = 'magenta' )
            else:
                self.visualize_button = Button( self.vax, 'Show Clusters', color = 'mediumorchid', hovercolor = 'magenta' )

            self.visualize_button.on_clicked( visualize_button_callback )

            for axis in [ 'top', 'bottom', 'left', 'right' ]:
                self.vax.spines[ axis ].set_linewidth( 1.5 )

            # rotation button
            self.rotation = True
            def rotation_button_callback( event ):
                if self.rotation:
                    self.rotation = False
                    self.rotation_button.label.set_text('Start Rotation')
                else:
                    self.rotation = True
                    self.rotation_button.label.set_text('Stop Rotation')

            self.rotax = plt.axes([0.04, 0.02, btn_x_offset, btn_y_offset])
            self.rotation_button = Button( self.rotax, 'Stop Rotation', color = 'mediumorchid', hovercolor = 'magenta' )
            self.rotation_button.on_clicked( rotation_button_callback )

            for axis in [ 'top', 'bottom', 'left', 'right' ]:
                self.rotax.spines[ axis ].set_linewidth( 1.5 )      

            # trailing button
            def trailing_button_callback( event ):
                if self.trail_mode == 0:
                    self.trail_mode = 1
                    self.trailing_button.label.set_text('Show Full Trail')
                elif self.trail_mode == 1:
                    self.trail_mode = 2
                    self.trailing_button.label.set_text('Stop Trailing')
                elif self.trail_mode == 2:
                    self.trail_mode = 0
                    self.trailing_button.label.set_text('Start Trailing')

            self.tratax = plt.axes([0.20, 0.06, btn_x_offset, btn_y_offset])
            self.trailing_button = Button( self.tratax, 'Start Trailing', color = 'mediumorchid', hovercolor = 'magenta' )
            self.trailing_button.on_clicked( trailing_button_callback )

            for axis in [ 'top', 'bottom', 'left', 'right' ]:
                self.tratax.spines[ axis ].set_linewidth( 1.5 )
        
        draw_visualization_buttons()

        def set_alpha_values(alpha_val, i, lines = True):

            if lines: 
                self.mean_lines[i-1][0].set_alpha(alpha_val)
            else:
                self.mean_lines[i-1][0].set_alpha(0)

            if self.visualization_state == "Clusters":
                self.scat[i].set_alpha(alpha_val)
            elif self.visualization_state == "Spheres":
                self.sphere[i].set_alpha(alpha_val)
                self.sphere[i].set_linewidth(alpha_val)

            #self.labels[i].set_alpha(alpha_val)
            self.labels[i].set_alpha(0.85)
            self.labels[i].set_color( ( grayscale - (grayscale - self.label_colors[i][0]) * alpha_val,
                                        grayscale - (grayscale - self.label_colors[i][1]) * alpha_val ,
                                        grayscale - (grayscale - self.label_colors[i][2]) * alpha_val ) ) 

        self.current_bufflen = -1

        angle = 0

        self.prev_pred = 100
        local_minmax = MinMaxScaler()        
        frame_counter = 0
        segment_data = []

        self.t = time.time()
        local_score_display_rate = 2
        local_score_display_counter = 0
        rate = 0

        while not self._exit_event.is_set():
            #t3 = time.time()
            data = None
            # pull samples from queue
            while self._queue.qsize() > 0:
                try:                    
                    data = self._queue.get( timeout = 1e-3 )
                except: # self._queue.Empty:
                    pass

            if self._queue_segment.qsize() > 0:
                try:
                    
                    segment_data = self._queue_segment.get( timeout = 1e-3 )

                except: # self._queue.Empty:
                    pass
            
            if self._queue_classifier.qsize() > 0:
                try:
                    
                    classifier = self._queue_classifier.get( timeout = 1e-3 )
                    self.run_classifier = classifier

                    self.stdscaler = self.run_classifier.SpatialClassifier.stdscaler
                    self.proj = self.run_classifier.SpatialClassifier.proj
                    self.class_pairs = self.run_classifier.SpatialClassifier.class_pairs
                    self.radius = self.run_classifier.SpatialClassifier.radius
                    self.proj = self.run_classifier.SpatialClassifier.proj
                    self.onset_threshold = self.threshold_ratio

                    projection()

                    continue

                except: # self._queue.Empty:
                    pass
                
            if self.new_train:
                self.new_train = False

            if data is not None:

                if self.pause:
                    continue

                local_score_display_counter += 1
                
                if self.realtime == True:
                    if len(data) > 2:
                        self.onset_threshold = data[2][0]
                        projection()
                        continue
                    data_mean = data[0].reshape(1, -1)
                    
                    external_pred = [data[1]]                   
                else:
                    data_mean = data.reshape(1, -1)

                if frame_counter < 0:
                    frame_counter += 1
                else:
                    frame_counter = 0
                    if self.num_classes > 3:                            
                        data_scaled = self.run_classifier.SpatialClassifier.stdscaler.transform(data_mean)#data.reshape(1, -1))
                        #data_scaled = self.stdscaler.transform(data_mean)#data.reshape(1, -1))
                        self.data_scatter = self.proj.transform(data_scaled)
                    else:
                        self.data_scatter = self.ica_scatter.transform(data_mean[:,:8])

                    if self.run_repetitive_testing:
                        self.data_scatter = self.data_scatter[:,:3]
                        self.data_scatter -= self.cluster_mid_points[self.current_class-1][:3]
                        self.data_scatter = np.matmul(self.data_scatter, self.line_rotation_matrices[self.current_class-1])
                        
                    self.scat[-2]._offsets3d = (self.data_scatter[:,0],self.data_scatter[:,1],self.data_scatter[:,2])

                    if self.visualization_options_dict['Crosshair perspective']:

                        rad_elev = math.radians(self.sax.elev)
                        rad_azim = math.radians(self.sax.azim)

                        x = self.max_camera_distance * np.cos(rad_azim) * np.cos(rad_elev)
                        y = self.max_camera_distance * np.sin(rad_azim) * np.cos(rad_elev)
                        z = self.max_camera_distance * np.sin(rad_elev)

                        #self.campozplot._offsets3d = ([x],[y],[z])

                        dist_from_camera = np.linalg.norm(self.data_scatter[:,:3] - [x,y,z])

                        ns_ratio = 1-(dist_from_camera/(1.5*self.max_camera_distance))

                        ns_ratio = 0 if ns_ratio < 0 else ns_ratio 

                        #print("RATIO: ", ns_ratio, "X: ", x, "Y: ", y, "Z: ", z, "ELEV: ", self.sax.elev, "AZIM: ", self.sax.azim)

                        #X_size = [ns_ratio*175+100][0]
                        X_size = [ns_ratio*500][0]
                        
                        #print("RATIO: ", ns_ratio, "X_size: ", X_size)
                        if X_size < 75:
                            self.scat[-2]._sizes3d = ([75])
                        else:
                            self.scat[-2]._sizes3d = ([X_size])
                    else:
                        self.scat[-2]._sizes3d = ([200])

                #TODO add new scat dataset for segments
                if len(segment_data) > 0:
                    if self.num_classes > 3:                            
                        segment_data_scaled = self.stdscaler.transform(segment_data)#data.reshape(1, -1))
                        segment_data_scatter = self.proj.transform(segment_data_scaled)
                    else:
                        segment_data_scatter = self.ica_scatter.transform(segment_data)
                    segment_data_scatter = np.vstack(segment_data_scatter)
                    self.scat[-1]._offsets3d = (segment_data_scatter[:,0],segment_data_scatter[:,1],segment_data_scatter[:,2])
                    self.scat[-1].set_alpha(0.55)

                    segment_data = []

                if self.rotation and not self.run_repetitive_testing:
                    self.sax.view_init(30, angle)
                    angle += 0.25

                r = np.zeros((2,len(self.CLASSES)))                  

                for key, i in enumerate(self.radius):
                    try:
                        r[0,key] = np.linalg.norm(self.mu[key] - self.data_scatter)
                        if r[0,key] < np.linalg.norm(i):
                            r[1,key] = 1
                    except:
                        print('error')

                r[0,:] = np.ones(len(self.CLASSES)) - local_minmax.fit_transform(r[0,:].reshape(-1, 1))[:,0]
                #predictions
                #print("elev: ", self.sax.elev)

                try:
                    onset_occured = np.linalg.norm(self.data_scatter-self.mu[0]) > self.onset_threshold #np.mean(data[:8]) > onset_threshold:
                except:
                    onset_occured = np.linalg.norm(self.data_scatter-self.mu[0][:3]) > self.onset_threshold

                if onset_occured:
                    pred, _ = self.run_classifier.emg_classify(data_mean, self.onset_threshold, self.classifier, True, True)

                else:
                    pred, _ = self.run_classifier.emg_classify(data_mean, self.onset_threshold, self.classifier, True, True)
                    pred = [0]

                try:
                    axes_projection = self.run_classifier.SpatialClassifier.projection
                    axes_distance = self.run_classifier.SpatialClassifier.axes_distance

                    if self.run_repetitive_testing:
                        for i, val in enumerate(axes_projection):
                            if len(axes_projection) != 0:
                                axes_projection[i] = val[:3]
                                axes_projection[i] -= self.cluster_mid_points[self.current_class-1][:3]
                                axes_projection[i] = np.matmul(val[:3], self.line_rotation_matrices[self.current_class-1])

                    if len(axes_projection) == 0:
                        for lines in self.projection_lines:
                            lines[0].set_alpha(0)
                        self.scat[-2]._facecolor3d = ( 1, 1, 1, 1 )

                except Exception as e: pass

                if self.realtime == True:
                    pred = external_pred    

                if type(pred) == int:
                    pred = [pred]

                grayscale = 0.75
                current_max_scalar = 0

                for i in range(self.num_classes):

                    alpha_val = 0

                    current_visualize = self.show_clusters[i]
                    
                    if self.visualization_state == "Clusters":
                        
                        if self.visualization_options_dict['Cluster fade']:
                            alpha_val = 0.1 + (r[0,i]/1.5)                   
                        else:
                            alpha_val = 0.1 + (r[0,i]/1.5)

                    elif self.visualization_state == "Spheres":
                        alpha_val = 0.1 + (r[0,i]/3)
                        if self.visualization_options_dict['Cluster fade']:
                            alpha_val = (r[0,i])**2 - 0.1
                            if alpha_val > 0.55:
                                alpha_val = 0.55              
                        else:
                            alpha_val = 0.1 + (r[0,i]/3)
                    
                    if alpha_val > 1:
                        alpha_val = 1
                    if alpha_val < 0:
                        alpha_val = 0

                    labrot = R.from_euler('zy', [-self.sax.azim, self.sax.elev], degrees = True)
                    
                    try:
                        self.labels[i]._position3d = np.matmul(self.labels_starting_points[i]-self.mu[i], labrot.as_matrix() )+self.mu[i]
                    except:
                        self.labels[i]._position3d = np.matmul(self.labels_starting_points[i]-self.mu[i][:3], labrot.as_matrix() )+self.mu[i][:3]

                    if self.visualization_mode == 'Separability':

                        set_alpha_values(alpha_val,i, False)
                    
                    elif self.visualization_mode == 'Reviewer':

                        set_alpha_values(alpha_val,i, False)

                    if self.prev_pred != pred:
                        if len(pred) > 1:
                            if i in pred:
                                if self.visualization_state == "Clusters":
                                    self.scat[i]._linewidth = [20]
                                    self.scat[i]._edgecolor3d = (0, 0, 0, 1)
                                self.labels[i].set_path_effects([PathEffects.withStroke(linewidth=1.5, foreground='k')])
                            else:
                                if self.visualization_state == "Clusters":
                                    self.scat[i]._edgecolor3d = (1, 1, 1, 1)
                                self.labels[i].set_path_effects([PathEffects.withStroke(linewidth=0, foreground='k')])

                        else:
                            if pred[0] == i:
                                if self.visualization_state == "Clusters":
                                    self.scat[i]._linewidth = [20]
                                    self.scat[i]._edgecolor3d = (0, 0, 0, 1)
                                self.labels[i].set_path_effects([PathEffects.withStroke(linewidth=1.5, foreground='k')])
                            else:
                                if self.visualization_state == "Clusters":
                                    self.scat[i]._edgecolor3d = (1, 1, 1, 1)
                                self.labels[i].set_path_effects([PathEffects.withStroke(linewidth=0, foreground='k')])

                try:
                    
                    if not r[1,:].any() and onset_occured:
                        if self.currentlyGatheringNewData:
                            try:
                                self.new_data_buffer.append(data[0])
                            except:
                                self.new_data_buffer.append(data)
                            
                            self.scat[-1].set_alpha(0.55)
                            
                            class_buffer_empty = self.scat[-1]._offsets3d[0] == (np.array([0]), np.array([0]), np.array([0]))

                            if class_buffer_empty.all():
                                self.scat[-1]._offsets3d = (self.data_scatter[:,0],self.data_scatter[:,1],self.data_scatter[:,2])
                            else:
                                self.scat[-1]._offsets3d = np.concatenate((self.scat[-1]._offsets3d, (self.data_scatter[:,0],self.data_scatter[:,1],self.data_scatter[:,2])), axis=1)

                        try:
                            if len(self.new_data_buffer) <= ( self.class_subdict_size ) and self.current_bufflen != len(self.new_data_buffer):
                                progbar()
                        except:
                            pass
                except: pass

                try:
                    if len(self.new_data_buffer) > ( self.class_subdict_size ):
                        try:
                            if self.add_class.color is not 'forestgreen':
                                self.add_class.color = 'forestgreen'
                                self.add_class.hovercolor = 'mediumseagreen'
                        except:
                            pass
                        if self.ow_class.color is not 'forestgreen':
                            self.ow_class.color = 'forestgreen'
                            self.ow_class.hovercolor = 'mediumseagreen'
                        self.selected_texts = [False, False, False]
                        self.currentlyGatheringNewData = False
                        self.new_data_buffer.popleft()
                        
                    elif len(self.new_data_buffer) > ( self.class_subdict_size // 2 ):
                        if self.update_class.color is not 'lightgreen':
                            self.update_class.color = 'lightgreen'
                            self.update_class.hovercolor = 'mediumseagreen'
                    try:
                        if len(self.new_data_buffer) < ( self.class_subdict_size ) and self.add_class.color is not 'darkgray':
                            self.add_class.color = 'darkgray'
                            self.add_class.hovercolor = 'lightgray'
                    except:
                        pass
                    if len(self.new_data_buffer) < ( self.class_subdict_size ) and self.ow_class.color is not 'darkgray':
                        self.ow_class.color = 'darkgray'
                        self.ow_class.hovercolor = 'lightgray'
                    if len(self.new_data_buffer) <= ( self.class_subdict_size // 2 ) and self.update_class.color is not 'darkgray':
                        self.update_class.color = 'darkgray'
                        self.update_class.hovercolor = 'lightgray'

                    self.current_bufflen = len(self.new_data_buffer)
                except:
                    pass

                draw_trail(self.scat[-2])
                self.prev_pred = pred

            #t4 = time.time()-t3
            plt.pause( 0.001 )
            #print("before draw", t4, "after draw", time.time()-t3)

        plt.close(fig )
        self.close()

    def add( self, data ):
        """
        Add data to be plotted

        Parameters
        ----------
        data : numpy.ndarray (n_features,)
            The data sample to be added to the plotting queue
        """
        try:
            self._queue.put( data, timeout = 1e-3 )
        except Exception as e:
            Warning(e)
            pass

    def add2( self, data ):
        """
        Add data to detect button press

        Parameters
        ----------
        data : int
            The data sample to be added to the plotting queue
        """
        try:
            self._queue2.put( data, timeout = 1e-3 )
        except Exception as e:
            Warning(e)
            pass

    def add3( self, Xtrain, ytrain, classes ):
        """
        Add data to detect button press

        Parameters
        ----------
        data : int
            The data sample to be added to the plotting queue
        """
        try:
            self._queue3.put( ( Xtrain, ytrain, classes), timeout = 1e-3 )
        except Exception as e:
            Warning(e)
            pass

    def add_segment( self, data ):
        """
        Add data to be plotted

        Parameters
        ----------
        data : numpy.ndarray (n_features,)
            The data sample to be added to the plotting queue
        """
        try:
            self._queue_segment.put( data, timeout = 1e-3 )
        except Exception as e:
            Warning(e)
            pass

    def add_classifier( self, data ):
        """
        Add data to be plotted

        Parameters
        ----------
        data : numpy.ndarray (n_features,)
            The data sample to be added to the plotting queue
        """
        try:
            self._queue_classifier.put( data, timeout = 1e-3 )
        except Exception as e:
            Warning(e)
            pass

    def close( self ):
        """
        Stop the plotting subprocess while releasing subprocess resources
        """
        self._exit_event.set()
        while self._queue.qsize() > 0:
            try:
                self._queue.get( timeout = 1e-3 )
            except Exception as e:
                Warning(e)
                pass
        #self._plotter.join()

        self._plotter.terminate

    @property
    def state3( self ):
        """
        Returns
        -------
        tuple [numpy.ndarray (n_samples, n_features), list (classes)]
            A newly labeled segment of features
        """
        if self._queue3.qsize() > 0:
            try:
                return self._queue3.get( timeout = 1e-3 )
            except self._queue3.Empty:
                return None

if __name__ == '__main__':

    from TrainingDataGeneration import TrainingDataGenerator
    from scipy.io import loadmat

    SUBJECT = 1
    SMG = False
    offline_test = True
    
    if len(sys.argv) > 1:
        SUBJECT = sys.argv[1]
        if len(sys.argv) > 2:
            RUN_FL = int(sys.argv[2])

    PARAM_FILE = os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ), 
                 'data', 'Subject_%s' % SUBJECT, 'RESCU_Protocol_Subject_%s.pkl' % SUBJECT )

    with open( PARAM_FILE, 'rb' ) as pkl:
        Parameters = pickle.load( pkl )

    EMG_SCALING_FACTOR = 10000.0

    tdg = TrainingDataGenerator()
    # compute training features and labels
    Xtrain, ytrain = tdg.emg_pre_extracted( CLASSES = Parameters['Calibration']['CUE_LIST'],
                                            PREVIOUS_FILE = '',
                                            practice_session = 0,
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
                                            FL = False,
                                            subject = SUBJECT)

    if not SMG:
        reviewer = SepRepPlot(SUBJECT = SUBJECT, 
                                CLASSES = Parameters['Calibration']['CUE_LIST'], 
                                Xtrain = Xtrain, 
                                ytrain = ytrain, 
                                CLASSIFIER = Parameters['Classification']['CLASSIFIER'], 
                                ALL_CLASSES = Parameters['Calibration']['ALL_CLASSES'],
                                CLUSTER_TRAINING_SAMPLES = Parameters['Classification']['CLUSTER_TRAINING_SAMPLES'],
                                realtime = False,
                                Parameters = Parameters, 
                                perc_lo = [0 for i in range(len(Parameters['Calibration']['CUE_LIST']))] , 
                                perc_hi = [1 for i in range(len(Parameters['Calibration']['CUE_LIST']))], 
                                threshold_ratio = 0.3 )
    if offline_test:

        if SMG:

            SMG_FILE = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), 
                 'Data', 'proportional dataset', 'regular', 'without transitions', 'processedDataset_subject1_quickTransition.mat')
    
            training_data = loadmat( SMG_FILE )

            SMG_FILE = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), 
                        'Data', 'proportional dataset', 'regular', 'with transitions', 'processedDataset_subject1_quickTransition.mat')

            testing_data = loadmat( SMG_FILE )

            classes = ['rest','index','middle','power','tripod','key','wrist']
            classes_in = ['rest','index','power','tripod']
            labels = [0,1,2,3,4,5,6]

            for i in range(len(classes)):
                classes[i] = classes[i].upper()
            for i in range(len(classes_in)):
                classes_in[i] = classes_in[i].upper()

            Xtrain_train = training_data['data']
            ytrain_train = training_data['labels']
            ytrain_train = ytrain_train.flatten()

            Xtrain_test = testing_data['data']
            ytrain_test = testing_data['labels']
            ytrain_test = ytrain_test.flatten()

            Class_data = {}
            Class_labels = {}
            Class_test_data = {}
            Class_test_labels = {}

            for i in ytrain_train:
                if i == 2:
                    print(i)

            for key,i in enumerate(classes):

                if i in classes_in:

                    if key > 2:
                
                        Class_data[i] = Xtrain_train[ytrain_train==key,:]
                        Class_labels[i] = (np.ones(Xtrain_train[ytrain_train==key,:].shape[0])*labels[key]-1).astype(int)

                        Class_test_data[i] = Xtrain_test[ytrain_test==key,:]
                        Class_test_labels[i] = (np.ones(Xtrain_test[ytrain_test==key,:].shape[0])*labels[key]-1).astype(int)
                    
                    else:

                        Class_data[i] = Xtrain_train[ytrain_train==key,:]
                        Class_labels[i] = (np.ones(Xtrain_train[ytrain_train==key,:].shape[0])*labels[key]).astype(int)

                        Class_test_data[i] = Xtrain_test[ytrain_test==key,:]
                        Class_test_labels[i] = (np.ones(Xtrain_test[ytrain_test==key,:].shape[0])*labels[key]).astype(int)


            Xtrain_test = []
            ytrain_test = []
            Xtrain_train = []
            ytrain_train = []
            
            for i in classes_in:
                current_class = Class_data[i]
                current_labels = Class_labels[i]

                Xtrain_train.append(Class_data[i])
                Xtrain_test.append(Class_test_data[i])

                ytrain_train.append(Class_labels[i])
                ytrain_test.append(Class_test_labels[i])

            Xtrain_test = np.vstack(Xtrain_test)
            ytrain_test = np.hstack(ytrain_test).flatten()
            Xtrain_train = np.vstack(Xtrain_train)
            ytrain_train = np.hstack(ytrain_train).flatten()

            reviewer = SepRepPlot(SUBJECT = SUBJECT, 
                                    CLASSES = classes_in, 
                                    Xtrain = Xtrain_train, 
                                    ytrain = ytrain_train, 
                                    CLASSIFIER = Parameters['Classification']['CLASSIFIER'], 
                                    ALL_CLASSES = classes_in,
                                    CLUSTER_TRAINING_SAMPLES = ytrain_test.size,
                                    realtime = False,
                                    Parameters = Parameters, 
                                    perc_lo = [0 for i in range(len(classes_in))] , 
                                    perc_hi = [1 for i in range(len(classes_in))], 
                                    threshold_ratio = 0.3 )

            while not reviewer._exit_event.is_set():
                for i in range(Xtrain_test.shape[0]):
                    if reviewer._exit_event.is_set():
                        break
                    data = Xtrain_test[i, :]
                    #print( 'Expected Class: ' + sorted_labels[ytrain_test[i]], end = '     ', flush = True )
                    reviewer.add(data)
                    time.sleep( 0.02 )
                
                Xtrain_test = reviewer.Xtrain
                Xtrain = reviewer.Xtrain

            '''for i in range(Xtrain.shape[0]):
                data = Xtrain[i, :]
                reviewer.add(data)
                time.sleep( 0.02 )'''

        else:
            Xtrain_test = Xtrain
            while not reviewer._exit_event.is_set():
                for i in range(Xtrain_test.shape[0]):
                    if reviewer._exit_event.is_set():
                        break
                    data = Xtrain_test[i, :]
                    reviewer.add(data)
                    time.sleep( 0.01 )
                    '''if ytrain[i] == 2:
                        reviewer.add(data)
                        time.sleep( 0.1 )'''

    else:
        local_surce_info = address()

        print( 'Creating EMG feature filters...', end = '', flush = True )
        td5 = TimeDomainFilter()
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

        smoothor = Smoothing( Parameters )


        print( 'Starting data collection...', end = '\n', flush = True )
        emg.run()

        emg_window = []
        
        while not reviewer._exit_event.is_set():
            
            data = emg.state

            if data is not None:
                emg_window.append( data[:Parameters['General']['DEVICE_ELECTRODE_COUNT']].copy() / EMG_SCALING_FACTOR )
            
            if len( emg_window ) == Parameters['General']['WINDOW_SIZE']:
                win = np.vstack( emg_window )
                feat = np.hstack( [ td5.filter( win ), fft.filter( win ) ] )[:-Parameters['General']['DEVICE_ELECTRODE_COUNT']]                 
                feat_filtered = smoothor.filter( feat )
                emg_window = emg_window[Parameters['General']['WINDOW_STEP']:]
                
                reviewer.add(feat_filtered)

        emg.close()
        