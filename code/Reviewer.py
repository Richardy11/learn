import copy
import multiprocessing as mp
import os
import pickle
import sys
import math
import time
import traceback
import warnings
from collections import deque
import matplotlib
import numpy as np
from scipy.io import savemat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler

matplotlib.use( 'QT5Agg')

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider, TextBox
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
from Q15Functions import Q15Functions

from local_addresses import address
from global_parameters import global_parameters

local_source_info = address()
downsample = DownSampler()

warnings.simplefilter(action='ignore', category=FutureWarning)

class ReviewerPlot:
    def __init__(   self, CLASSES, SUBJECT, Xtrain, ytrain, CLASSIFIER, ALL_CLASSES, CLUSTER_TRAINING_SAMPLES, 
                    realtime = False, Parameters = {}, perc_lo = [], perc_hi = [], threshold_ratio = 0.5, recalculate_RLDA = True, proj = [], stdscaler = [] ):
    
        FILE_DIRECTORY = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), 'data', 'Subject_%s' % SUBJECT ) # if rescu.py is in main folder

        self.SUBJECT = SUBJECT
        self.realtime = realtime

        self.new_train = True
        self.CLUSTER_TRAINING_SAMPLES = CLUSTER_TRAINING_SAMPLES
        
        self.CLASSES = CLASSES
        self.num_classes = len( self.CLASSES )
        self.class_subdict_size = self.CLUSTER_TRAINING_SAMPLES

        self.new_data_buffer = deque()

        self.simQ15 = False

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

        self.currentlyGatheringNewData = False

        self.ica_scatter = FastICA(n_components=3)

        self.recalculate_RLDA = recalculate_RLDA
        if self.recalculate_RLDA:
            self.stdscaler = MinMaxScaler()
            self.proj = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
        else:
            self.stdscaler = stdscaler
            self.proj = proj

        # compute training features and labels
        self.Xtrain, self.ytrain = Xtrain, ytrain

        self.classifier = CLASSIFIER
        self.visualization_state = 'Clusters'   

        self.perc_lo = perc_lo
        self.perc_hi = perc_hi
        self.threshold_ratio = threshold_ratio     

        self.reinit_classifier()

        self.Parameters = Parameters
        if not self.realtime:
            self.smoothor = Smoothing( self.Parameters )

        # multiprocessing variables
        self._run_multiproc()

    def reinit_classifier(self):
        self.run_classifier = RunClassifier(   class_subdict_size = self.class_subdict_size, 
                                                Xtrain = self.Xtrain, 
                                                ytrain = self.ytrain, 
                                                classes = self.CLASSES,
                                                perc_lo = self.perc_lo,
                                                perc_hi = self.perc_hi,
                                                threshold_ratio = 0.5)#self.threshold_ratio )

        self.class_pairs, self.radius = self.run_classifier.init_classifiers(   self.Xtrain, self.ytrain, self.CLASSES, 
                                                                                perc_lo = self.perc_lo,
                                                                                perc_hi = self.perc_hi,
                                                                                threshold_ratio = 0.5)#self.threshold_ratio)

        #self.threshold_radius = self.run_classifier.SpatialClassifier.threshold_radius
        self.onset_threshold = self.threshold_ratio #self.run_classifier.SpatialClassifier.onset_threshold

    def _get_class_color(self):

        current_colors = []
        for i in range(self.num_classes):
            current_colors.append(self.colors[self.CLASSES[i]])

        current_colors.append(self.colors['new class'])

        return current_colors

    def _run_multiproc(self):

        self._queue = mp.Queue()
        self._queue_segment = mp.Queue()
        

        self._exit_event = mp.Event()
        self._plotter = mp.Process( target = self._plot )

        # start plotting process
        self._plotter.start()

    def _plot(self):

        def progbar():
            tax.clear()
            percent = (len(self.new_data_buffer) / ( self.class_subdict_size ))
            num_eq = int( 17 * percent )
            num_spec = 17-num_eq
            progtxt = 'Data Buffer' + '[' + ( num_eq * '=' ) + ( num_spec * '   ' ) + ']   ' + str(int(100*percent)) + '%'
            # progressbar text
            self.prog_box = tax.text(0,0.5, progtxt, horizontalalignment='right', verticalalignment='center', transform=tax.transAxes, size='smaller')
            tax.xaxis.set_visible(False)
            tax.yaxis.set_visible(False)
    
        def projection():
            sax.clear()
            if self.num_classes > 3:
                if self.recalculate_RLDA:
                    #TODO Q15 branch
                    if self.simQ15:
                        Xtrain_scaled = self.stdscaler.fit_transform(self.Xtrain)
                        self.proj.fit(Xtrain_scaled, self.ytrain)
                        self.q15 = Q15Functions()
                        self.q15.fit(self.proj.scalings_, self.proj._max_components)
                        Xtrain_scatter = self.q15.transform(Xtrain_scaled, save=True)
                        #Xtrain_scatter = self.proj.fit_transform(Xtrain_scaled, self.ytrain)
                    else:
                        Xtrain_scaled = self.stdscaler.fit_transform(self.Xtrain)
                        Xtrain_scatter = self.proj.fit_transform(Xtrain_scaled, self.ytrain)
                else:
                    Xtrain_scaled = self.stdscaler.transform(self.Xtrain)
                    Xtrain_scatter = self.proj.transform(Xtrain_scaled)
            else:
                Xtrain_scatter = self.ica_scatter.fit_transform(self.Xtrain[:,:8])
            
            self.scat = []
            self.labels = []
            self.label_colors = []
            for i in range(self.num_classes + 2):
                if i == self.num_classes:
                    self.scat.append(sax.scatter( 0, 0, 0, [i], marker='X', s=200, c='white', edgecolor = 'k' ))
                elif i == self.num_classes + 1:
                    self.scat.append(sax.scatter( 0, 0, 0, [i], c='darkgrey', alpha = 0, edgecolor = 'k' ))
                else:
                    temp_class = Xtrain_scatter[self.ytrain == i,:]
                    if self.visualization_state == "Clusters":
                        self.scat.append(sax.scatter( temp_class[:,0], temp_class[:,1], temp_class[:,2], self.ytrain[self.ytrain == i], c = self._get_class_color()[i] ))
                        self.label_colors.append((self.scat[-1]._facecolor3d[0][0], self.scat[-1]._facecolor3d[0][1], self.scat[-1]._facecolor3d[0][2]))# , temp_class[:,2].max() + (temp_class[:,2].max() - np.mean(temp_class[:,2])) / 3
                    self.labels.append(sax.text(np.mean(temp_class[:,0]), np.mean(temp_class[:,1]),  np.mean(temp_class[:,2])+np.linalg.norm(self.radius[i])*1.5, self.CLASSES[i].upper(), None, size = 'x-large', ha='center', weight='semibold'))

            self.crosshair = []
            for i in range(3):
                self.crosshair.append( sax.plot((0, 0), (0, 0), (0, 0), color='black') )

            self.mu = []
            for i in range(len(self.CLASSES)):
                self.mu.append(np.mean( Xtrain_scatter[self.ytrain==i,:] , axis=0 ))

            self.scat_mu = []
            for i in range(self.num_classes):
                self.scat_mu.append(sax.scatter( self.mu[i][0], self.mu[i][1], self.mu[i][2], [i], c = self._get_class_color()[i], s=75 ))

            #TODO: integrate better
            #self.onset_threshold = np.linalg.norm(self.threshold_radius)

            u = np.linspace(0, 2*np.pi, 15)
            v = np.linspace(0, np.pi, 15)
            coord1 = self.mu[0]
            x = self.onset_threshold * np.outer(np.cos(u), np.sin(v)) + coord1[0]
            y = self.onset_threshold * np.outer(np.sin(u), np.sin(v)) + coord1[1]
            z = self.onset_threshold * np.outer(np.ones(np.size(u)), np.cos(v)) + coord1[2]
            threshold_sphere = sax.plot_surface(x, y, z, color=self._get_class_color()[0], alpha=0.05, shade = True, linewidth = 0.05, edgecolors=[0.15,0.15,0.15])

            if self.visualization_state == "Spheres":
                u = np.linspace(0, 2*np.pi, 15)
                v = np.linspace(0, np.pi, 15)
                self.sphere = []
                for i in range(len(self.CLASSES)):
                    coord1 = np.mean( Xtrain_scatter[self.ytrain==i,:] , axis=0 )
                    coord2 = self.radius[i] + coord1
                    x = np.linalg.norm(self.radius[i]) * np.outer(np.cos(u), np.sin(v)) + coord1[0]
                    y = np.linalg.norm(self.radius[i]) * np.outer(np.sin(u), np.sin(v)) + coord1[1]
                    z = np.linalg.norm(self.radius[i]) * np.outer(np.ones(np.size(u)), np.cos(v)) + coord1[2]
                    self.sphere.append(sax.plot_surface(x, y, z, color=self._get_class_color()[i], alpha=0.4, shade = True, linewidth = 0.5, edgecolors=[0.15,0.15,0.15]))

                    try:
                        self.label_colors.append((self.sphere[-1]._facecolors3d[0][0], self.sphere[-1]._facecolors3d[0][1], self.sphere[-1]._facecolors3d[0][2]))
                    except:
                        self.label_colors.append((self.sphere[-1]._facecolor3d[0][0], self.sphere[-1]._facecolor3d[0][1], self.sphere[-1]._facecolor3d[0][2]))
                    #sax.plot((coord1[0], coord2[0]), (coord1[1], coord2[1]), (coord1[2], coord2[2]), color='black', alpha = 0.5)
                

            minmax = [ min([ min(Xtrain_scatter[:, 0]),min(Xtrain_scatter[:, 1]),min(Xtrain_scatter[:, 2]) ]),
                        max( [ max(Xtrain_scatter[:, 0]), max(Xtrain_scatter[:, 1]), max(Xtrain_scatter[:, 2]) ] ) ]

            self.xLim = [ minmax[0], minmax[1] ]
            self.yLim = [ minmax[0], minmax[1] ]
            self.zLim = [ minmax[0], minmax[1] ]

            self.max_camera_distance = math.sqrt(minmax[1]**2*3)*2

            sax.set_xlim(self.xLim)
            sax.set_ylim(self.yLim)
            sax.set_zlim(self.zLim)

            sax.set_xlim(self.xLim)
            sax.set_ylim(self.yLim)
            sax.set_zlim(self.zLim)

            sax.xaxis._axinfo['tick'].update({'inward_factor': 0,
                                                'outward_factor': 0})
            sax.yaxis._axinfo['tick'].update({'inward_factor': 0,
                                                'outward_factor': 0})
            sax.zaxis._axinfo['tick'].update({'inward_factor': 0,
                                                'outward_factor': 0})

            sax.set_xticks(np.linspace(self.xLim[0],self.xLim[1],10))
            sax.set_yticks(np.linspace(self.yLim[0],self.yLim[1],10))
            sax.set_zticks(np.linspace(self.zLim[0],self.zLim[1],10))
            
            a = []
            for i in range(10):
                a.append('')
            sax.set_xticklabels(a)
            sax.set_yticklabels(a)
            sax.set_zticklabels(a)

            sax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            sax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            sax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

            sax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            sax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            sax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

        def textplot(close_class = ''):
            bax.clear()
            if not any(self.selected_texts):
                self.text_box = bax.text(0,0.5, 'Current patterns could be added as new movement', horizontalalignment='left', verticalalignment='center', size='medium', transform=bax.transAxes, c='forestgreen')
            elif self.selected_texts[0]:
                self.text_box = bax.text(0,0.5, self.warning_texts[0], horizontalalignment='left', verticalalignment='center', size='medium', c = 'firebrick')
            elif self.selected_texts[1]:
                self.text_box = bax.text(0,0.5, self.warning_texts[1], horizontalalignment='left', verticalalignment='center', size='medium', c = 'firebrick')
                self.text_box = bax.text(0,0.5, '                                                     ' + close_class, horizontalalignment='left', verticalalignment='center', size='medium', c = 'firebrick', fontweight = 'bold')
            elif self.selected_texts[2]:
                self.text_box = bax.text(0,0.5, self.warning_texts[2], horizontalalignment='left', verticalalignment='center', size='medium', c='forestgreen', fontweight = 'bold')
            bax.xaxis.set_visible(False)
            bax.yaxis.set_visible(False)
            remove_border(bax)

        def remove_border(ax):
            for axis in [ 'top', 'bottom', 'left', 'right' ]:
                ax.spines[ axis ].set_visible( False )

        if self.realtime:
            fig = plt.figure(figsize = ( 6, 6 ), tight_layout = 3)
            matplotlib.rcParams['font.size'] = 7
            mngr = plt.get_current_fig_manager()
            geom = mngr.window.geometry()
            mngr.window.setWindowTitle( "Reviewer" )
            x, y, dx, dy = geom.getRect()
            mngr.window.setGeometry( 510, 470, 550, 550 )
        else:
            fig = plt.figure(figsize = ( 12, 9 ), tight_layout = 3)
            mngr = plt.get_current_fig_manager()
            mngr.window.setWindowTitle( "Reviewer" )

        grid_rows = 9
        grid_columns = 12

        # create GUI layout
        gs = fig.add_gridspec( grid_rows, grid_columns )

        if self.realtime:
            sax = fig.add_subplot(gs[ :, :], projection = '3d')
        else:
            sax = fig.add_subplot(gs[ :, :3*grid_columns//4], projection = '3d')

        projection()

        # Display text tips
        bax = plt.axes([0.25, 0.9, 0.01, 0.1])
        textplot()

        btn_x = 0.85
        btn_y = 0.9
        btn_x_offset = 0.14
        btn_y_offset = 0.025
        btn_y_offset_offset = 0.04

        if not self.realtime:

            # progressbar text
            tax = plt.axes([0.99, 0.66, 0.25, 0.025])
            remove_border(tax)
            progbar()

            # generate radio buttons
            def radio_callback( event ):
                label = self.radio.value_selected
                self.rax_label = label

            rax = plt.axes([0.72, 0.7, 0.2, 0.275])
            if self.realtime:
                self.radio = RadioButtons( rax, self.CLASSES, activecolor = 'green')
            else:
                self.radio = RadioButtons( rax, self.all_classes, activecolor = 'green')
            self.rax_label =  self.all_classes[0]
            self.radio.on_clicked( radio_callback )
            remove_border(rax)

            def overwrite_class(percent):
                class_index = self.CLASSES.index(self.rax_label)

                if percent == 100:
                    if len(self.new_data_buffer) > self.class_subdict_size:
                        temp_class, Idx = downsample.uniform(np.vstack(self.new_data_buffer), self.class_subdict_size )
                    else:
                        temp_class = np.vstack(self.new_data_buffer)

                    self.Xtrain[self.ytrain == class_index,:] = temp_class

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
            
            # add_class button
            def add_class_callback( event ):
                if len(self.new_data_buffer) >= ( self.class_subdict_size ):
                    
                    if self.rax_label not in self.CLASSES:

                        new_Xtrain = []
                        new_ytrain = []
                        class_count = 0

                        for i in range(self.num_classes):
                            class_temp = self.Xtrain[self.ytrain == i, :]
                            
                            temp_class, Idx = downsample.uniform(np.vstack(class_temp), self.class_subdict_size )
                            new_Xtrain.append( temp_class )
                            new_ytrain.append( class_count * np.ones( ( new_Xtrain[-1].shape[0], ) ) )
                            class_count += 1

                        if len(self.new_data_buffer) > self.class_subdict_size:
                            temp_class, Idx = downsample.uniform(np.vstack(self.new_data_buffer), self.class_subdict_size )
                        else:
                            temp_class = np.vstack(self.new_data_buffer)

                        new_Xtrain.append( temp_class )
                        new_ytrain.append( class_count * np.ones( ( new_Xtrain[-1].shape[0], ) ) )

                        self.Xtrain = np.vstack( new_Xtrain )
                        self.ytrain = np.hstack( new_ytrain )

                        self.CLASSES.append(self.rax_label)
                        new_barplot_classes = copy.deepcopy(self.CLASSES)
                        new_barplot_classes.append('New class')
                        self.barplot_classes = new_barplot_classes

                        self.num_classes = self.num_classes+1
                        textplot()
                        
                        self.reinit_classifier()
                        
                        sax.clear()

                        projection()

                        self.new_data_buffer = deque()
                        self.current_bufflen = -1

                        self.new_train = True
                        
                        self.prev_pred = 100
                    else:
                        print('already trained class, try overwriting or updating instead')

                else:
                    print('Not enough datapoints')

            
            # ADD MOVEMENT BUTTON
            if self.realtime:
                pass
            else:
                addax = plt.axes([btn_x, btn_y+btn_y_offset_offset, btn_x_offset, btn_y_offset])
                add_class = Button( addax, 'Add Movement' )            
                add_class.on_clicked( add_class_callback )
                add_class.color = 'darkgray'
                add_class.hovercolor = 'lightgray'

                for axis in [ 'top', 'bottom', 'left', 'right' ]:
                    addax.spines[ axis ].set_linewidth( 1.5 )
            
            # ow_class button
            def overwrite_class_callback( event ):
                if len(self.new_data_buffer) >= ( self.class_subdict_size ):
                    
                    if self.rax_label in self.CLASSES:
                        self.Xtrain = overwrite_class(100)
                        textplot()
                        
                        self.reinit_classifier()
                        
                        sax.clear()

                        projection()

                        self.new_data_buffer = deque()
                        self.current_bufflen = -1

                        self.new_train = True
                        
                        self.prev_pred = 100
                    else:
                        print('untrained class, try adding instead')

                else:
                    print('Not enough datapoints')
            
            # OVERWRITE MOVEMENT BUTTON
            owax = plt.axes([btn_x, btn_y, btn_x_offset, btn_y_offset])
            ow_class = Button( owax, 'Overwrite Movement' )
                
            ow_class.on_clicked( overwrite_class_callback )
            ow_class.color = 'darkgray'
            ow_class.hovercolor = 'lightgray'

            for axis in [ 'top', 'bottom', 'left', 'right' ]:
                owax.spines[ axis ].set_linewidth( 1.5 )

            def update_callback( event ):
                if len(self.new_data_buffer) >= ( self.class_subdict_size // 2 ):
                    
                    if self.rax_label in self.CLASSES:
                        self.Xtrain = overwrite_class(50)

                        self.reinit_classifier()
                        
                        sax.clear()

                        projection()

                        self.new_data_buffer = deque()
                        self.current_bufflen = -1

                        self.new_train = True
                        
                        self.prev_pred = 100
                    else:
                        print('untrained class, try adding instead')

            #update button
            upax = plt.axes([btn_x, btn_y-btn_y_offset_offset, btn_x_offset, btn_y_offset])
            update_class = Button( upax, 'Update Movement' )
            update_class.on_clicked( update_callback )
            update_class.color = 'darkgray'
            update_class.hovercolor = 'lightgray'

            for axis in [ 'top', 'bottom', 'left', 'right' ]:
                upax.spines[ axis ].set_linewidth( 1.5 )

            # reset buffer button
            def reset_buffer_callback( event ):
                self.currentlyGatheringNewData = True
                self.new_data_buffer = deque()
                self.current_bufflen = -1
                self.selected_texts = [False, False, False]
                self.scat[-1]._offsets3d = (np.array([0]),np.array([0]),np.array([0]))

            rbax = plt.axes([btn_x, btn_y-btn_y_offset_offset*2, btn_x_offset, btn_y_offset])
            reset_buffer = Button( rbax, 'Gather New Data' )
            reset_buffer.on_clicked( reset_buffer_callback )
            reset_buffer.color = 'goldenrod'
            reset_buffer.hovercolor = 'gold'

            for axis in [ 'top', 'bottom', 'left', 'right' ]:
                rbax.spines[ axis ].set_linewidth( 1.5 )

            if not self.realtime:
                # delete movement button
                def delete_movement_callback( event ):
                    class_index = self.CLASSES.index(self.rax_label)

                    self.Xtrain = np.delete( self.Xtrain, np.where(self.ytrain == class_index), 0 )
                    self.ytrain = np.delete( self.ytrain, np.where(self.ytrain == self.num_classes - 1), 0 )
                    self.num_classes -= 1
                    self.CLASSES.remove( self.rax_label )
                    
                    self.barplot_classes.remove( self.rax_label )
                    
                    # reset the barplot axis
                    textplot()

                    self.reinit_classifier()

                    sax.clear()
                    
                    projection()            
                    
                    self.new_data_buffer = deque()
                    self.current_bufflen = -1

                    self.new_train = True
                    

                    self.prev_pred = 100

                dax = plt.axes([btn_x, btn_y-btn_y_offset_offset*3, btn_x_offset, btn_y_offset])
                delete_movement_class = Button( dax, 'Delete Movement' )
                delete_movement_class.on_clicked( delete_movement_callback )
                delete_movement_class.color = 'goldenrod'
                delete_movement_class.hovercolor = 'gold'

                for axis in [ 'top', 'bottom', 'left', 'right' ]:
                    dax.spines[ axis ].set_linewidth( 1.5 )

            # exit button
            def exit_button_callback( event ):
                self.close()

            ebax = plt.axes([0.87, 0.1, 0.12, 0.05])
            exit_button = Button( ebax, 'Exit' )
            exit_button.on_clicked( exit_button_callback )
            exit_button.color = 'firebrick'
            exit_button.hovercolor = 'indianred'

            for axis in [ 'top', 'bottom', 'left', 'right' ]:
                ebax.spines[ axis ].set_linewidth( 1.5 )

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

            sbax = plt.axes([0.74, 0.1, 0.12, 0.05])
            save_button = Button( sbax, 'Save' )
            save_button.on_clicked( save_button_callback )
            save_button.color = 'cornflowerblue'
            save_button.hovercolor = 'royalblue'

            for axis in [ 'top', 'bottom', 'left', 'right' ]:
                sbax.spines[ axis ].set_linewidth( 1.5 )

            slax = plt.axes([0.74, 0.49, 0.1, 0.01])
            slax.text(0,0.5, 'Data processing options', horizontalalignment='left', verticalalignment='center', size='medium', transform=slax.transAxes, c='black', fontweight = 'bold')
            slax.xaxis.set_visible(False)
            slax.yaxis.set_visible(False)
            remove_border(slax)

            # simultaneous button
            self.simultaneous_mode = False
            def simultaneous_button_callback( event ):
                if self.simultaneous_mode:
                    self.simultaneous_mode = False
                    simultaneous_button.label.set_text('Simultanous Classes')
                else:
                    self.simultaneous_mode = True
                    simultaneous_button.label.set_text('Single Classes')

            simax = plt.axes([0.74, 0.45, btn_x_offset, btn_y_offset])
            simultaneous_button = Button( simax, 'Simultanous Classes' )
            simultaneous_button.on_clicked( simultaneous_button_callback )
            simultaneous_button.color = 'mediumorchid'
            simultaneous_button.hovercolor = 'magenta'

            for axis in [ 'top', 'bottom', 'left', 'right' ]:
                simax.spines[ axis ].set_linewidth( 1.5 )


        if self.realtime:
            vlax = plt.axes([0.04, 0.1, 0.1, 0.01])
        else:
            vlax = plt.axes([0.74, 0.61, 0.1, 0.01])
        vlax.text(0,0.5, 'Visualization options', horizontalalignment='left', verticalalignment='center', size='medium', transform=vlax.transAxes, c='black', fontweight = 'bold')
        vlax.xaxis.set_visible(False)
        vlax.yaxis.set_visible(False)
        remove_border(vlax)

        # visualization mode button
        def visualize_button_callback( event ):
            if self.visualization_state == 'Spheres':
                self.visualization_state = 'Clusters'
                visualize_button.label.set_text( 'Show Spheres' )
            else:
                self.visualization_state = 'Spheres'
                visualize_button.label.set_text( 'Show Clusters' )
            projection()

        if self.realtime:
            vax = plt.axes([0.04, 0.06, btn_x_offset, btn_y_offset])
        else:
            vax = plt.axes([0.74, 0.57, btn_x_offset, btn_y_offset])
        if self.visualization_state == 'Clusters':
            visualize_button = Button( vax, 'Show Spheres' )
        else:
            visualize_button = Button( vax, 'Show Clusters' )

        visualize_button.on_clicked( visualize_button_callback )
        visualize_button.color = 'mediumorchid'
        visualize_button.hovercolor = 'magenta'

        for axis in [ 'top', 'bottom', 'left', 'right' ]:
            vax.spines[ axis ].set_linewidth( 1.5 )


        # rotation button
        self.rotation = True
        def rotation_button_callback( event ):
            if self.rotation:
                self.rotation = False
                rotation_button.label.set_text('Start Rotation')
            else:
                self.rotation = True
                rotation_button.label.set_text('Stop Rotation')

        if self.realtime:
            rotax = plt.axes([0.04, 0.02, btn_x_offset, btn_y_offset])
        else:
            rotax = plt.axes([0.74, 0.53, btn_x_offset, btn_y_offset])
        rotation_button = Button( rotax, 'Stop Rotation' )
        rotation_button.on_clicked( rotation_button_callback )
        rotation_button.color = 'mediumorchid'
        rotation_button.hovercolor = 'magenta'

        for axis in [ 'top', 'bottom', 'left', 'right' ]:
            rotax.spines[ axis ].set_linewidth( 1.5 )      

        self.current_bufflen = -1

        angle = 0

        self.prev_pred = 100
        local_minmax = MinMaxScaler()
        
        frame_counter = 0
        segment_data = []

        #t = time.time()
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
                
            if self.new_train:
                self.new_train = False

            if data is not None:
                '''print('Iteration:  ' ,time.time()-t)
                t = time.time()'''
                
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
                        data_scaled = self.stdscaler.transform(data_mean)
                        if self.simQ15:                     
                            data_scatter = self.q15.transform(data_scaled)
                        else:    
                            data_scatter = self.proj.transform(data_scaled)
                        
                    else:
                        data_scatter = self.ica_scatter.transform(data_mean[:,:8])
                    self.scat[-2]._offsets3d = (data_scatter[:,0],data_scatter[:,1],data_scatter[:,2])
                    xCross = data_scatter[:,0][0]
                    yCross = data_scatter[:,1][0]
                    zCross = data_scatter[:,2][0]
                    for i in range(3):
                        if i == 0:
                            self.crosshair[i][0]._verts3d = ((self.xLim[0], self.xLim[1]), (yCross, yCross), (zCross, zCross))
                        elif i == 1:
                            self.crosshair[i][0]._verts3d = ((xCross, xCross), (self.yLim[0], self.yLim[1]), (zCross, zCross))
                        elif i == 2:
                            self.crosshair[i][0]._verts3d = ((xCross, xCross), (yCross, yCross), (self.zLim[0], self.zLim[1]))

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

                if self.rotation:
                    sax.view_init(30, angle)
                    angle += 0.25

                #separation

                r = np.zeros((2,len(self.CLASSES)))                  

                for key, i in enumerate(self.radius):
                    try:
                        r[0,key] = np.linalg.norm(self.mu[key] - data_scatter)
                        if r[0,key] < np.linalg.norm(i):
                            r[1,key] = 1
                    except:
                        print('error')

                r[0,:] = np.ones(len(self.CLASSES)) - local_minmax.fit_transform(r[0,:].reshape(-1, 1))[:,0]
                #predictions

                onset_occured = np.linalg.norm(data_scatter-self.mu[0]) > self.onset_threshold #np.mean(data[:8]) > onset_threshold:

                '''for i in range(len(axes_projection)):
                    self.projection_lines[i][0]._verts3d = ( ( data_scatter[:,0], axes_projection[i][0] ), ( data_scatter[:,1], axes_projection[i][1] ), ( data_scatter[:,2], axes_projection[i][2] ) )    '''    

                if self.realtime == True:
                    pred = external_pred
                else:
                    if onset_occured:
                        pred, axes_projection = self.run_classifier.emg_classify(data_mean, self.onset_threshold, self.classifier, True, True)

                    else:
                        pred, axes_projection = self.run_classifier.emg_classify(data_mean, self.onset_threshold, self.classifier, True, True)
                        pred = [0] 

                if type(pred) == int:
                    pred = [pred]

                grayscale = 0.75

                for i in range(self.num_classes):
                    if self.visualization_state == "Clusters":
                        alpha_val = 0.1 + (r[0,i]/1.5)
                    elif self.visualization_state == "Spheres":
                        alpha_val = 0.1 + (r[0,i]/3)
                    if alpha_val > 1:
                        alpha_val = 1
                    grayscale = 0.65
                    if self.visualization_state == "Clusters":
                        self.scat[i].set_alpha(alpha_val)
                        self.labels[i].set_alpha(alpha_val)
                    elif self.visualization_state == "Spheres":
                        self.sphere[i].set_alpha(alpha_val)

                    self.labels[i].set_color( ( grayscale - (grayscale - self.label_colors[i][0]) * alpha_val,
                                                grayscale - (grayscale - self.label_colors[i][1]) * alpha_val ,
                                                grayscale - (grayscale - self.label_colors[i][2]) * alpha_val ) )                                                          

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

                self.prev_pred = pred

                try:
                    
                    if not r[1,:].any() and onset_occured:
                        if self.currentlyGatheringNewData:
                            self.new_data_buffer.append(data)
                            
                            self.selected_texts = [False, False, True]
                            textplot()
                            self.scat[-1].set_alpha(0.55)
                            
                            class_buffer_empty = self.scat[-1]._offsets3d[0] == (np.array([0]), np.array([0]), np.array([0]))

                            if class_buffer_empty.all():
                                self.scat[-1]._offsets3d = (data_scatter[:,0],data_scatter[:,1],data_scatter[:,2])
                            else:
                                self.scat[-1]._offsets3d = np.concatenate((self.scat[-1]._offsets3d, (data_scatter[:,0],data_scatter[:,1],data_scatter[:,2])), axis=1)

                            for i in range(3):
                                self.crosshair[i][0].set_color('g')
                                self.crosshair[i][0].set_linestyle('-')
                                self.crosshair[i][0].set_alpha(1)
                                self.scat[-2]._facecolor3d = (0.133333,0.5451,0.133333,1)

                        else:
                            
                            self.selected_texts = [False, False, False]
                            textplot()

                            for i in range(3):
                                self.crosshair[i][0].set_color('g')
                                self.crosshair[i][0].set_linestyle(':')
                                self.crosshair[i][0].set_alpha(0.6)
                                self.scat[-2]._facecolor3d = (1,1,1,1)

                    else:
                            
                        if r[1,:].any() and self.selected_texts[1] is False:
                            self.selected_texts[1] = True
                            self.selected_texts[2] = False
                            textplot(self.CLASSES[np.argmax(r[0,:])])

                        elif onset_occured and self.selected_texts[2] is False:
                            self.selected_texts[0] = True
                            self.selected_texts[1] = False
                            textplot()

                        for i in range(3):
                            self.crosshair[i][0].set_color('k')
                            self.crosshair[i][0].set_linestyle(':')
                            self.crosshair[i][0].set_alpha(1)
                            self.scat[-2]._facecolor3d = (1,1,1,1)
                    
                            

                    try:
                        if len(self.new_data_buffer) <= ( self.class_subdict_size ) and self.current_bufflen != len(self.new_data_buffer):
                            progbar()
                    except:
                        pass

                except Exception as e:
                    print(e)
                    traceback.print_exc()

                try:
                    if len(self.new_data_buffer) > ( self.class_subdict_size ):
                        try:
                            if add_class.color is not 'forestgreen':
                                add_class.color = 'forestgreen'
                                add_class.hovercolor = 'mediumseagreen'
                        except:
                            pass
                        if ow_class.color is not 'forestgreen':
                            ow_class.color = 'forestgreen'
                            ow_class.hovercolor = 'mediumseagreen'
                        self.selected_texts = [False, False, False]
                        textplot()
                        self.currentlyGatheringNewData = False
                        self.new_data_buffer.popleft()
                        
                    elif len(self.new_data_buffer) > ( self.class_subdict_size // 2 ):
                        if update_class.color is not 'lightgreen':
                            update_class.color = 'lightgreen'
                            update_class.hovercolor = 'mediumseagreen'
                    try:
                        if len(self.new_data_buffer) < ( self.class_subdict_size ) and add_class.color is not 'darkgray':
                            add_class.color = 'darkgray'
                            add_class.hovercolor = 'lightgray'
                    except:
                        pass
                    if len(self.new_data_buffer) < ( self.class_subdict_size ) and ow_class.color is not 'darkgray':
                        ow_class.color = 'darkgray'
                        ow_class.hovercolor = 'lightgray'
                    if len(self.new_data_buffer) <= ( self.class_subdict_size // 2 ) and update_class.color is not 'darkgray':
                        update_class.color = 'darkgray'
                        update_class.hovercolor = 'lightgray'

                    self.current_bufflen = len(self.new_data_buffer)
                except:
                    pass

                #print(time.time() - t3)
            plt.pause( 0.001 )

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

if __name__ == '__main__':

    from TrainingDataGeneration import TrainingDataGenerator
    from scipy.io import loadmat
    import matplotlib.cm as cm

    SUBJECT = 1
    SMG = True
    offline_test = True

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
        reviewer = ReviewerPlot(SUBJECT = SUBJECT, 
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

            SMGFolder = 'SMGGood'
            #SMGFolder = 'SMGBad'

            PARAM_FILE = os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ), 
                 'data', 'Subject_%s' % SUBJECT, SMGFolder, 'RESCU_Protocol_Subject_%s.pkl' % SUBJECT )

            with open( PARAM_FILE, 'rb' ) as pkl:
                Parameters = pickle.load( pkl )

            SMG_FILE = os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ), 
                 'data', 'Subject_%s' % SUBJECT, 'SMG', 'trial 1', 'TestData.pkl')
    
            with open( SMG_FILE, 'rb' ) as pkl:
                testing_data = pickle.load( pkl )

            Xtrain_test = np.vstack(testing_data)


    
            Xtrain_calib_test, ytrain = tdg.smg_pre_extracted(CLASSES = Parameters['Calibration']['CUE_LIST'], subject = SUBJECT, folder='trial 2')

            Xtrain_test = np.concatenate((Xtrain_calib_test, Xtrain_test), axis = 0)



            SMG_FILE = os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ), 
                 'data', 'Subject_%s' % SUBJECT, SMGFolder, 'calibrate_subject_1.mat')
    
            Xtrain, ytrain = tdg.smg_pre_extracted(CLASSES = Parameters['Calibration']['CUE_LIST'], subject = SUBJECT, folder='trial 1')

            Xtrain_test = np.concatenate((Xtrain, Xtrain_calib_test, Xtrain_test), axis = 0)

            '''stdscaler = MinMaxScaler()

            Xtrain = np.delete(Xtrain, list(range(260,280,1)), 1)
            Xtrain_test = np.delete(Xtrain_test, list(range(260,280,1)), 1)

            Xtrain = np.delete(Xtrain, list(range(360,400,1)), 1)
            Xtrain_test = np.delete(Xtrain_test, list(range(360,400,1)), 1)

            Xtrain = np.delete(Xtrain, list(range(0,40,1)), 1)
            Xtrain_test = np.delete(Xtrain_test, list(range(0,40,1)), 1)

            Xtrain = np.delete(Xtrain, list(range(100,115,1)), 1)
            Xtrain_test = np.delete(Xtrain_test, list(range(100,115,1)), 1)

            Xtrain = np.delete(Xtrain, list(range(135,150,1)), 1)
            Xtrain_test = np.delete(Xtrain_test, list(range(135,150,1)), 1)

            stdscaler.fit(Xtrain)
            Xtrain = stdscaler.transform(Xtrain)
            Xtrain_test = stdscaler.transform(Xtrain_test)'''

            if True:
                trimsXtrain = [125, 250, 375]
                trimsXtrain_test = [125, 2700, 2825, 1450, 1575, 4000, 4125]
                
                fig, stax = plt.subplots(figsize=(16,10), nrows= 2, ncols= 1, dpi= 80, tight_layout = 6)
                stax[0].imshow(Xtrain.T, interpolation='bilinear', cmap=plt.cm.Greys_r)
                stax[0].set_title('Training Data', fontweight = 'bold')
                stax[1].imshow(Xtrain_test.T, interpolation='bilinear', cmap=plt.cm.Greys_r)
                stax[1].set_title('Test Data', fontweight = 'bold')

                stax[0].vlines(trimsXtrain, ymin=0, ymax=np.repeat(Xtrain.shape[1], len(trimsXtrain)), color='r', alpha=0.75, linewidth=2)
                stax[1].vlines(trimsXtrain_test, ymin=0, ymax=np.repeat(Xtrain.shape[1], len(trimsXtrain_test)), color='r', alpha=0.75, linewidth=2)

                fig, sax = plt.subplots(figsize=(16,10), nrows= 1, ncols= 3, dpi= 80, tight_layout = 6)
                #np.concatenate( (Xtrain_test[:125, :], Xtrain_test[1450:1575, :], Xtrain_test[2700:2825, :], Xtrain_test[4000:4110, :]) , axis=0)
                '''trainDataToPlotAx0 = np.concatenate( ( Xtrain[:125, :], Xtrain_test[:125, :], Xtrain_test[2700:2825, :] ), axis=0) 
                trainDataToPlotAx1 = np.concatenate( ( Xtrain[125:250, :], Xtrain_test[1450:1575, :] ), axis=0)
                trainDataToPlotAx2 = np.concatenate( ( Xtrain[250:375, :], Xtrain_test[4000:4125, :] ), axis=0)'''

                trainDataToPlotAx0 = np.concatenate( ( Xtrain_test[2700:2825, :], Xtrain_test[:125, :], Xtrain[:125, :] ), axis=0) 
                trainDataToPlotAx1 = np.concatenate( ( Xtrain_test[1450:1575, :], Xtrain[125:250, :] ), axis=0)
                trainDataToPlotAx2 = np.concatenate( ( Xtrain_test[4000:4125, :], Xtrain[250:375, :] ), axis=0)

                sax[0].imshow(trainDataToPlotAx0.T,interpolation='bilinear', cmap=plt.cm.coolwarm)
                sax[0].set_title('Rest', fontweight = 'bold')
                sax[1].imshow(trainDataToPlotAx1.T,interpolation='bilinear', cmap=plt.cm.coolwarm)
                sax[1].set_title('Power', fontweight = 'bold')
                sax[2].imshow(trainDataToPlotAx2.T,interpolation='bilinear', cmap=plt.cm.coolwarm)
                sax[2].set_title('Tripod', fontweight = 'bold')

                x = np.arange(0,trainDataToPlotAx0.shape[0],1)
                y = np.arange(0,trainDataToPlotAx0.shape[1],1)
                X0,Y0 = np.meshgrid(x,y)

                x = np.arange(0,trainDataToPlotAx1.shape[0],1)
                y = np.arange(0,trainDataToPlotAx1.shape[1],1)
                X1,Y1 = np.meshgrid(x,y)

                x = np.arange(0,trainDataToPlotAx2.shape[0],1)
                y = np.arange(0,trainDataToPlotAx2.shape[1],1)
                X2,Y2 = np.meshgrid(x,y)

                fig = plt.figure(figsize=plt.figaspect(0.5))
                sfax = []

                for i in range(1,4,1):
                    sfax.append(fig.add_subplot(1, 3, i, projection='3d'))

                stdscaler = MinMaxScaler()
                stdscaler.fit(Xtrain)
                Xtrain = stdscaler.transform(Xtrain)
                Xtrain_test = stdscaler.transform(Xtrain_test)

                trainDataToPlotAx0 = np.concatenate( ( Xtrain_test[2700:2825, :], Xtrain_test[:125, :], Xtrain[:125, :] ), axis=0) 
                trainDataToPlotAx1 = np.concatenate( ( Xtrain_test[1450:1575, :], Xtrain[125:250, :] ), axis=0)
                trainDataToPlotAx2 = np.concatenate( ( Xtrain_test[4000:4125, :], Xtrain[250:375, :] ), axis=0)
                
                fig.tight_layout()
                sfax[0].plot_surface(X0, Y0, trainDataToPlotAx0.T, cmap=plt.cm.coolwarm)
                sfax[1].plot_surface(X1, Y1, trainDataToPlotAx1.T, cmap=plt.cm.coolwarm)
                sfax[2].plot_surface(X2, Y2, trainDataToPlotAx2.T, cmap=plt.cm.coolwarm)

                #fig.colorbar(surf)

                for i in range(3):
                    sfax[i].set_ylabel('Features')
                    sfax[i].set_xlabel('Samples')
                    #sfax[i].set_zlabel('Values')
                    sfax[i].view_init(90,90)

            plt.show()
            
            reviewer = ReviewerPlot(SUBJECT = SUBJECT, 
                                    CLASSES = Parameters['Calibration']['CUE_LIST'], 
                                    Xtrain = Xtrain, 
                                    ytrain = ytrain, 
                                    CLASSIFIER = Parameters['Classification']['CLASSIFIER'], 
                                    ALL_CLASSES = Parameters['Calibration']['CUE_LIST'],
                                    CLUSTER_TRAINING_SAMPLES = ytrain.size,
                                    realtime = False,
                                    Parameters = Parameters, 
                                    perc_lo = [0 for i in range(len(Parameters['Calibration']['CUE_LIST']))] , 
                                    perc_hi = [1 for i in range(len(Parameters['Calibration']['CUE_LIST']))], 
                                    threshold_ratio = 0.3 )

            while not reviewer._exit_event.is_set():
                for i in range(Xtrain_test.shape[0]):
                    if reviewer._exit_event.is_set():
                        break
                    data = Xtrain_test[i, :]
                    if i == 625:
                        print('Change to Other calibration data')
                    elif i == 1250:
                        print('Change to Test dta')
                    #print( 'Expected Class: ' + sorted_labels[ytrain_test[i]], end = '     ', flush = True )
                    reviewer.add(data)
                    time.sleep( 0.0002 )
                
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
                    time.sleep( 0.1 )

    else:
        local_surce_info = address()

        print( 'Creating EMG feature filters...', end = '', flush = True )
        td5 = TimeDomainFilter()
        fft = FourierTransformFilter( fftlen = Parameters['General']['FFT_LENGTH'] )
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
        