import queue
import multiprocessing as mp
from turtle import color

import numpy as np

import matplotlib
matplotlib.use( "QT5Agg" )
import matplotlib.pyplot as plt
from copy import deepcopy

from matplotlib.widgets import Slider
from global_parameters import global_parameters

matplotlib.rcParams['font.size'] = 7

class RealTimeGlidePlot:
    def __init__(   self, classes = ('open', 'power', 'pronate', 'supinate', 'tripod'), all_classes = ('open', 'power', 'pronate', 'supinate', 'tripod'),
                    prop_lo = [0.001,0.001,0.001,0.001,0.001,0.001,0.001], prop_hi = [0.03,0.03,0.03,0.03,0.03,0.03,0.03],
                    perc_lo = [0.25,0.25,0.25,0.25,0.25,0.25,0.25], perc_hi = [0.75,0.75,0.75,0.75,0.75,0.75,0.75],  
                    bar_min = 0, bar_max = 0.6, 
                    sampling_rate = 20, method = 'Spatial', spatial_velocity_dicts = {}):
        """
        Constructor 

        Parameters
        ----------
        classes : iterable of str
            The class labels for the control signal
        bar_min : float
            The maximum value for a bar in the plot
        bar_max : float
            The minimum value for a bar in the plot

        Returns
        -------
        obj
            A RealTimeGlidePlot object
        """
        # plotting variables
        try:
            self._classes = deepcopy(classes)
            self._classes.remove('REST')
        except:
            self._classes = classes
        self._all_classes = all_classes
        self._n_classes = len( self._classes )

        max_prop = 0
        min_prop = 1000000
        self.max_ratio = []

        for key in spatial_velocity_dicts.keys():
            min_prop = spatial_velocity_dicts[key]['min'] if spatial_velocity_dicts[key]['min'] < min_prop else min_prop
            max_prop = spatial_velocity_dicts[key]['max'] if spatial_velocity_dicts[key]['max'] > max_prop else max_prop

        for key in spatial_velocity_dicts.keys():
            self.max_ratio.append(spatial_velocity_dicts[key]['max']/max_prop)

        global_params = global_parameters()
        self.palette = global_params.palette

        self.classifier = method

        self._thresholds = {}
        for key, cl in enumerate(self._classes):
            self._thresholds[cl] = [prop_lo[key], prop_hi[key]]

        self._percentages = {}
        for key, cl in enumerate(self._classes):
            self._percentages[cl] = [perc_lo[key], perc_hi[key]*self.max_ratio[key]]
            
        self._bar_min = bar_min if self.classifier != 'Spatial' else 0
        self._bar_max = bar_max if self.classifier != 'Spatial' else 1.5
        self._buffer_size = int( 5.0 * sampling_rate )

        # multiprocessing variables
        self._queue = mp.Queue()
        self._change_queue = mp.Queue()
        self._threshold_queue = mp.Queue()
        self._percentages_queue = mp.Queue()
        
        self._exit_event = mp.Event()
    
        self._plotter = mp.Process( target = self._plot )
        self._plotter.start()

    def _plot(self):
        """
        The plotting subprocess function
        """
        fig = plt.figure( figsize = ( 5.0, 4.0 ) )
        mngr = plt.get_current_fig_manager()
        geom = mngr.window.geometry()
        x,y,dx,dy = geom.getRect()
        mngr.window.setGeometry(510,35,550, 400)
        mngr.window.setWindowTitle( "Proportional Control" )
        gs = fig.add_gridspec( 6 * self._n_classes, 6 )
        
        ax = fig.add_subplot( gs[:-int(2*self._n_classes),:4] )

        # create the bar graph
        xplot = np.arange( 0, self._n_classes )
        yplot = np.zeros( shape = ( self._n_classes, ) )
        bar = []
        for i in self._classes:
            color = self.palette[ self._all_classes.index( i ) ]
            bar.append(ax.bar( xplot, yplot, edgecolor = color, color = color, zorder = -1 ))
        

        # create low and high threshold plots
        lo_eps = []
        hi_eps = []
        for i, cl in enumerate(self._classes):
            xeps = [ i - 0.5, i + 0.5 ]
            if self.classifier == 'Spatial':
                lo_eps.append( ax.plot( xeps, 100 * self._percentages[cl][0] * np.ones( shape = ( 2, ) ), color = 'red', linestyle = '--' ) )
                hi_eps.append( ax.plot( xeps, 100 * self._percentages[cl][1] * np.ones( shape = ( 2, ) ), color = 'red', linestyle = '--' ) )
            else:
                lo_eps.append( ax.plot( xeps, 100 * self._thresholds[cl][0] * np.ones( shape = ( 2, ) ), color = 'red', linestyle = '--' ) )
                hi_eps.append( ax.plot( xeps, 100 * self._thresholds[cl][1] * np.ones( shape = ( 2, ) ), color = 'red', linestyle = '--' ) )
        
        # create threshold sliders
        lo_sliders = []
        lo_sliders_ax = []

        hi_sliders = []
        hi_sliders_ax = []

        eps_resolution = 0.01 * ( self._bar_max - self._bar_min )
        for i, cl in enumerate(self._classes):
            # create threshold sliders
            color = self.palette[ self._all_classes.index( cl ) ]
            lo_sliders_ax.append( fig.add_subplot( gs[4*i+2,4] ) )
            if self.classifier == 'Spatial':
                lo_sliders.append( Slider( lo_sliders_ax[i], '', self._bar_min, self._bar_max - eps_resolution, valinit = self._percentages[cl][0], valstep = eps_resolution, color=color ) )
            else:
                lo_sliders.append( Slider( lo_sliders_ax[i], '', self._bar_min, self._bar_max - eps_resolution, valinit = self._thresholds[cl][0], valstep = eps_resolution, color=color ) )
            lo_sliders[i].valtext.set_visible( False )
            lo_sliders_ax[i].set_title( 'LOW', fontsize = 8 )
            #lo_sliders_ax[i].set_title( '%d' % self._thresholds[i], fontsize = 10 )

            hi_sliders_ax.append( fig.add_subplot( gs[4*i+2,5] ) )
            if self.classifier == 'Spatial':
                hi_sliders.append( Slider( hi_sliders_ax[i], '', self._bar_min + eps_resolution, self._bar_max, valinit = self._percentages[cl][1], valstep = eps_resolution, color=color ) )
            else:
                hi_sliders.append( Slider( hi_sliders_ax[i], '', self._bar_min + eps_resolution, self._bar_max, valinit = self._thresholds[cl][1], valstep = eps_resolution, color=color ) )
            
            hi_sliders_ax[i].set_title( 'HIGH', fontsize = 8 )
            #hi_sliders_ax[i].set_title( '%d' % self._thresholds[self._n_classes + i], fontsize = 10 )

            hi_sliders[i].valtext.set_text( ' ' + self._classes[i].replace(' ', '\n ') )

            # create threshold callbacks
            def lo_slider_update( val, idx = i ):
                lo_eps[ idx ][ 0 ].set_ydata( 100 * val * np.ones( shape = ( 2, ) ) )
                if self.classifier == 'Spatial':
                    self._percentages[self._classes[idx]][0] = val/self.max_ratio[idx]
                else:
                    self._thresholds[self._classes[idx]][0] = val
                lo_sliders_ax[idx].set_title( 'LOW', fontsize = 8 )
                #lo_sliders_ax[idx].set_title( '%d' % val, fontsize = 10 )
                hi_sliders[idx].valmin = val + eps_resolution # update slider range
                self._change_queue.put( True, timeout = 1e-3 )
                if self.classifier == 'Spatial':
                    self._percentages_queue.put( self._percentages, timeout = 1e-3 )
                else:
                    self._threshold_queue.put( self._thresholds, timeout = 1e-3 )
            
            def hi_slider_update( val, idx = i ):
                hi_eps[ idx ][ 0 ].set_ydata( 100 * val * np.ones( shape = ( 2, ) ) )
                if self.classifier == 'Spatial':
                    self._percentages[self._classes[idx]][1] = val/self.max_ratio[idx]
                else:
                    self._thresholds[self._classes[idx]][1] = val
                hi_sliders_ax[idx].set_title( 'HIGH', fontsize = 8 )
                #hi_sliders_ax[idx].set_title( '%d' % val, fontsize = 10 )
                hi_sliders[idx].valtext.set_text( ' ' + self._classes[idx].replace(' ', '\n ') )
                lo_sliders[idx].valmax = val - eps_resolution # update slider range
                self._change_queue.put( True, timeout = 1e-3 )
                if self.classifier == 'Spatial':
                    self._percentages_queue.put( self._percentages, timeout = 1e-3 )
                else:
                    self._threshold_queue.put( self._thresholds, timeout = 1e-3 )
                
            
            # bind callbacks
            lo_sliders[i].on_changed( lo_slider_update )
            hi_sliders[i].on_changed( hi_slider_update )

        # prettify the plot
        ax.set_ylim(  self._bar_min*100, self._bar_max*100 + 0.05 * self._bar_max*100 )
        ax.set_ylabel( "Class Value" )

        ax.set_xticks( xplot )
        ax.set_xticklabels( self._classes, rotation = 45 )
        # ax.set_xlabel( "Predicted Movement Class" )

        ax.set_title( "Proportional range" )
        
        ax = fig.add_subplot( gs[0,4:] )
        ax.axis( 'off' )
        ax.set_title( 'THRESHOLD SLIDERS' )

        # create velocity subplot
        lines = []
        ax = fig.add_subplot( gs[-self._n_classes:,:] )
        for i, cl in enumerate(self._classes):
            color = self.palette[ self._all_classes.index( cl ) ]
            xplot = np.linspace( 0, 1.0, self._buffer_size )
            yplot = np.zeros( shape = ( self._buffer_size, ) )
            lines.append( ax.plot( xplot, yplot, label = self._classes[i], color = color ) )
        
        ax.legend( loc = 'lower center', bbox_to_anchor = (0.5, -0.8), ncol = int( np.ceil( self._n_classes / 2.0 ) ), frameon = False )
        ax.set_ylabel( 'Velocity' )
        ax.set_ylim( -0.05, 1.05 )
        ax.set_xticks( [] )

        plt.show( block = False )

        while not self._exit_event.is_set():
            # pull data
            data = []
            while self._queue.qsize() > 0:
                try:
                    data.append( self._queue.get( timeout = 1e-3 ) )
                except queue.Empty:
                    pass

            # update plots
            if len( data ):
                
                pred, speed, val = zip( *data )

                # TODO update the last bar graph value for each class
                for k in range( 0, self._n_classes ):
                    try:
                        idx = dict( map( reversed, enumerate( pred ) ) )[self._classes[k]]
                        bar[k][k].set_height( val[idx] * 100 )
                    except KeyError:
                        bar[k][k].set_height( 0 )

                # update appropriate velocity
                new_ydata = np.zeros( shape = ( self._n_classes, len( data ) ) )
                for i in range( 0, self._n_classes ):
                    for j in range( 0, len( data ) ):
                        new_ydata[i,j] = speed[j] if ( self._classes[i] == pred[j] ) else 0.0

                for i in range( 0, self._n_classes ):
                    ydata = lines[ i ][ 0 ].get_ydata()
                    ydata = np.hstack( [ ydata, new_ydata[i,:] ] )
                    lines[ i ][ 0 ].set_ydata( ydata[-self._buffer_size:] )


            # small pause so we can update plot
            plt.pause( 0.005 )
        plt.close()
  
    def add(self, pred, speed, val):
        """
        Parameters
        ----------
        pred : str
            The predicted class (must be in self._classes)
        speed : float
            The velocity of the predicted class
        val : float
            The value for the bar plot
        """
        if pred in self._classes or pred == 'REST':
            try:
                speed = min( max( speed, 0.0 ), 1.0 )
                val = min( max( self._bar_min, val ), self._bar_max )
                self._queue.put( ( pred, speed, val ), timeout = 1e-3 )
            except queue.Full:
                pass

    def close( self ):
        """
        Stop the plotting subprocess while releasing subprocess resources
        """
        self._exit_event.set()
        while self._queue.qsize() > 0:
            try:
                self._queue.get( timeout = 1e-3 )
            except queue.Empty:
                pass
        self._plotter.join()
   
    @property
    def change(self):
        """
        Return the newest GUI parameters (if they've been updated since the last time this function was called)
        """
        if self._change_queue.qsize():
            try:
                param = self._change_queue.get( timeout = 1e-3 )
                return param
            except:
                pass
        return False
        
    @property
    def thresholds( self ):
        """
        The current threshold values for each movement class

        Returns
        -------
        numpy.ndarray (self._n_classes,)
            The threshold for each class        
        """
        if self._threshold_queue.qsize():
            try:
                self.current_thresh =  self._threshold_queue.get( timeout = 1e-3 )
            except queue.Empty:
                pass

        if self._percentages_queue.qsize():
            try:
                self.current_percentages =  self._percentages_queue.get( timeout = 1e-3 )
            except queue.Empty:
                pass
        try:
            if self.classifier == 'Spatial':
                return self.current_percentages
            else:
                return self.current_thresh
        except:
            if self.classifier == 'Spatial':
                return self._percentages
            else:
                return self._thresh


if __name__ == '__main__':
    import time

    CLASSES = ('OPEN', 'POWER', 'PALM UP', 'PALM DOWN', 'TRIPOD')
    N_SAMPLES_PER_CLASS = 100
    
    data = np.hstack( [ np.abs( np.sin( np.linspace( 0.0, 2 * np.pi, N_SAMPLES_PER_CLASS ) ) ) for _ in range( len( CLASSES ) ) ] )
    pred = [ x for x in CLASSES for _ in range( N_SAMPLES_PER_CLASS ) ]
    
    glide = RealTimeGlidePlot( classes = CLASSES )

    for i in range( data.shape[0] ):
        glide.add( pred[i], data[i], data[i] )
        time.sleep( 5.0 / N_SAMPLES_PER_CLASS )
        if glide.change:
            print(glide.thresholds)

    glide.close()