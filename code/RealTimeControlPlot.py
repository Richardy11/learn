import queue
import multiprocessing as mp

import numpy as np

import matplotlib
matplotlib.use( "QT5Agg" )
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

class RealTimeControlPlot:
    def __init__(self, classes = ('rest', 'open', 'power', 'pronate', 'supinate', 'tripod'), num_channels = 1, sampling_rate = 1000, buffer_time = 5, title = None):
        """
        Constructor

        Parameters
        ----------
        classes : iterable of str
            The class labels for the control signal
        sampling_rate : float
            The expected sampling rate of the data stream
        buffer_time : float
            The amount of data in the plot at once (in sec)
        title : str or None
            The title of the plotting axis

        Returns
        -------
        obj
            A RealTimeControlPlot object
        """
        # data variables
        self._classes = classes
        self.show_raw = True
        self._num_channels = num_channels

        self._sampling_rate = sampling_rate
        self._buffer_time = buffer_time

        # plot visualizing variables
        self._title = title

        # multiprocessing variables
        self._queue = mp.Queue()
        self._exit_event = mp.Event()
        self._plotter = mp.Process( target = self._plot )

        # start plotting process
        self._plotter.start()

    def _plot( self ):
        """
        The plotting subprocess function
        """
        buffer_length = int( self._sampling_rate * self._buffer_time )

        fig = plt.figure()
        mngr = plt.get_current_fig_manager()
        geom = mngr.window.geometry()
        mngr.window.setWindowTitle( "Control Outputs" )

        ax = fig.add_subplot( 111 )

        mngr.window.setGeometry(1065,35,650, 400)
        
        xplot = np.linspace( 0, 1, buffer_length )
        yplot = np.zeros( shape = ( buffer_length, ) )
        
        lines = []
        for i in range( self._num_channels ):
            lines.append( ax.plot( xplot, yplot, linewidth = 2, zorder = -i, label = 'CTRL-%02d' % (i+1) ) )

        ax.set_xticks( [] )
        ax.set_ylim( bottom = -1, top = len( self._classes ) )
        ax.set_yticks( np.arange( 0, len( self._classes ) ) )
        ax.set_yticklabels( self._classes, rotation = 45 )
        ax.set_title( self._title if self._title is not None else "RealTime Plot" )
        ax.legend( loc = 'upper left' )

        sample = np.zeros( ( self._num_channels, ), dtype = int )

        ax = plt.axes( [ 0.10, 0.0, 0.3, 0.18 ] )
        fig.subplots_adjust( bottom = 0.2, left = 0.15 )
        chkbox = CheckButtons( ax, labels = [ "Show Raw Classification"], actives = [ True ] )
        for axis in [ 'top', 'bottom', 'left', 'right' ]:
            ax.spines[ axis ].set_visible( False )

        def toggle_visibility( label ):
            
            if self.show_raw:
                self.show_raw = False
                for i in range( 1, self._num_channels ):
                    lines[i][ 0 ].set_ydata( [0] * buffer_length )
                self._num_channels = 1

            else:
                self.show_raw = True
                self._num_channels = 2
            
        chkbox.on_clicked( toggle_visibility )

        while not self._exit_event.is_set():
            data = []

            # pull samples from queue
            while self._queue.qsize() > 0:
                ctrl = self._queue.get() # ctrl is a tuple of str
                for i in range( self._num_channels ):
                    sample[i] = self._classes.index( ctrl[i] )
                data.append( sample )
            
            # if we go valid samples
            if data:
                
                # concatenate data
                data = np.vstack( data )

                # update plots
                for i in range( self._num_channels ):
                    ydata = lines[i][ 0 ].get_ydata()
                    ydata = np.append( ydata, data[:,i] )
                    ydata = ydata[-buffer_length:]

                    lines[i][ 0 ].set_ydata( ydata )
            
            # small pause so we can update plot
            plt.pause( 0.005 )
        plt.close( fig )

    def add( self, ctrl ):
        """
        Add control signal to be plotted

        Parameters
        ----------
        ctrl : tuple of str
            The control signal sample to be added to the plotting queue
        """
        try:
            self._queue.put( ctrl, timeout = 1e-3 )
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

if __name__ == '__main__':
    import time

    N_CHANNELS = 3
    CLASSES = ('rest', 'open', 'power', 'pronate', 'supinate', 'tripod')
    N_SAMPLES_PER_CLASS = int( 125 // len( CLASSES ) )

    data = [ x for x in CLASSES for _ in range( N_SAMPLES_PER_CLASS ) ]
    data = np.array( data + data[::-1] )
     
    plot = RealTimeControlPlot(  classes = CLASSES, num_channels = N_CHANNELS, sampling_rate = 25, buffer_time = 5, title = "Random Control Signal" )

    n_samples = len( data )
    for i in range( n_samples ):
        ctrl = []
        for j in range( N_CHANNELS ):
            ctrl.append( np.roll( data, j * N_SAMPLES_PER_CLASS )[i] )

        plot.add( tuple( ctrl ) )
        time.sleep( 0.04 )
    plot.close()
