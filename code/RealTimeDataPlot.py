import queue
import multiprocessing as mp

import numpy as np

import matplotlib
matplotlib.use( "QT5Agg" )
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
matplotlib.rcParams['font.size'] = 7

from matplotlib.widgets import CheckButtons

class RealTimeDataPlot:
    def __init__(self, num_channels = 8, channel_min = 0, channel_max = 1, sampling_rate = 1000, buffer_time = 5, title = None, classifier = 'EASRC', source = 'Sense'):
        """
        Constructor

        Parameters
        ----------
        num_channels : int
            The number of channels for a single sample
        channel_min : float
            The minimum value a sample can take
        channel_max : float
            The maximum value a sample can take
        sampling_rate : float
            The expected sampling rate of the data stream
        buffer_time : float
            The amount of data in the plot at once (in sec)
        title : str or None
            The title of the plotting axis

        Returns
        -------
        obj
            A RealTimeDataPlot object
        """
        # data variables
        self._num_channels = num_channels
        self._channel_min = channel_min
        self._channel_max = channel_max

        if classifier in ['Spatial', 'Simultaneous']:
            if source == 'Myo Band':
                self._channel_max_individual = 0.3/50
            elif source == 'Sense':
                self._channel_max_individual = 0.3
        else:
            self._channel_max_individual = channel_max

        self._classifier = classifier

        self._sampling_rate = sampling_rate
        self._buffer_time = buffer_time

        self._threshold = mp.Value( 'd', self._channel_min )

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

        fig = plt.figure( figsize = ( 9.0 , 3.0 ))
        mngr = plt.get_current_fig_manager()
        geom = mngr.window.geometry()
        mngr.window.setWindowTitle( "Signals" )
        x,y,dx,dy = geom.getRect()

        grid_rows = 1
        grid_columns = 20

        # create GUI layout
        gs = fig.add_gridspec( grid_rows, grid_columns )

        ax = fig.add_subplot( gs[ :, :7 ] )

        mngr.window.setGeometry(5,470,500, 550)

        label_format = '{:,.0000f}'
        
        lines = []
        yrange = ( self._channel_max_individual - self._channel_min )
        yoffsets = ( yrange * np.arange( 0, self._num_channels ) + yrange / 2 ).tolist()
        for i in range( self._num_channels ):
            xplot = np.linspace( 0, 1, buffer_length )
            yplot = np.zeros( shape = ( buffer_length, ) )
            
            lines.append( ax.plot( xplot, yplot + yoffsets[i], linewidth = 2 ) )

        ax.set_xticks( [] )
        ax.set_ylim( bottom = yoffsets[0] - yrange, top = yoffsets[-1] + yrange )
        ax.set_yticks( yoffsets )
        ax.set_yticklabels( [ "          CH%02d" % ( i + 1 ) for i in range( self._num_channels ) ], rotation = 45 )
        ax.set_title( self._title if self._title is not None else "RealTime Channels" )

        # plot of data average
        ax = fig.add_subplot( gs[ :, 8: ] )
        avg = ax.plot( np.linspace( 0, 1, buffer_length ), np.zeros( shape = ( buffer_length, ) ), zorder = -1 )


        # plot threshold
        eps_ydata = self._threshold.value * np.ones( shape = ( buffer_length, ) )
        eps = ax.plot( np.linspace( 0, 1, buffer_length ), eps_ydata, color = 'red', linestyle = '--', visible = True )

        ax.set_xticks( [] )
        ax.set_ylim( bottom = self._channel_min, top = self._channel_max )
        ticks_loc = ax.get_yticks().tolist()
        ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_yticklabels(range(len(ticks_loc)), rotation = 45 )
        ax.set_title( "Mean Channel Values" )

        # add check box to toggle threshold plot
        '''ax = plt.axes( [ 0.55, 0, 0.17, 0.18 ] )
        fig.subplots_adjust( bottom = 0.2 )

        for axis in [ 'top', 'bottom', 'left', 'right' ]:
            ax.spines[ axis ].set_visible( False )


        chkbox = CheckButtons( ax, labels = [ "Show Threshold"], actives = [ True ] )
        def toggle_visibility( label ):
            eps[0].set_visible( not eps[0].get_visible() )
        chkbox.on_clicked( toggle_visibility )'''

        fig.subplots_adjust( bottom = 0 )
        fig.tight_layout()
        while not self._exit_event.is_set():
            data = []

            # pull samples from queue
            while self._queue.qsize() > 0:
                sample = self._queue.get()
                
                if type(sample) == list:

                    distance = sample[1]

                    sample = sample[0]
                
                if len( sample.shape ) == 1: 
                    sample = np.expand_dims( sample, axis = 0 )

                if sample.shape[1] == self._num_channels:
                    data.append( sample )
            
            # if we go valid samples
            if data:
                # concatenate data
                data = np.vstack( data )

                # update plots
                for i in range( self._num_channels ):
                    ydata = lines[ i ][ 0 ].get_ydata()
                    ydata = np.append( ydata, data[ :, i ] + yoffsets[ i ] )
                    ydata = ydata[-buffer_length:]

                    lines[ i ][ 0 ].set_ydata( ydata )

                # update mean plot
                mean_data = avg[ 0 ].get_ydata()
                if self._classifier in ['Spatial', 'Simultaneous']:
                    mean_data = np.append( mean_data, distance )
                else:
                    mean_data = np.append( mean_data, np.mean( data, axis = 1 ) )
                mean_data = mean_data[-buffer_length:]
                
                avg[ 0 ].set_ydata( mean_data )

            # update threshold (if showing and its been updated)
            if eps[ 0 ].get_visible() and ( self._threshold.value != eps_ydata[0] ):
                eps_ydata = self._threshold.value * np.ones( shape = ( buffer_length, ) )
                eps[ 0 ].set_ydata( eps_ydata )
            
            # small pause so we can update plot
            plt.pause( 0.005 )
        plt.close( fig )

    def add( self, data ):
        """
        Add data to be plotted

        Parameters
        ----------
        data : numpy.ndarray (n_channels,)
            The data sample to be added to the plotting queue
        """
        try:
            self._queue.put( data, timeout = 1e-3 )
        except queue.Full:
            pass

    def set_threshold( self, eps ):
        """
        Set the threshold to highlight in the mean channel value subplot

        Parameters
        ----------
        eps : float
            The threshold value (must be in the range of [channel_min, channel_max])
        """
        if eps >= self._channel_min and eps <= self._channel_max:
            self._threshold.value = eps

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

    N_CHANNELS = 8
    N_SAMPLES = 10000

    data = np.random.rand( N_SAMPLES, N_CHANNELS )
    plot = RealTimeDataPlot( num_channels = N_CHANNELS, channel_min = 0, channel_max = 1/50, title = "Random Data" )

    for sample in data:
        plot.add( sample/50 )
        plot.set_threshold( np.mean( sample )/50 )
        time.sleep( 1e-3 )
    plot.close()
