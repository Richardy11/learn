import queue
import multiprocessing as mp

import matplotlib
#matplotlib.use( 'QT5Agg' )

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
matplotlib.rcParams['font.size'] = 6
#matplotlib.rcParams['font.weight'] = 'bold'

import numpy as np

from global_parameters import global_parameters

class UpdateGUI:
    @staticmethod
    def labels2title( labels, classes ):
        unique_labels = np.unique(  labels )
        if len( unique_labels ) > 1 and 0 in unique_labels:
            unique_labels = unique_labels[unique_labels != 0]

        title = [ classes[ int( label ) ] for label in unique_labels ]
        return ', '.join( title )

    def __init__(self, classes = ('REST', 'OPEN', 'POWER', 'PRONATE', 'SUPINATE', 'TRIPOD', 'KEY'), all_classes = ('REST', 'OPEN', 'POWER', 'PRONATE', 'SUPINATE', 'TRIPOD', 'KEY'), n_segments = 5):
        self._classes = tuple( classes )
        self._all_classes = all_classes
        self._n_buttons = 5

        # data variables
        self._segments = [ None ] * self._n_buttons
        self._cache = None

        # gui variables
        self._btn_idx = None
        self._rax_idx = 0

        global_params = global_parameters()
        self.palette = global_params.palette

        # multiprocessing variables
        self._in_buffer = mp.Queue()
        self._out_buffer = mp.Queue()

        self._exit_event = mp.Event()

        self._process = mp.Process( target = self._gui )
        self._process.start()

    def _gui(self):
        fig = plt.figure( figsize = (5, 3) )

        # put figure in top left corner
        mngr = plt.get_current_fig_manager()
        geom = mngr.window.geometry()
        x, y, dx, dy = geom.getRect()
        mngr.window.setGeometry( 5, 35, 500, 400 )
        mngr.window.setWindowTitle( "RESCU Update GUI" )

        # create GUI layout
        rows = self._n_buttons*2-1
        widths = [1, 50, 50]
        heights = [1.5 if i%2 == 1 else 3 for i in range(rows)]
        gs = fig.add_gridspec( rows, 3, width_ratios=widths, height_ratios=heights, hspace = 0.2, wspace = 0.1 )
        rest_ax, ax, btn, rest_indicators = [], [], [], []

        for i in range( 0, rows ):
            # define plot axis
            if i%2 == 0:
                ax.append( fig.add_subplot( gs[i, 1] ) )
            else:
                rest_ax.append( fig.add_subplot( gs[i, 1] ) )

            # add title
            if i == 0: ax[-1].set_title( "Please select segment to update." )

            # create buttons
            # define callback

            # create button and bind callback
            if i%2 == 0:
                btn.append( Button( ax[int(i/2)], '', color = 'white', hovercolor = 'white' ) )
                for axis in ['top', 'bottom', 'left', 'right' ]:
                    ax[int(i/2)].spines[axis].set_linewidth( 0 )
            else:
                rest_indicators.append( Button( rest_ax[int(i/2)], '', color = 'white', hovercolor = 'white' ) )
                for axis in ['top', 'bottom', 'left', 'right' ]:
                    rest_ax[int(i/2)].spines[axis].set_linewidth( 0 )


        '''# define exit callback
        def continue_callback( event ):
            try:
                self._out_buffer.put( -1, timeout = 1e-3 )
            except queue.Full:
                pass
        btn.append( Button( ax[-1], 'Continue' ) )
        btn[-1].on_clicked( continue_callback )
        btn[-1].color = 'orange'
        btn[-1].hovercolor = 'orange' '''

        # generate radio buttons
        col2_idx = int( np.ceil(rows/2) )
        rax = fig.add_subplot( gs[:col2_idx,2] )
        rax.set_title( 'Please identify the correct label.' )
        radio = RadioButtons( rax, self._classes, activecolor = 'green' )

        def radio_callback( event ):
            label = radio.value_selected
            self._rax_idx = self._classes.index( label )
        radio.on_clicked( radio_callback )

        # undo return button
        def undo_callback( event ):
            try:
                self._out_buffer.put( [-3], timeout = 1e-3 )
            except queue.Full:
                pass

        uax = fig.add_subplot( gs[-3:-2,2] )
        undo = Button( uax, 'Undo One Update')
        undo.on_clicked( undo_callback )


        # generate update button
        def update_callback( event ):
            if self._btn_idx is not None:
                # generate re-labeled features
                feat = self._segments[self._btn_idx][0]
                label = int( self._rax_idx )
                try:
                    self._out_buffer.put( ( feat, label, self._btn_idx ), timeout = 1e-3 )

                    # update button color
                    btn[self._btn_idx].color = 'yellow'
                    btn[self._btn_idx].hovercolor = 'yellow'

                    # reset button thickness (but leave color)
                    for axis in [ 'top', 'bottom', 'left', 'right' ]:
                        ax[self._btn_idx ].spines[ axis ].set_linewidth( 0.5 )
                    
                    # reset radio
                    radio.set_active( 0 )

                    # reset chosen values
                    self._btn_idx = None
                    self._rax_idx = 0
                except queue.Full:
                    pass

        uax = fig.add_subplot( gs[ -1:, 2] )
        update = Button( uax, 'Update Segment' )
        update.on_clicked( update_callback )

        line_ax = []
        last_label = 0
        # logic loop
        plt.show( block = False )

        def remove_border(ax):
            for axis in [ 'top', 'bottom', 'left', 'right' ]:
                ax.spines[ axis ].set_visible( False )
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)

        while not self._exit_event.is_set():
            try:
                packets = self._in_buffer.get( timeout = 1e-3 )

                if len(packets) == 0:
                    continue

                to_add = []
                for p in reversed(packets):
                    to_add.append(p)

                self._segments = to_add + self._segments[:-len(packets)] # newest on top
                for l in range(len(line_ax)):
                    fig.delaxes(line_ax[l])
                line_ax = []

                '''for b in range(len(ax)):
                    fig.delaxes(ax[b])
                ax = []

                for i in range( 0, rows ):
                    # define plot axis
                    if i%2 == 0:
                        ax.append( fig.add_subplot( gs[i, 1] ) )'''

                for i in reversed(range( self._n_buttons )):
                    if self._segments[i] is not None:

                        def inner_segment_callback( event, idx = i ):
                            if btn[idx].color != 'red': # if a valid button
                                # change line thickness to indicate chosen button
                                for j in range( 0, self._n_buttons ):
                                    for axis in ['top', 'bottom', 'left', 'right' ]:
                                        ax[j].spines[axis].set_linewidth( 2 if j == idx else 1 )
                                self._btn_idx = idx

                                feat = self._segments[self._btn_idx][0]
                                label = int( self._rax_idx )
                                try:
                                    self._out_buffer.put( ( -2, feat, label, self._btn_idx ), timeout = 1e-3 )
                                except queue.Full:
                                    pass

                        # update button color
                        ax[i].clear()
                        if i == 0: ax[i].set_title( "Please select segment to update." )
                        current_label = self._classes[ int( self._segments[i][1][0] ) ]  + ' ' + str(self._segments[i][3]/1000) + 's '

                        if self._segments[i][5][0] == 1: # or int( self._segments[i][1][0] ) != last_label:
                            btn[i] = Button(    ax[i], 
                                                label = current_label, 
                                                color = self.palette[ self._all_classes.index( self._classes[ int( self._segments[i][1][0] ) ] ) ], 
                                                hovercolor = self.palette[ self._all_classes.index( self._classes[ int( self._segments[i][1][0] ) ] ) ] )
                            btn[i].on_clicked( inner_segment_callback )
                            
                            '''if self._segments[i][5][0] != 1:
                                last_label = int( self._segments[i][1][0] )'''
                        else:
                            btn[i] = Button(    ax[i], 
                                                label = current_label, 
                                                color = 'lightgrey', 
                                                hovercolor = 'lightgrey' )

                        if i < self._n_buttons-1:
                            rest_ax[i].clear()
                            if self._segments[i][5][0] == 1:
                                # update button color
                                rest_indicators[i] = Button(rest_ax[i], 'REST ' +  str(self._segments[i][4]/1000) + 's ', color = 'lightgrey', hovercolor = 'lightgrey')
                            else:
                                rest_indicators[i] = Button(rest_ax[i], '', color = 'white', hovercolor = 'white')

                        if self._segments[i][5][1] > 1 and self._segments[i][5][0] == self._segments[i][5][1]:
                            end_line = (i*2+(self._segments[i][5][1]*2-1))

                            if end_line < rows:
                                line_ax.append(fig.add_subplot( gs[ i*2: end_line ,0] ))
                            else:
                                line_ax.append(fig.add_subplot( gs[ i*2: rows , 0] ))

                            line_ax[-1].set(xlim=(-0.5,0.1))
                            line_ax[-1].axvline(x=0, c = self.palette[ self._all_classes.index( self._classes[ int( self._segments[i][1][0] ) ] ) ] )
                            remove_border(line_ax[-1])
                        elif self._segments[i][5][1] == 1:
                            line_ax.append(fig.add_subplot( gs[ i*2 ,0] ))

                            line_ax[-1].set(xlim=(-0.5,0.1))
                            line_ax[-1].axvline(x=0, c = self.palette[ self._all_classes.index( self._classes[ int( self._segments[i][1][0] ) ] ) ] )
                            remove_border(line_ax[-1])

                        # update button outline (color / thickness)
                        for axis in [ 'top', 'bottom', 'left', 'right' ]:
                            ax[i].spines[ axis ].set_linewidth( 2 if i == 0 else ax[i-1].spines[ axis ].get_linewidth() )
                            if self._segments[i][2] == -1:
                                ax[i].spines[ axis ].set_color( 'red' )
                            elif self._segments[i][2] == 1:
                                ax[i].spines[ axis ].set_color( 'blue' )
                            elif self._segments[i][2] == 0:
                                ax[i].spines[ axis ].set_color( 'black' )

                # update chosen button (if chosen)
                if self._btn_idx is not None:
                    self._btn_idx += 1
                    if self._btn_idx == self._n_buttons:
                        self._btn_idx = None

                # redraw the canvas
                fig.canvas.draw()
            except queue.Empty:
                pass
            plt.pause( 0.005 )

    def close(self):
        """
        Close the GUI, freeing all resources allocated
        """
        if self._process.is_alive:
            self._exit_event.set()

        # empty in buffer
        while True:
            try:
                self._in_buffer.get( timeout = 1e-3 )
            except queue.Empty:
                break

        # empty out buffer
        while True:
            try:
                self._out_buffer.get( timeout = 1e-3 )
            except queue.Empty:
                break

    def add(self, packets = []):
        """
        Add a segment and the corresponding labels to the update gui

        Parameters
        ----------
        feat : numpy.ndarray (n_samples, n_features)
            The feature vectors of the segment
        labels : numpy.ndarray (n_labels,)
            The classifier labels of the feature vectors
        status : int { -1, 0, 1 }
            The status of this segment. Do not use for updating (-1). Should be updated (1). Neutral (0)
            This status is ignored if the segment is set to cache.
        no_cache : bool
            True if data is from a normal segment, False if from inter-segment rest data
        """
        try:
            self._in_buffer.put( ( packets ), timeout = 1e-3 )
        except queue.Full:
            pass

    '''def add(self, feat, labels, status = 0, no_cache = True, length = 0, rest_length = 0, order = []):
        """
        Add a segment and the corresponding labels to the update gui

        Parameters
        ----------
        feat : numpy.ndarray (n_samples, n_features)
            The feature vectors of the segment
        labels : numpy.ndarray (n_labels,)
            The classifier labels of the feature vectors
        status : int { -1, 0, 1 }
            The status of this segment. Do not use for updating (-1). Should be updated (1). Neutral (0)
            This status is ignored if the segment is set to cache.
        no_cache : bool
            True if data is from a normal segment, False if from inter-segment rest data
        """
        try:
            self._in_buffer.put( ( feat, labels, status, no_cache, length, rest_length, order ), timeout = 1e-3 )
        except queue.Full:
            pass'''

    @property
    def state(self):
        """
        Returns
        -------
        tuple [numpy.ndarray (n_samples, n_features), int]
            A newly labeled segment of features
        """
        if self._out_buffer.qsize() > 0:
            try:
                return self._out_buffer.get( timeout = 1e-3 )
            except queue.Empty:
                return None

if __name__ == '__main__':
    import time

    N_SEGMENTS = 5
    CLASSES = ('REST', 'OPEN', 'POWER', 'PRONATE', 'SUPINATE', 'TRIPOD', 'KEY')

    gui = UpdateGUI( classes = CLASSES, n_segments = N_SEGMENTS )

    try:
        t0 = time.perf_counter()
        t1 = time.perf_counter()
        order_length = 1
        order_current = 1
        while time.perf_counter() - t0 < 60:
            class_idx = int( len( CLASSES ) * np.random.random() )
            if time.perf_counter() - t1 > 1:
                feat = np.random.random( size = ( int( np.random.rand() * 500 ), 288 ) )
                labels = class_idx * np.ones( shape = ( feat.shape[0] ) )
                status = np.random.randint( -1, 2 )

                gui.add( [feat, labels, status, np.random.randint( 0, 2000 ), np.random.randint( 0, 10000 ), [order_current, order_length]] )

                if order_current == order_length:
                    order_length = np.random.randint( 1, 4 )
                    order_current = 1
                else:
                    order_current += 1

                #print( [order_current, order_length] )
                t1 = time.perf_counter()

            relabel = gui.state
            if relabel is not None:
                print( [order_current, order_length] )
    finally:
        gui.close()


# NOTE: Blue button outline if a segment should be updated!!
#       Red button outline if a segment should not be used for an update!! This takes priority.
#       in add(), add a boolean flag to determine how to process data (True = normal, False = rest cache)
#           add Update Rest button to follow this functionality