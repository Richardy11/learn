import os
import multiprocessing as mp
import queue

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class CueVisualizer:
    """ A Python implementation of a grip cue visualizer """
    def __init__(self, cue_path):
        """
        Constructor

        Parameters
        ----------
        cue_path : str
            The path to the folder holding all of the cues you'd like to visualize

        Returns
        -------
        obj
            A CueVisualizer interface object
        """
        self._fig = None
        self._ax = None
        self._cue_images = {}

        for f in os.listdir( cue_path ):
            if f.endswith( '.png' ):
                img = mpimg.imread( os.path.join( cue_path, f ) )
                self._cue_images.update( { os.path.splitext( f )[0] : img } )    # remove extension from filename to use as key

        self._queue = mp.Queue()
        self._exit_event = mp.Event()
        self._visualizer = mp.Process( target = self._show )

        self._visualizer.start()

    def __del__(self):
        self.close()

    def _show( self ):
        while not self._exit_event.is_set():
            msg = None
            while self._queue.qsize() > 0:
                try:
                    msg = self._queue.get( timeout = 1e-3 )
                except queue.Empty: pass

            if msg is not None:
                #print( msg )
                if self._fig is None or not plt.fignum_exists(self._fig.number):
                    self._fig = plt.figure()
                    self._ax = self._fig.add_subplot(111)
                    self._fig.canvas.set_window_title('Cue Visualizer')
                    plt.axis('off')
                    plt.show(block=False)
                try:
                    self._ax.imshow(self._cue_images[msg])
                    # plt.imshow( self._cue_images[msg] )
                    # self._fig.canvas.draw()
                    # self._fig.canvas.flush_events()
                    # plt.pause(0.01)
                except KeyError: pass
                plt.pause( 0.05 )

    def publish( self, msg ):
        """
        Publish commanded grip cue to an image figure

        Parameters
        ----------
        msg : str
            The name of the cue to send
        """
        self._queue.put( msg )
        # if self._fig is None or not plt.fignum_exists(self._fig.number):
        #     self._fig = plt.figure()
        #     self._ax = self._fig.add_subplot( 111 )
        #     self._fig.canvas.set_window_title('Cue Visualizer')
        #     plt.axis( 'off' )
        #     plt.show( block = False )
        # try:
        #     self._ax.imshow( self._cue_images[msg] )
        #     # plt.imshow( self._cue_images[msg] )
        #     # self._fig.canvas.draw()
        #     # self._fig.canvas.flush_events()
        #     plt.pause( 0.01 )
        # except KeyError:
        #     raise RuntimeWarning( 'Invalid cue!', msg )

    def close(self):
        while self._queue.qsize() > 0:
            try:
                msg = self._queue.get(timeout=1e-3)
            except queue.Empty: pass
        self._exit_event.set()

if __name__ == '__main__':
    pass
    # cue = CueVisualizer()
    # moves = [ 'rest', 'open', 'power', 'pronate', 'supinate', 'tripod', 'index' ]
    #
    # print( '----- Movement Commands -----' )
    # print( '| 00  -----  REST           |' )
    # print( '| 01  -----  OPEN           |' )
    # print( '| 02  -----  POWER          |' )
    # print( '| 03  -----  PRONATE        |' )
    # print( '| 04  -----  SUPINATE       |' )
    # print( '| 05  -----  TRIPOD         |' )
    # print( '| 06  -----  INDEX POINT    |' )
    # print( '-----------------------------' )
    # print( '| Press [Q] to quit!        |' )
    # print( '-----------------------------' )
    #
    # done = False
    # while not done:
    #     cmd = input( 'Command: ' )
    #     if cmd.lower() == 'q':
    #         done = True
    #     else:
    #         try:
    #             idx = int( cmd )
    #             if idx in range( 0, len( moves ) ):
    #                 cue.publish( moves[ idx ] )
    #         except ValueError:
    #             pass
    # print( 'Bye-bye!' )