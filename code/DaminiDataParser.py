from Core2Controller import Core2Adapt
from time import time, sleep
import numpy as np
import matplotlib.pyplot as plt
import random

import queue
import struct
import asyncio
import multiprocessing as mp
import csv

from local_addresses import address

class Core2Stream:

    def __init__(self):

        self.local_source_info = address()

        #plot
        self._speriod = 1.0 / 200.0
        self._channelcount = 8
        self._name = 'Core2 Stream'
        # TODO CHANGE RANGE OF CHANNELS IN VISUALIZATION, WILL GO +/- self.offset
        self.offset = 10000

        # viewing variables
        self._view_event = mp.Event()
        self._view_exit_event = mp.Event()
        self._plot_buffer = mp.Queue()
        self._viewer = None

        self.file_opened = False

        core2 = Core2Adapt(address = self.local_source_info.Core2_mac)

        while not core2.connection:
            pass

        core2.set_electrodes()
        res = None
        t = time()
        while res is None and time() - t < 5:
            res = core2.get_electrodes()
        print("DATA VALIDATION GET_ENABLED", res)

        """
        CHANGE THE STEP YOU WANT TO PULL DATA FROM
        """
        core2.set_debug_emg_streaming( override=True, step=0 )

        """
        SET STREAM PROPERTIES
        """
        data_type = 0 # type of data
        frequency = 1 # millisecs/sample
        quantity = np.ceil(5000/frequency) # number of samples (don't need to change)

        samp_rate = np.floor(1000/frequency)
        msg_time = np.floor(quantity/samp_rate)

        runtime = 5 # in seconds

        # viewing variables
        self.view()

        self.data_stream = []
        
        t = 4
        r = 0
        rows = 0
        while not core2.check_ack(140):
            if time() - t > 3:
                
                if r > 0:
                    print('Retry stream start, attempt: ', r)
                r += 1

                core2.stream_data_start(data_type=data_type,frequency=frequency,quantity=quantity)
                t = time()
        print('Starting Stream with Data type: ', data_type, ', Frequency: ', samp_rate, ' Hz, Quantity: ', quantity)

        t = time()

        while time() - t < runtime:

            #print(time() - t)
            
            core2.stream_data_start(data_type=data_type, frequency=frequency, quantity=quantity)
            
            t2 = time()
            
            while time() - t2 < msg_time*0.9 and time() - t < runtime:
                data = core2.get_emg_data()
                if data is not None:
                    if self._view_event.is_set():
                        # DEBUG VISUALIZATION WITH RANDOM NUMBERS
                        #self._plot_buffer.put( np.array(data['DATA'] ) + random.randint(-self.offset,self.offset))
                        self._plot_buffer.put( np.array(data['DATA'] ) )     
            
                    self.data_stream.append( data['DATA'] ) 
                    rows += 1

        core2.cmd_stop_stream_data()
        
        t = time()
        while data is not None or time()-t < 1:
                data = core2.get_emg_data()
                if data is not None:
                    t = time()
                    if self._view_event.is_set():
                        # DEBUG VISUALIZATION WITH RANDOM NUMBERS
                        #self._plot_buffer.put( np.array(data['DATA'] ) + random.randint(-self.offset,self.offset))
                        self._plot_buffer.put( np.array(data['DATA'] ) )     
            
                    self.data_stream.append( data['DATA'] )
                    rows += 1
        
        print('rows', rows)
        # SAVE TO FILE
        f = open('./data/core2stream_.csv', 'w', newline='')
        writer = csv.writer(f)
        writer.writerows( self.data_stream )
        f.close()

        sleep(1)
        core2.close()
        self.close()

    def _plot(self):
        """
        Generates a visualization of the incoming data in real-time
        """
        gui = plt.figure()
        gui.canvas.set_window_title( self._name )
        
        # line plot
        emg_plots = []
        
        emg_offsets = np.array( [self.offset*i for i in range(1, 2*self._channelcount, 2)] )

        ax = gui.add_subplot( 1, 1, 1 )
        num_emg_samps = int( 5 * np.round( 1.0 / self._speriod ) )
        for i in range( 0, 8 ):
            xdata = np.linspace( 0, 1, num_emg_samps )
            ydata = emg_offsets[ i ] * np.ones( num_emg_samps )
            emg_plots.append( ax.plot( xdata, ydata ) )
        
        ax.set_ylim( 0, self.offset*self._channelcount*2 )
        ax.set_xlim( 0, 1 )
        ax.set_yticks( emg_offsets.tolist() )
        ax.set_yticklabels( [ 'EMG01', 'EMG02', 'EMG03', 'EMG04', 'EMG05', 'EMG06', 'EMG07', 'EMG08' ] )
        ax.set_xticks( [] )  

        plt.tight_layout()
        plt.show( block = False )
        while self._view_event.is_set() and not self._view_exit_event.is_set():
            try:
                data = []
                while self._plot_buffer.qsize() > 0: data.append( self._plot_buffer.get() )
                if data:
                    # concate to get a block of data
                    data = np.vstack( data )                
                    # update electrophysiological data
                    for i in range( 0, self._channelcount):
                        ydata = emg_plots[ i ][ 0 ].get_ydata()
                        ydata = np.append( ydata, data[ :, i ] + emg_offsets[ i ] )
                        ydata = ydata[-num_emg_samps:]
                        emg_plots[ i ][ 0 ].set_ydata( ydata )
                plt.pause( 0.005 )
            except: self._view_event.clear()
        plt.close( gui )

    def view(self):
        """ 
        Launches the GUI viewer of the Myo armband
        """
        if not self._view_event.is_set():
            self._view_event.set()
            self._viewer = mp.Process( target = self._plot )
            self._viewer.start()

    def flush(self):
        empty = False
        while not empty:
            try: self._plot_buffer.get( timeout = 1e-3 )
            except queue.Empty: empty = True

    def close(self):
        self._view_exit_event.set()
        self._viewer.join()

if __name__ == '__main__':
    core2stream = Core2Stream()

'''print(data)
self.data_stream.append(data)
if not self.file_opened:
    f = open('./data/core2stream_.csv', 'w')
    self.file_opened = True
else:
    f = open('./data/core2stream_.csv', 'a')
writer = csv.writer(f)
writer.writerow(data)
f.close()'''