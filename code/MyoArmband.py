import sys
import warnings
warnings.simplefilter("ignore", UserWarning)
sys.coinit_flags = 2

import queue
import struct
import asyncio
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt

from bleak import BleakClient
from bleak import BleakError

import Quaternion as quat

class MyoArmband:
    """ A Python implementation of a Myo armband interface """
    MYOHW_CMD_CHAR  = 'd5060401-a904-deb9-4748-2c7f4a124842'
    MYOHW_IMU_CHAR  = 'd5060402-a904-deb9-4748-2c7f4a124842'
    MYOHW_EMG0_CHAR = 'd5060105-a904-deb9-4748-2c7f4a124842'
    MYOHW_EMG1_CHAR = 'd5060205-a904-deb9-4748-2c7f4a124842'
    MYOHW_EMG2_CHAR = 'd5060305-a904-deb9-4748-2c7f4a124842'
    MYOHW_EMG3_CHAR = 'd5060405-a904-deb9-4748-2c7f4a124842'
    MYOHW_CLS_CHAR  = 'd5060103-a904-deb9-4748-2c7f4a124842'

    MYOHW_DATA_MODE = bytearray( b'\x01\x03\x02\x01\x01' )

    MYOHW_ORIENTATION_SCALE = 16384.0
    MYOHW_ACCELEROMETER_SCALE = 2048.0
    MYOHW_GYROSCOPE_SCALE = 16.0

    def __init__(self, name = 'MyoArmband', mac = 'c9:7e:78:c7:09:da'):
        """
        Constructor

        Parameters
        ----------
        name : str
            The unique device handle used to refer to the connected hardware
        mac : str
            The MAC address for the BLE radio

        Returns
        -------
        obj
            A Myo armband interface object
        """
        # device variables
        self._name = name
        self._mac = mac
        self._speriod = 1.0 / 200.0
        self._channelcount = 13     # 8 EMG + 4 IMU + 1 CLS

        # state variables
        self._state = np.zeros( self._channelcount, )
        self._state[8] = 1.0

        self._state_buffer = mp.Queue()
        self._state_buffer_max = int( 5 * 200.0 ) # 5 x sampling rate

        # streaming variables
        self._connect_event = mp.Event()
        self._exit_event = mp.Event()

        self._stream_event = mp.Event()
        self._print_event = mp.Event()
        self._streamer = mp.Process( target = self._connect )

        # viewing variables
        self._view_event = mp.Event()
        self._plot_buffer = mp.Queue()
        self._viewer = None

        # connect synchronization
        self._streamer.start()
        self._connect_event.wait()

    def _connect(self):
        """
        Synchronous wrapper to connect to BTLE device
        """
        loop = asyncio.get_event_loop()
        loop.run_until_complete( self._stream( loop ) )
        loop.close()

        # clean processing queues so we can join
        self.flush()
        empty = False
        while not empty:
            try: self._plot_buffer.get( timeout = 1e-3 )
            except queue.Empty: empty = True

    async def _stream(self, loop):
        """
        Asynchronous streaming function

        Parameters
        ----------
        loop : asyncio.windows_events._WindowsSelectorEventLoop
            The event loop that this code should run in
        """
        for i in range( 0, 5 ):
            retry = False
            try:
                async with BleakClient( self._mac, loop = loop, timeout = 30 ) as client:
                    # enable data streams
                    await client.write_gatt_char( MyoArmband.MYOHW_CMD_CHAR, MyoArmband.MYOHW_DATA_MODE, response = False )

                    # subscribe to notifications
                    await client.start_notify( MyoArmband.MYOHW_IMU_CHAR, self._imu_callback )
                    await client.start_notify( MyoArmband.MYOHW_EMG0_CHAR, self._emg_callback )
                    await client.start_notify( MyoArmband.MYOHW_EMG1_CHAR, self._emg_callback )
                    await client.start_notify( MyoArmband.MYOHW_EMG2_CHAR, self._emg_callback )
                    await client.start_notify( MyoArmband.MYOHW_EMG3_CHAR, self._emg_callback )
                    await client.start_notify( MyoArmband.MYOHW_CLS_CHAR, self._cls_callback )

                    self._connect_event.set()
                    while not self._exit_event.is_set():
                        await asyncio.sleep( 1, loop = loop )
                    return

            except BleakError:
                print( "\nRetrying connection..." )
                retry = True

            if not retry:
                break   
            
        self._connect_event.set()
        self._exit_event.set()

    def _imu_callback(self, uuid, data):
        """
        Callback function for IMU characteristic

        Parameters
        ----------
        uuid : str
            The UUID of the characteristic sending notifications
        data : bytearray
            The data in the characteristic
        """
        imu = np.array( struct.unpack( '<10h', bytes( data ) ) )[:4] / MyoArmband.MYOHW_ORIENTATION_SCALE
        self._state[8:12] = imu

    def _emg_callback(self, uuid, data):
        """
        Callback function EMG characteristic

        Parameters
        ----------
        uuid : str
            The UUID of the characteristic sending notifications
        data : bytearray
            The data in the characteristic
        """
        emg = np.array( struct.unpack( '<16b', bytes( data ) ) )
        
        for i in range( 0, 2 ):
            if i == 0: self._state[:8] = emg[:8]
            else: self._state[:8] = emg[8:]

            while self._state_buffer.qsize() > self._state_buffer_max:
                self._state_buffer.get( timeout = 1e-3 )
            self._state_buffer.put( self._state.copy(), timeout = 1e-3 )

            if self._print_event.is_set(): print( self._name, ':', self._state )
            if self._view_event.is_set(): self._plot_buffer.put( self._state )

    def _cls_callback(self, uuid, data):
        """
        Callback function for classifier characteristic

        Parameters
        ----------
        uuid : str
            The UUID of the characteristic sending notifications
        data : bytearray
            The data in the characteristic
        """
        pass
        # self._state[-1] = int( data )
        # print( uuid, data )

    def _plot(self):
        """
        Generates a visualization of the incoming data in real-time
        """
        gui = plt.figure()
        gui.canvas.set_window_title( self._name )
        
        # orientation plots
        orientations = []
        orient_colors = 'rgb'
        orient_titles = [ 'ROLL', 'PITCH', 'YAW' ]
        for i in range( 0, 3 ):
            ax = gui.add_subplot( 2, 3, i + 1, 
                 projection = 'polar', aspect = 'equal' )
            ax.plot( np.linspace( 0, 2*np.pi, 100 ), 
                     np.ones( 100 ), color = orient_colors[ i ],
                     linewidth = 2.0 )
            orientations.append( ax.plot( np.zeros( 2 ), np.linspace( 0, 1, 2 ), 
                                         color = orient_colors[ i ], linewidth = 2.0  ) )
            ax.set_rticks( [] )
            ax.set_rmax( 1 )
            ax.set_xlabel( orient_titles[ i ] )
            ax.grid( True )

        # line plot
        emg_plots = []
        emg_offsets = np.array( [ 128, 384, 640, 896, 1152, 1408, 1664, 1920 ] )

        ax = gui.add_subplot( 2, 1, 2 )
        num_emg_samps = int( 5 * np.round( 1.0 / self._speriod ) )
        for i in range( 0, 8 ):
            xdata = np.linspace( 0, 1, num_emg_samps )
            ydata = emg_offsets[ i ] * np.ones( num_emg_samps )
            emg_plots.append( ax.plot( xdata, ydata ) )
        
        ax.set_ylim( 0, 2048 )
        ax.set_xlim( 0, 1 )
        ax.set_yticks( emg_offsets.tolist() )
        ax.set_yticklabels( [ 'EMG01', 'EMG02', 'EMG03', 'EMG04', 'EMG05', 'EMG06', 'EMG07', 'EMG08' ] )
        ax.set_xticks( [] )  

        plt.tight_layout()
        plt.show( block = False )
        while self._view_event.is_set():
            try:
                data = []
                while self._plot_buffer.qsize() > 0: data.append( self._plot_buffer.get() )
                if data:
                    # concate to get a block of data
                    data = np.vstack( data )

                    # update orientation data
                    angles = quat.to_euler( data[-1, 8:12] )
                    for i in range( 0, 3 ):
                        tdata = np.ones( 2 ) * angles[ i ]
                        rdata = np.linspace( 0, 1, 2 )
                        orientations[ i ][ 0 ].set_data( tdata, rdata )
                
                    # update electrophysiological data
                    for i in range( 0, 8):
                        ydata = emg_plots[ i ][ 0 ].get_ydata()
                        ydata = np.append( ydata, data[ :, i ] + emg_offsets[ i ] )
                        ydata = ydata[-num_emg_samps:]
                        emg_plots[ i ][ 0 ].set_ydata( ydata )
                plt.pause( 0.005 )
            except: self._view_event.clear()
        plt.close( gui )

    @property
    def name(self):
        """
        The unique handle for the device
        
        Returns
        -------
        str
            The specified unique handle of this device interface
        """ 
        return self._name

    @property
    def state(self):
        """
        The current state for the device
        
        Returns
        -------
        numpy.ndarray (n_channels,)
            The last measured sensor readings
        """
        try:
            return self._state_buffer.get( timeout = 1e-3 )
        except queue.Empty:
            return None

    @property
    def speriod(self):
        """
        The sampling period for the device
        
        Returns
        -------
        float
            The sampling period of the device
        """ 
        return self._speriod

    @property
    def channelcount(self):
        """
        The channel count for the device

        Returns
        -------
        int
            The number of channels per sensor measurement
        """ 
        return self._channelcount

    def run(self, display = False):
        """ 
        Starts the acquisition process of the Myo armband
        
        Parameters
        ----------
        display : bool
            Flag determining whether sensor measurements should be printed to console (True) or not (False)
        """
        if not self._stream_event.is_set():
            if display: self._print_event.set()
            else: self._print_event.clear()
            self._stream_event.set()

    def flush(self):
        """
        Dispose of all previous data
        """
        empty = False
        while not empty:
            try: self._state_buffer.get( timeout = 1e-3 )
            except queue.Empty: empty = True

    def stop(self):
        """
        Stops the acquisition process of the Myo armband
        """
        if self._stream_event.is_set():
            self._stream_event.clear()
        while self._state_buffer.qsize() > 0: self._state_buffer.get()

    def close( self ):
        self._exit_event.set()

    def view(self):
        """ 
        Launches the GUI viewer of the Myo armband
        """
        if not self._view_event.is_set():
            self._view_event.set()
            self._viewer = mp.Process( target = self._plot )
            self._viewer.start()
            
    def hide(self):
        """
        Closes the GUI viewer of the Myo armband
        """
        if self._view_event.is_set():
            self._view_event.clear()
            self._viewer.join()

if __name__ == '__main__':
    import sys
    import inspect
    import argparse

    # helper function for booleans
    def str2bool( v ):
        if v.lower() in [ 'yes', 'true', 't', 'y', '1' ]: return True
        elif v.lower() in [ 'no', 'false', 'n', 'f', '0' ]: return False
        else: raise argparse.ArgumentTypeError( 'Boolean value expected!' )

    # parse commandline entries
    class_init = inspect.getargspec( MyoArmband.__init__ )
    arglist = class_init.args[1:]   # first item is always self
    defaults = class_init.defaults
    parser = argparse.ArgumentParser()
    for arg in range( 0, len( arglist ) ):
        try: tgt_type = type( defaults[ arg ][ 0 ] )
        except: tgt_type = type( defaults[ arg ] )
        if tgt_type is bool:
            parser.add_argument( '--' + arglist[ arg ], 
                                type = str2bool, nargs = '?',
                                action = 'store', dest = arglist[ arg ],
                                default = defaults[ arg ] )
        else:
            parser.add_argument( '--' + arglist[ arg ], 
                                type = tgt_type, nargs = '+',
                                action = 'store', dest = arglist[ arg ],
                                default = defaults[ arg ] )
    args = parser.parse_args()
    for arg in range( 0, len( arglist ) ):
        attr = getattr( args, arglist[ arg ] )
        if isinstance( attr, list ) and not isinstance( defaults[ arg ], list ):
            setattr( args, arglist[ arg ], attr[ 0 ]  )

    # create interface
    myo = MyoArmband( name = args.name, mac = args.mac )
    myo.run( display = False )
    myo.view()

    while True:
        state = myo.state
        if state is not None:
            print( state )