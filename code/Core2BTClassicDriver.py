import bluetooth

import time
import queue
import copy

import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import struct

CMD_CONNECT = 0x80
CMD_DISCONNECT = 0x81

class Core2BTClassicDriver:    
    
    def __init__(self, name = 'Sense', mac = 'EC:FE:7E:1D:C7:81', num_electrodes = 8, gain = 4, srate = 1000, mode = 'emg', device = 'bebionic_nogrips', wrist = True, elbow = True):
        # device variable
        self._mac = mac
        self._name = name
        self._channelcount = num_electrodes
        self._speriod = 1.0 / srate

        self.ack_timeout = 4

        self.test = True

        self._connection_event = mp.Event()
        self._exit_event = mp.Event()
        self._ack_event = mp.Event()
        self._flush_completed = mp.Event()
        self._connection_acked = mp.Event()

        self.timer = time.time()

        self._state_buffer = mp.Queue()
        self._transfer_buffer = mp.Queue()
        self._ACK_buffer = mp.Queue()

        self._streamer = mp.Process( target = self._connect )
        self._streamer.start()

    def check_ack_timeout(self, expected_ack):
        if time.time() - self.timer > self.ack_timeout:
            print('ACK timed out on command: ', expected_ack, 'after ',time.time() - self.timer, ' secs')
            self.timer = time.time()
            return True
        else:
            return False

    def ACKcheck(self, expected_ack):
        try:
            latest_ack = self._ACK_buffer.get(timeout = 1e-4)

        except: 
            return self.check_ack_timeout(expected_ack)

        if expected_ack in latest_ack or CMD_DISCONNECT in latest_ack:

            if expected_ack == CMD_CONNECT:
                self._connection_acked.set()

            del latest_ack[latest_ack.index(expected_ack)]
            
            try:
                self._ACK_buffer.put(latest_ack, timeout = 1e-4)
            except: pass

            self.timer = time.time()

            return True
        else: 
            return self.check_ack_timeout(expected_ack)

    def _connect(self):
        """
        Connect to specified Sense controller

        Notes
        -----
        This function is the target of the the child polling process
        """

        try:

            self._bt = bluetooth.BluetoothSocket( bluetooth.RFCOMM )
            self._bt.connect( ( self._mac, 1 ) )
            self._bt.setblocking( False )
            #self._bt.settimeout(0.001)

            #self._bt.send(bytes(bytearray(b'\xff\x02\x00\x81\x82')))

            self._bt.send(bytes(bytearray(b'\xff\x02\x00\x80\x81')))

            print('Connected to Core2 device over Serial Bluetooth')

            self._connection_event.set()

            t = time.time()

            excepted_ack = 0x80
            self.timer = time.time()

            while True:
                if self.ACKcheck(excepted_ack):
                    break

                try:
                    rx_data = self._bt.recv( 2048 )
                except:
                    continue

                try:
                    temp = self._state_buffer.get( timeout = 1e-3 )
                    temp.extend(rx_data)
                    rx_data = bytearray(temp).copy()
                except Exception as e: pass

                #print(rx_data)

                self._state_buffer.put( bytearray(rx_data).copy(), timeout = 1e-3 )

            excepted_ack = None
            self._ack_event.set()

            while self._connection_event.is_set() and not self._exit_event.is_set():

                if excepted_ack is not None:
                    if self.ACKcheck(excepted_ack):
                        #print('acked ', excepted_ack)
                        excepted_ack = None
                        self._ack_event.set()

                if self._ack_event.is_set() and self._connection_acked.is_set():
                    try:
                        data = self._transfer_buffer.get(timeout = 1e-4)
                        #print(data)
                        
                        if len(data) > 0:
                            tx_data = copy.deepcopy(data[0])
                            self._transfer_buffer.put(data[1:], timeout = 1e-4)
                    except:
                        tx_data = None
                    
                    if tx_data is not None:

                        self._bt.send(bytes(tx_data))
                        excepted_ack = tx_data[3]
                        #print('expected ack', excepted_ack)
                        if excepted_ack == 129:
                            excepted_ack == None
                            #self._bt.settimeout(0.001)

                        self._ack_event.clear()
                        tx_data = None

                try:
                    rx_data = self._bt.recv( 2048 )
                except:
                    continue

                try:
                    temp = self._state_buffer.get( timeout = 1e-3 )
                    temp.extend(rx_data)
                    rx_data = bytearray(temp).copy()
                except Exception as e: pass

                #print(rx_data)
                self._state_buffer.put( bytearray(rx_data).copy(), timeout = 1e-3 )
            
        except Exception as e:
            print(e)

        finally:
            self.flush()

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
            data = self._state_buffer.get( timeout = 1e-3 )
            return data
        except queue.Empty:
            return None
    
    def transfer(self, data):
        """
        The current state for the device
        
        Returns
        -------
        numpy.ndarray (n_channels,)
            The last measured sensor readings
        """
        self._transfer_buffer.put( [data], timeout = 1e-3 )

    def ack(self, data):
        """
        The current state for the device
        
        Returns
        -------
        numpy.ndarray (n_channels,)
            The last measured sensor readings
        """
        self._ACK_buffer.put( data, timeout = 1e-3 )

    @property
    def connection(self):
        """
        The current state for the device
        
        Returns
        -------
        numpy.ndarray (n_channels,)
            The last measured sensor readings
        """
        if self._connection_event.is_set():
            return True
        else:
            return False

    @property
    def connection_acked(self):
        """
        The current state for the device
        
        Returns
        -------
        numpy.ndarray (n_channels,)
            The last measured sensor readings
        """
        if self._connection_acked.is_set():
            return True
        else:
            return False

    @property
    def ack_ready(self):
        """
        The current state for the device
        
        Returns
        -------
        numpy.ndarray (n_channels,)
            The last measured sensor readings
        """
        if self._ack_event.is_set():
            return True
        else:
            return False
    
    @property
    def exit(self):
        """
        The current state for the device
        
        Returns
        -------
        numpy.ndarray (n_channels,)
            The last measured sensor readings
        """
        if self._exit_event.is_set():
            return True
        else:
            return False

    def flush(self):
        """
        Dispose of all previous data
        """
        self._flush_completed.clear()

        empty = False
        while not empty:
            try: self._state_buffer.get( timeout = 1e-3 )
            except queue.Empty: empty = True
        empty = False
        while not empty:
            try: self._transfer_buffer.get( timeout = 1e-3 )
            except queue.Empty: empty = True
        empty = False
        while not empty:
            try: self._ACK_buffer.get( timeout = 1e-3 )
            except queue.Empty: empty = True
        
        self._flush_completed.set()

    @property
    def flush_completed(self):

        if self._flush_completed.is_set():
            return True
        else:
            return False

    def close(self):
        self._exit_event.set()
        time.sleep(0.01)
        t = time.time()
        self._connection_event.clear()
        self._streamer.join()
        print('Core2 Driver process terminated in ', time.time()-t,'s')

if __name__ == '__main__':

    core2driver = Core2BTClassicDriver()