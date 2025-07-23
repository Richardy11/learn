import asyncio
from socket import timeout
import numpy as np
from bleak import BleakScanner, BleakClient
import multiprocessing as mp
import queue
import time

import csv

timing = False
rec_time = 15

UUID_CHAR_BRSP_INFO = '99564A02-DC01-4D3C-B04E-3BB1EF0571B2'
UUID_CHAR_BRSP_MODE = 'A87988B9-694C-479C-900E-95DFA6C00A24'
UUID_CHAR_BRSP_RX   = 'BF03260C-7205-4C25-AF43-93B1C299D159'
UUID_CHAR_BRSP_TX   = '18CDA784-4BD3-4370-85BB-BFED91EC86AF'
UUID_CHAR_BRSP_CTS  = '0A1934F5-24B8-4F13-9842-37BB167C6AFF'
UUID_CHAR_BRSP_RTS  = 'FDD6B4D3-046D-4330-BDEC-1FD0C90CB43B'

class Core2BLEDriver:
    def __init__(self, mac = 'EC:FE:7E:1D:C7:81', test = False):
        
        self.mac = mac
        '''self.data_stream = []
        self.file_opened = False'''

        self.test = test

        self._connection_event = mp.Event()
        self._exit_event = mp.Event()
        self._ack_event = mp.Event()
        self._ack_event.set()

        self.timer = time.time()

        self._state_buffer = mp.Queue()
        self._transfer_buffer = mp.Queue()

        self._streamer = mp.Process( target = self._connect )

        self._streamer.start()
    

    def _connect(self):
        """
        Synchronous wrapper to connect to BTLE device
        """
        self.loop = asyncio.new_event_loop()
        self.loop.run_until_complete( self.run() )
        self.loop.close()
        # clean processing queues so we can join
        self.flush()
        '''empty = False
        while not empty:
            try: self._plot_buffer.get( timeout = 1e-3 )
            except queue.Empty: empty = True'''

    def notification_handler(self, sender, data):

        '''try:
            if data[3] == 0:
                self._ack_event.set()
        except:
            pass'''
        self._state_buffer.put( data.copy(), timeout = 1e-3 )
        print(time.time()-self.timer)
        self.timer = time.time()
        


    async def run(self):
        async with BleakClient(self.mac, loop=self.loop, timeout = 20) as client:
            try:

                await client.write_gatt_char(UUID_CHAR_BRSP_MODE, bytearray(b'\x01') )
                #await client.write_gatt_char(UUID_CHAR_BRSP_RX, bytearray(b'\xff\x02\x00\x80\x81') )
                
                await client.start_notify(UUID_CHAR_BRSP_TX, self.notification_handler)
                print('notification started')

                self._connection_event.set()
                
                if self.test:
                    await client.write_gatt_char(UUID_CHAR_BRSP_RX, bytearray(b'\xff\x02\x00\x80\x81') )#bytearray(b'\xff\x02\x00\x80\x81') bytearray(b'\xff\x07\x00\x8c\x05\n\x00\x88\x13') )

                    '''while not self._ack_event.is_set():
                        await asyncio.sleep(0.001, loop=loop)'''

                    #await client.write_gatt_char(UUID_CHAR_BRSP_RX, bytearray(b'\xff\x07\x00\x8c\x05\x32\x00\xfa\x00\xc3') )

                    t = time.time()
                    while time.time() - t < 10:
                        await asyncio.sleep(0.001, loop=self.loop)

                else:
                    await client.write_gatt_char(UUID_CHAR_BRSP_RX, bytearray(b'\xff\x02\x00\x80\x81') )#bytearray(b'\xff\x02\x00\x80\x81') bytearray(b'\xff\x07\x00\x8c\x05\n\x00\x88\x13') )

                    while self._connection_event.is_set() and not self._exit_event.is_set():
                        await asyncio.sleep(0.0001, loop=self.loop)
                        if True: #self._ack_event.is_set():
                            try:
                                data = self._transfer_buffer.get(timeout = 1e-4)
                            except:
                                continue
                            
                            if len(data) > 0:

                                await client.write_gatt_char(UUID_CHAR_BRSP_RX, data )
                                self._ack_event = mp.Event()


            except Exception as e:
                print(e)
            
            finally:
                print('Disconnecting from Core2 device')
                await client.write_gatt_char(UUID_CHAR_BRSP_RX, bytearray(b'\xff\x02\x00\x81\x82') )
                await client.disconnect()
                print('Successfully disconnected from Core2 device')

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
    
    def transfer(self, data):
        """
        The current state for the device
        
        Returns
        -------
        numpy.ndarray (n_channels,)
            The last measured sensor readings
        """
        self._transfer_buffer.put( data, timeout = 1e-3 )

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
        empty = False
        while not empty:
            try: self._state_buffer.get( timeout = 1e-3 )
            except queue.Empty: empty = True
        empty = False
        while not empty:
            try: self._transfer_buffer.get( timeout = 1e-3 )
            except queue.Empty: empty = True

    def close(self):
        self._exit_event.set()
        time.sleep(0.01)
        t = time.time()
        self._connection_event.clear()
        self._streamer.join()
        print('Core2 Driver process terminated in ', time.time()-t,'s')

if __name__ == '__main__':

    core2driver = Core2BLEDriver(test = True) 
    '''loop = asyncio.get_event_loop()
    result = loop.run_until_complete(core2driver.run(address, loop))'''

