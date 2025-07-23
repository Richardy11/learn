from Core2Controller import Core2Adapt
from time import time, sleep
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import pickle

import asyncio

import queue
import struct
import asyncio
import multiprocessing as mp
import csv

from local_addresses import address

Test_data = {   'Add_filter' : {
                                'UnanimousVoting' : { 'Default': 5, '0': 5, '1': 5, '2': 5},
                                'OnsetThreshold'  : { 'Threshold': 0.2, 'Latching' : True },
                                'ProportionalControl' :   {   'Reference_MMAV': 0.01, 
                                        'General': {'Min_Speed_Multiplier': 0.1, 'Max_Speed_Multiplier': 0.9, 'MMAV_ratio_lower': 0.1, 'MMAV_ratio_upper': 0.9},
                                        '0' : {'Min_Speed_Multiplier': 0.1, 'Max_Speed_Multiplier': 0.9, 'MMAV_ratio_lower': 0.1, 'MMAV_ratio_upper': 0.9}, 
                                        '1' : {'Min_Speed_Multiplier': 0.1, 'Max_Speed_Multiplier': 0.9, 'MMAV_ratio_lower': 0.1, 'MMAV_ratio_upper': 0.9}, 
                                        '2' : {'Min_Speed_Multiplier': 0.1, 'Max_Speed_Multiplier': 0.9, 'MMAV_ratio_lower': 0.1, 'MMAV_ratio_upper': 0.9}
                                    },
                                'VelocityRamp' : { 'Ramp_Length' : 50, 'Increment' : 1, 'Decrement' : 2,'Min_Speed_Multiplier' : 0.1, 'Max_Speed_Multiplier' : 0.9 }


                                },
                
                'Modify_filter' : {
                                'UnanimousVoting' : { 'Default': 5, '0': 5, '1': 10, '2': 7},
                                'OnsetThreshold'  : { 'Threshold': 0.5, 'Latching' : True },
                                'ProportionalControl' :   {   'Reference_MMAV': 0.02, 
                                        'General': {'Min_Speed_Multiplier': 0.2, 'Max_Speed_Multiplier': 0.9, 'MMAV_ratio_lower': 0.1, 'MMAV_ratio_upper': 0.9},
                                        '0' : {'Min_Speed_Multiplier': 0.2, 'Max_Speed_Multiplier': 0.9, 'MMAV_ratio_lower': 0.1, 'MMAV_ratio_upper': 0.9}, 
                                        '1' : {'Min_Speed_Multiplier': 0.2, 'Max_Speed_Multiplier': 0.9, 'MMAV_ratio_lower': 0.1, 'MMAV_ratio_upper': 0.9}, 
                                        '2' : {'Min_Speed_Multiplier': 0.2, 'Max_Speed_Multiplier': 0.9, 'MMAV_ratio_lower': 0.1, 'MMAV_ratio_upper': 0.9}
                                    },
                                'VelocityRamp' : { 'Ramp_Length' : 50, 'Increment' : 2, 'Decrement' : 2,'Min_Speed_Multiplier' : 0.1, 'Max_Speed_Multiplier' : 0.9 }


                                },

                'RLDA' :        {
                                'RLDA' : np.empty([0,0]),
                                'Means': np.empty([0,0])

                                },
                
                'CMD_DATA_MONITOR' :        {
                                'EDIT_CACHE' : [0, 5, 0.2, 5, 10],
                                'RUN_TEMPORARY_CACHE': 2200,
                                'VIEW_CACHE_SUMMARY': 1,
                                'VIEW_SEGMENT_DETAILS': [1,2200],
                                'GET_SEGMENT_FEATURE_VECTORS': [1,2200],
                                'REQUEST_EVENTS': [600,700,800,900]
                                }


            }

UTIs = {    'RESERVED': [0, 0], 
            'SimpleHand': [1,0], 
            'MotorDrivenWrist': [2,0], 
            'MotorDrivenElbow': [3,0], 
            'DacWrist': [4,0], 
            'DacElbow': [5,0],
            'Taska': [6,0], 
            'Espire Elbow': [7,0], 
            'Covvi Nexus': [8,0], 
            'I-limb Quantum': [9,0], 
            'Michelangelo': [10,0]
        }

UATs = {    'RESERVED': [0, 0], 
            'Rest': [1,0], 
            'BasicOpen': [2,0], 
            'BasicClose': [3,0], 
            'PronateWrist': [4,0], 
            'SupinateWrist': [5,0],
            'FlexWrist': [6,0], 
            'ExtendWrist': [7,0], 
            'BendElbow': [8,0], 
            'ExtendElbow': [9,0], 
            'AssumeGrip': [10,0], 
            'CloseInGrip': [11,0], 
            'FreeSwing': [12,0], 
            'EngageClutch': [13,0]
        }

class Core2Test:

    def __init__(self):

        self.local_source_info = address()

        #plot
        self._speriod = 1.0 / 200.0
        self._channelcount = 8
        self._name = 'Core2 Stream'
        self.offset = 1

        # viewing variables
        self._view_event = mp.Event()
        self._view_exit_event = mp.Event()
        self._plot_buffer = mp.Queue()
        self._viewer = None

        self.file_opened = False

        self.core2 = Core2Adapt(address = self.local_source_info.Core2_mac)

        while not self.core2.connection:
            pass#print('faszom')

        self.TestPackets()

        #self.TestStream()

        sleep(10)
        self.core2.close()
        self.close()

    def TestPackets(self):

        timeout = 5

        # Test RLDA
        '''for i in range(self.num_classes-1):
            try:
                Test_data['RLDA']['RLDA'] = np.vstack( (Test_data['RLDA']['RLDA'], np.ones(288)*( (i+1) * 0.1) ) )
            except:
                Test_data['RLDA']['RLDA'] = np.ones(288)*( (i+1) * 0.1 )

        for i in range(self.num_classes):
            try:
                Test_data['RLDA']['Means'] = np.vstack( (Test_data['RLDA']['Means'], np.ones(2)*( (i+1) * 0.1) ) )
            except:
                Test_data['RLDA']['Means'] = np.ones(2)*( (i+1) * 0.1)

        self.core2.replace_RLDA_all(Test_data['RLDA']['RLDA'], Test_data['RLDA']['Means'])

        res = None
        t = time.time()
        while res is None and time.time() - t < timeout:
            res = self.core2.get_rlda()
        print("DATA VALIDATION GET_ENABLED", res)

        self.core2.edit_RLDA_axis(axis_number=1, start_cluster=0, end_cluster=1, boundary=[0.1,0.9], class_indicies=[0,1,1], proportional_ramp_start=0, proportional_ramp_stop=1, min_proportional_speed=0.11, max_proportional_speed=0.92)'''
        


        self.core2.set_electrodes()
        res = None
        t = time.time()
        while res is None and time.time() - t < timeout:
            res = self.core2.get_electrodes()
        print("DATA VALIDATION GET_ENABLED", res)


        self.core2.set_gains( gains=[6,6,6], electrodes=[0,4,7] )
        #self.core2.set_gains( gains=[6,6,6], electrodes=[0,4,8])

        res = None
        t = time.time()
        while res is None and time.time() - t < timeout:
            res = self.core2.get_gains()
        print("DATA VALIDATION GET_GAINS", res)

        
        res = None
        self.core2.stream_filter_activity(255)
        t = time.time()
        while res is None and time.time() - t < timeout:
            res = self.core2.get_filter_activity()
        print("DATA VALIDATION FILTERING_HISTORY", res)

        res = None
        t = time.time()
        while res is None and time.time() - t < timeout:
            res = self.core2.request_time_data()
        print("DATA VALIDATION CMD_REQUEST_TIME", res)

        '''#FILTERING
        # Test filter
        for key in Test_data['Add_filter']:
            self.core2.add_filter_data(key, Test_data['Add_filter'][key])

        for key in Test_data['Modify_filter']:
            self.core2.set_filter_data(key, 0, Test_data['Modify_filter'][key])'''

        res = None
        t = time.time()
        while res is None and time.time() - t < timeout:
            res = self.core2.get_filter_list()
        print("DATA VALIDATION GET_FILTER_LIST", res)
        
        res = None
        t = time.time()
        while res is None and time.time() - t < timeout:
            res = self.core2.get_preview_order()
        print("DATA VALIDATION GET_PREVIEW_ORDER", res)
        
        res = None
        t = time.time()
        while res is None and time.time() - t < timeout:
            res = self.core2.get_process_order()
        print("DATA VALIDATION GET_PROCESS_ORDER", res)        
        
        res = None
        t = time.time()
        while res is None and time.time() - t < timeout:
            res = self.core2.view_cache_summary(0)
        print("DATA VALIDATION VIEW_CACHE_SUMMARY", res)
        
        res = None
        t = time.time()
        while res is None and time.time() - t < timeout:
            res = self.core2.view_segment_details(0, 1)
        print("DATA VALIDATION VIEW_SEGMENT_DETAILS", res)
        
        res = None
        t = time.time()
        while res is None and time.time() - t < timeout:
            res = self.core2.get_segment_feature_vectors(0, 1)
        print("DATA VALIDATION GET_SEGMENT_FEATURE_VECTORS", res)
        
        res = None
        t = time.time()
        while res is None and time.time() - t < timeout:
            res = self.core2.get_segment_events(0, 1, 0, 1)
        print("DATA VALIDATION GET_SEGMENT_EVENTS SEGMENTATION_ONSET", res)
        
        res = None
        t = time.time()
        while res is None and time.time() - t < timeout:
            res = self.core2.get_segment_events(0, 1, 0, 1)
        print("DATA VALIDATION GET_SEGMENT_EVENTS SEGMENT_SAVED", res)
        
        self.core2.set_DAC(4,122)
        res = None
        t = time.time()
        while res is None and time.time() - t < timeout:
            res = self.core2.get_peripherals()
        print("DATA VALIDATION GET_PERIPHERALS", res)
        
        res = None
        t = time.time()
        while res is None and time.time() - t < timeout:
            res = self.core2.get_terminus_list()
        print("DATA VALIDATION GET_TERMINUS_LIST", res)
        
        res = None
        t = time.time()
        while res is None and time.time() - t < timeout:
            res = self.core2.get_action_list()
        print("DATA VALIDATION GET_ACTION_LIST", res)
        
        res = None
        t = time.time()
        while res is None and time.time() - t < timeout:
            res = self.core2.suspend_action(0, 2, [1, 2])
        print("DATA VALIDATION SUSPEND_ACTIONS", res)
        
        # FEATURE_EXTRACT_BUFF_LEN
        # WORKS
        test_data = 199
        self.core2.set_feature_window_length(length=test_data)
        #self.core2.set_feature_window_length(length=[150, 200, 250], electrodes=[0, 4, 6])

        res = None
        t = time.time()
        while res is None and time.time() - t < timeout:
            res = self.core2.get_feature_window_length()
        print("DATA VALIDATION GET_FEATURE_EXTRACT_BUFF_LEN", res)
        
        res = None
        t = time.time()
        while res is None and time.time() - t < timeout:
            res = self.core2.get_general_fields()
        print("DATA VALIDATION GET_GENERAL_FIELDS", res)
        
        res = None
        t = time.time()
        while res is None and time.time() - t < timeout:
            res = self.core2.get_general_purpose_map_entry()
        print("DATA VALIDATION GET_GENERAL_PURPOSE_MAP_ENTRIES", res)
        
        res = None
        t = time.time()
        while res is None and time.time() - t < timeout:
            res = self.core2.get_firmware_version()
        print("DATA VALIDATION GET_FIRMWARE_VERSION", res)

    def TestStream(self):

        #self.core2.set_gains( gains=6)
        #self.core2.set_gains( gains=[6,6,6], electrodes=[0,4,8])

        #self.core2.set_electrodes( enabled_electrode = [True,True,False,True,True,False,True,True] )
        

        for i in range(6):

            #i = 5
            data_type = i # type of data, 1 is filtered EMG
            if i < 4:
                frequency = 4 # millisecs/sample
                quantity = 500
            else:
                frequency = 20
                quantity = 500 # number of samples (don't need to change)

            samp_rate = np.floor(1000/frequency)
            msg_time = np.floor(quantity/samp_rate)

            runtime = 15 # in seconds

            channels = [8,8,8,8,40,168]
            # viewing variables
            self.view(channels= channels[i])
            sleep(2)


            self.data_stream = []

            t = 4
            timeout = 3
            while not self.core2.check_ack(140):
                if time.time() - t > timeout:
                    print('starting...', data_type, frequency, quantity)
                    self.core2.stream_data_start(data_type=data_type,frequency=frequency,quantity=quantity)
                    t = time.time()
            print('start')

            t = time.time()

            while time.time() - t < runtime:

                print(time.time() - t)
                
                self.core2.stream_data_start(data_type=data_type,frequency=frequency,quantity=quantity)
                
                t2 = time.time()
                
                while time.time() - t2 < msg_time*0.9 and time.time() - t < runtime:
                    data = self.core2.get_emg_data()
                    if data is not None:
                        if self._view_event.is_set():
                            self._plot_buffer.put( np.array(data['DATA'] ) + random.randint(-self.offset,self.offset))            
                
                        self.data_stream.append( data['DATA'] ) 
                
            self.core2.cmd_stop_stream_data()

            while True:
                data = self.core2.get_emg_data()
                if data is not None:
                    if self._view_event.is_set():
                        self._plot_buffer.put( np.array(data['DATA'] ) + random.randint(-self.offset,self.offset))            
            
                    self.data_stream.append( data['DATA'] ) 
                else:
                    break

            '''self.core2.cmd_disconnect()
            
            self.core2.cmd_connect()'''

            #self.core2.cmd_stop_stream_data()
            self.core2.core2driver.flush()

            while not self.core2.flush_completed or not self.core2.core2driver.flush_completed:
                print('flushing')
            
            f = open('./data/core2stream_datatype_' + str(i) + '.csv', 'w', newline='')
            writer = csv.writer(f)
            writer.writerows( self.data_stream )
            f.close()

            self.flush()
            self.close()

            sleep(5)

    def _plot(self, channels = 8):
        """
        Generates a visualization of the incoming data in real-time
        """
        gui = plt.figure()
        gui.canvas.set_window_title( self._name )
        
        # line plot
        emg_plots = []
        
        emg_offsets = np.array( [self.offset*i for i in range(1, 2*channels, 2)] )

        ax = gui.add_subplot( 1, 1, 1 )
        num_emg_samps = int( 5 * np.round( 1.0 / self._speriod ) )
        for i in range( 0, channels ):
            xdata = np.linspace( 0, 1, num_emg_samps )
            ydata = emg_offsets[ i ] * np.ones( num_emg_samps )
            emg_plots.append( ax.plot( xdata, ydata ) )
        
        ax.set_ylim( 0, self.offset*channels*2 )
        ax.set_xlim( 0, 1 )
        ax.set_yticks( emg_offsets.tolist() )
        ylabs = ['CH_' + str(i+1) for i in range(channels)]
        ax.set_yticklabels( ylabs )
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
                    for i in range( 0, channels):
                        ydata = emg_plots[ i ][ 0 ].get_ydata()
                        ydata = np.append( ydata, data[ :, i ] + emg_offsets[ i ] )
                        ydata = ydata[-num_emg_samps:]
                        emg_plots[ i ][ 0 ].set_ydata( ydata )
                plt.pause( 0.005 )
            except: self._view_event.clear()
        plt.close( gui )

    def view(self, channels):
        """ 
        Launches the GUI viewer of the Myo armband
        """
        if not self._view_event.is_set():
            self._view_event.set()
            self._viewer = mp.Process( target = self._plot, args=[channels] )
            self._viewer.start()

    def flush(self):
        empty = False
        while not empty:
            try: self._plot_buffer.get( timeout = 1e-3 )
            except queue.Empty: empty = True

    def close(self):
        if self._view_event.is_set():
            self._view_exit_event.set()
            self._viewer.join()
            self._view_exit_event.clear()
            self._view_event.clear()

if __name__ == '__main__':
    core2stream = Core2Test()

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