import sys
import warnings
warnings.simplefilter("ignore", UserWarning)
sys.coinit_flags = 2

from copy import deepcopy as dc
import queue
import struct
import asyncio
import multiprocessing as mp
from multiprocessing import Manager
import time

import numpy as np
import matplotlib.pyplot as plt

from bleak import BleakClient
from bleak import BleakError

from Core2BLEDriver import Core2BLEDriver
from Core2BTClassicDriver import Core2BTClassicDriver

DEBUG = False

timing = True
rec_time = 45

UUID_CHAR_BRSP_INFO = '99564A02-DC01-4D3C-B04E-3BB1EF0571B2'
UUID_CHAR_BRSP_MODE = 'A87988B9-694C-479C-900E-95DFA6C00A24'
UUID_CHAR_BRSP_RX   = 'BF03260C-7205-4C25-AF43-93B1C299D159'
UUID_CHAR_BRSP_TX   = '18CDA784-4BD3-4370-85BB-BFED91EC86AF'
UUID_CHAR_BRSP_CTS  = '0A1934F5-24B8-4F13-9842-37BB167C6AFF'
UUID_CHAR_BRSP_RTS  = 'FDD6B4D3-046D-4330-BDEC-1FD0C90CB43B'

## RX STATE ENUMERATION
VOID_STATE = -1
START_STATE = 0
LENGTH_STATE = 1
DATA_STATE = 2
CRC_STATE = 3
## END

## COMMAND ENUMERATION
ACK = 0x0
NACK = 0x01
ACCEPT = 0x02
REJECT = 0x03
MSG_DATA = 0x05
MSG_PLAINTEXT = 0x06
MSG_RESPONSE = 0x07
MSG_EVENT = 0x08
CMD_CONNECT = 0x80
CMD_DISCONNECT = 0x81
CMD_REQUEST_TIME = 0x82
CMD_EDIT_CONTROL_STRATEGY = 0x83
CMD_DATA_MONITOR = 0x84
CMD_CONFIGURE_DAUGHTERBOARD = 0x85
CMD_MOVEMENT_ENGINE = 0x86
CMD_STREAM_DATA = 0x8C
CMD_STOP_STREAM_DATA = 0x8D
CMD_ADJUST_ELECTRODES = 0x8E
CMD_SET_LOGLEVEL = 0x8F
CMD_DEBUG_DRIVER = 0x90
CMD_CONFIGURATION = 0x91
## END

signal_type = { 'Raw': 0x00,
                'Filtered': 0x01,
                'MAV': 0x02,
                'Envelope': 0x03,
                'TD5': 0x04,
                'Full': 0x05
                }

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

class Core2Adapt:

    def __init__(self, address = 'EC:FE:7E:1D:C7:81', connection = 'BTClassic'):
        self._rx_state = VOID_STATE

        self.electrode_count = 8

        self.max_classes = 16

        self.num_classes = 3
        
        self.current_classifier_output = -1

        self.raw_emg_data_counter = -1
        self.previous_raw_emg_data_counter = -1

        self.filtered_emg_data_counter = -1
        self.previous_filtered_emg_data_counter = -1

        self.mav_emg_data_counter = -1
        self.previous_mav_emg_data_counter = -1

        self.envelope_emg_data_counter = -1
        self.previous_envelope_emg_data_counter = -1

        self.td5_emg_data_counter = -1
        self.previous_td5_emg_data_counter = -1

        self.full_emg_data_counter = -1
        self.previous_full_emg_data_counter = -1

        self.UFT = ["RESERVED", "UnanimousVoting", "OnsetThreshold", "ProportionalControl", "VelocityRamp"]

        #TODO
        self.testcounter = 0

        response_buffer = {}

        self.filter_data = {}
        self.terminus_data = {}

        self.response_queue = mp.Queue()
        self.response_queue.put( response_buffer, timeout = 1e-3 )

        self.rx_communication_queue = mp.Queue()
        self.tx_communication_queue = mp.Queue()

        self.rx_interaction_queue = mp.Queue()
        self.tx_interaction_queue = mp.Queue()

        self.ack_queue = mp.Queue()

        self._rx_exit_event = mp.Event()
        self._tx_exit_event = mp.Event()
        self._connection_event = mp.Event()
        self._record_exit_event = mp.Event()

        self._flush_completed = mp.Event()

        #self._connection_event.set()

        if connection == 'BLE':
            self.core2driver = Core2BLEDriver(mac = address)

            while not self.core2driver.connection:
                pass
        
        else:
            self.core2driver = Core2BTClassicDriver(mac = address)

            while not self.core2driver.connection:
                pass
        
        self._rx_process = mp.Process( target = self._rx_parsing )
        self._tx_process = mp.Process( target = self._tx_packaging )
        self._connection_process = mp.Process( target = self._record )

        self._rx_process.start()
        self._tx_process.start()
        self._connection_process.start()

    def _record(self):
        
        while not self.core2driver.exit and not self._record_exit_event.is_set():
            data = self.core2driver.state
            if data is not None:
                self._add_to_rx_communication_queue(data)

    def _tx_packaging(self):
        """
        Construct finalized byte array from mp queue and send to communication process mp queue
        """

        while not self._tx_exit_event.is_set():

            data = self._get_from_tx_interaction_queue()
            if len(data) > 0 and len(data[0]) > 0:
                for packet in data[0]:
                    
                    #packet = bytearray(item for sublist in packet for item in sublist)

                    tx_data = bytearray()
                    tx_data.extend( b'\xff' )

                    length = struct.pack('<H', np.int16(len(packet) + 1))

                    tx_data.extend( length )
                    
                    tx_data.extend( packet )

                    CRC = 0
                    for b in tx_data:
                        CRC += b 

                    CRC %= 256

                    tx_data.append( CRC )
                    
                    while not self.core2driver.ack_ready and not self._tx_exit_event.is_set():
                        pass

                    self.core2driver.transfer(tx_data)

    def _rx_parsing(self):
        """
        Construct finalized byte array
        
        Returns
        -------
        byte array
            parsed byte array with correctly ordered data

        """
        data_buffer = bytearray()
        terminate_local = False

        CRC = 0

        while not self._rx_exit_event.is_set() or len(data_buffer) > 0:
            
            command_byte = -1
            data_pack = bytearray()
            data = bytearray()

            data = self._get_from_rx_communication_queue()

            if len(data) > 0:

                for i in range(len(data)):
                    data_bytes = bytearray(item for sublist in data[i] for item in sublist)

                data_buffer.extend(data_bytes)                
            
            if len(data_buffer) > 0:

                if self._rx_state == VOID_STATE:
                    
                    for key, b in enumerate(data_buffer):
                        if b == 255:
                            
                            self._rx_state = START_STATE
                            data_buffer = data_buffer[key+1:]
                            CRC += 255
                            break

                    
                    if self._rx_state == VOID_STATE:
                        data_buffer = bytearray()

                if self._rx_state == START_STATE:

                    while len(data_buffer) < 2:

                        data = self._get_from_rx_communication_queue()
                        if len(data) > 0:
                            for i in range(len(data)):
                                data_bytes = bytearray(item for sublist in data[i] for item in sublist)
                            data_buffer.extend(data_bytes)
                        elif self._rx_exit_event.is_set():
                            break

                    self._rx_state = LENGTH_STATE

                if self._rx_state == LENGTH_STATE:

                    data_length = struct.unpack('<H', data_buffer[0:2])[0]

                    CRC += data_buffer[0] + data_buffer[1]

                    data_buffer = data_buffer[2:]

                    #TODO determine correct number
                    if data_length > 1000:
                        CRC = 0
                        self._rx_state = VOID_STATE
                        continue

                    while len(data_buffer) < data_length:

                        data = self._get_from_rx_communication_queue()
                        if len(data) > 0:
                            for i in range(len(data)):
                                data_bytes = bytearray(item for sublist in data[i] for item in sublist)
                            data_buffer.extend(data_bytes)
                        elif self._rx_exit_event.is_set():
                            break

                    self._rx_state = DATA_STATE

                if self._rx_state == DATA_STATE:

                    command_byte = data_buffer[0]

                    CRC += data_buffer[0]

                    data_buffer = data_buffer[1:]

                    if data_length > 2:
                        for i in range(0, data_length-2):
                            try:
                                CRC += data_buffer[i]
                            except:
                                if self._rx_exit_event.is_set():
                                    terminate_local = True
                                    break
                                else:
                                    continue
                        data_pack = dc(data_buffer[:data_length-2])

                    data_buffer = data_buffer[data_length-2:]

                    self._rx_state = CRC_STATE

                    if terminate_local: break
            
                if self._rx_state == CRC_STATE:

                    CRC %= 256

                    if CRC != data_buffer[0]:
                        pass

                    else:
                        self._command_distributor( command_byte, data_pack )
                    
                    CRC = 0
                    self._rx_state = VOID_STATE


                    try:
                        data_buffer = data_buffer[1:]
                    except:
                        data_buffer = bytearray()                

    def _command_distributor(self, command, data):

        if command == ACK:
            if DEBUG: print('ACK')
            self.ack(data)
        elif command == NACK:
            if DEBUG: print('NACK')
            self.nack(data)
        elif command == ACCEPT:
            if DEBUG: print('ACCEPT')
            self.accept(data)
        elif command == REJECT:
            if DEBUG: print('REJECT')
            self.reject(data)
        elif command == MSG_DATA:
            if DEBUG: print('MSG_DATA')
            self.msg_data(data)
        elif command == MSG_PLAINTEXT:
            if DEBUG: print('MSG_PLAINTEXT')
            self.msg_plaintext(data)
        elif command == MSG_RESPONSE:
            if DEBUG: print('MSG_RESPONSE')
            self.msg_response(data)
        elif command == MSG_EVENT:
            if DEBUG: print('MSG_EVENT')
            self.msg_event(data)
        elif command == CMD_CONNECT:
            if DEBUG: print('CMD_CONNECT')
            self.cmd_connect()
        elif command == CMD_DISCONNECT:
            if DEBUG: print('CMD_DISCONNECT')
            self.cmd_disconnect()
        elif command == CMD_REQUEST_TIME:
            if DEBUG: print('CMD_REQUEST_TIME')
            self.cmd_request_time(data)
        elif command == CMD_EDIT_CONTROL_STRATEGY:
            if DEBUG: print('CMD_EDIT_CONTROL_STRATEGY')
            self.cmd_edit_control_strategy(data)
        elif command == CMD_DATA_MONITOR:
            if DEBUG: print('CMD_DATA_MONITOR')
            self.cmd_data_monitor(data)
        elif command == CMD_CONFIGURE_DAUGHTERBOARD:
            if DEBUG: print('CMD_CONFIGURE_DAUGHTERBOARD')
            self.cmd_configure_daughterboard(data)
        elif command == CMD_MOVEMENT_ENGINE:
            if DEBUG: print('CMD_MOVEMENT_ENGINE')
            self.cmd_movement_engine(data)
        elif command == CMD_STREAM_DATA:
            if DEBUG: print('CMD_STREAM_DATA')
            self.cmd_stream_data(data)
        elif command == CMD_STOP_STREAM_DATA:
            if DEBUG: print('CMD_STOP_STREAM_DATA')
            self.cmd_stop_stream_data(data)
        elif command == CMD_ADJUST_ELECTRODES:
            if DEBUG: print('CMD_ADJUST_ELECTRODES')
            self.cmd_adjust_electrodes(data)
        elif command == CMD_SET_LOGLEVEL:
            if DEBUG: print('CMD_SET_LOGLEVEL')
            self.cmd_set_loglevel(data)
        elif command == CMD_DEBUG_DRIVER:
            if DEBUG: print('CMD_DEBUG_DRIVER')
            self.cmd_debug_driver(data)
        elif command == CMD_CONFIGURATION:
            if DEBUG: print('CMD_CONFIGURATION')
            self.cmd_configuration(data)
        else:
            if DEBUG: print('UNKNOWN COMMAND')

    def ack(self, data):
        
        try:
            self.core2driver.ack(data)
            self._add_ack_queue(data)
            #print('acked: ', data)
            if data[0] == CMD_CONNECT:                        
                while not self.core2driver.connection_acked:
                    pass
                self._connection_event.set()
        except Exception as e: print(e)
        '''tx_data = bytearray([ACK, data])
        self._add_to_tx_interaction_queue(tx_data)'''

    def nack(self, data):
        try:
            print('nacked: ', data)
        except Exception as e: print(e)
        '''tx_data = bytearray([NACK, data])
        self._add_to_tx_interaction_queue(tx_data)'''

    def accept(self, data):
        try:
            pass #print('accepted: ', data)
        except Exception as e: print(e)
        '''tx_data = bytearray([ACCEPT, data])
        self._add_to_tx_interaction_queue(tx_data)'''

    def reject(self, data):

        try:
            print('rejected: ', data)
        except Exception as e: print(e)
        
        '''tx_data = bytearray([REJECT, data])
        self._add_to_tx_interaction_queue(tx_data)'''

    def msg_data(self, data):

        ## MSG TYPE
        RAW_EMG = 0x00
        FILTERED_EMG = 0x01
        MAV_EMG = 0x02
        ENVELOPE_EMG = 0x03
        TD5_EMG = 0x04
        FULL_EMG = 0x05
        BETA_ROW = 0x06
        DICTIONARY_ROW = 0x07
        LEADOFF_INFO = 0x08
        GLIDE = 0x09
        FILTERING_HISTORY = 0x0A
        ## END

        res = []

        parse = type(data) == bytearray
    
        if parse:

            msg_type = data[0]
            data = data[1:]

            if msg_type == RAW_EMG:
                if DEBUG: print('RAW_EMG')
                self.previous_raw_emg_data_counter = self.raw_emg_data_counter
                self.raw_emg_data_counter = data[0]
                
                if (self.previous_raw_emg_data_counter+1)%256 != self.raw_emg_data_counter and self.previous_raw_emg_data_counter > -1:
                    print("raw_emg_data_counter mismatch, packet loss? current counter: ", self.raw_emg_data_counter, "previous counter: ", self.previous_raw_emg_data_counter)
                
                data = data[1:]
                for i in range(self.electrode_count):
                    int_data = data[i*4:i*4 + 4]
                    res.append(struct.unpack('<i', int_data)[0])
                
                data = data[32:]
                self.current_classifier_output = data[0]

                temp = self._get_response_queue('MSG_DATA')
                if temp is not None:
                    temp['TYPE'].append('RAW_EMG')
                    temp['DATA'].append(res)
                    temp['CLASS'].append(self.current_classifier_output)
                    self._add_to_response_queue('MSG_DATA', temp)
                else:
                    res = {'TYPE': ['RAW_EMG'], 'DATA': [res], 'CLASS': [self.current_classifier_output]}
                    self._add_to_response_queue('MSG_DATA', res)

            if msg_type == FILTERED_EMG:
                if DEBUG: print('FILTERED_EMG')
                self.previous_filtered_emg_data_counter = self.filtered_emg_data_counter
                self.filtered_emg_data_counter = data[0]
                
                if (self.previous_filtered_emg_data_counter+1)%256 != self.filtered_emg_data_counter and self.previous_filtered_emg_data_counter > -1:
                    print("filtered_emg_data_counter mismatch, packet loss? current counter: ", self.filtered_emg_data_counter, "previous counter: ", self.previous_filtered_emg_data_counter)
                
                data = data[1:]
                for i in range(self.electrode_count):
                    int_data = data[i*2:i*2 + 2]
                    int_data.extend(bytearray(b'\x00\x00'))
                    res.append(struct.unpack('<i', int_data)[0] / 32768)
                
                data = data[16:]
                self.current_classifier_output = data[0]

                temp = self._get_response_queue('MSG_DATA')
                if temp is not None:
                    #TODO for other types
                    temp['TYPE'].append('FILTERED_EMG')
                    temp['DATA'].append(res)
                    temp['CLASS'].append(self.current_classifier_output)
                    self._add_to_response_queue('MSG_DATA', temp)
                else:
                    res = {'TYPE': ['FILTERED_EMG'], 'DATA': [res], 'CLASS': [self.current_classifier_output]}
                    self._add_to_response_queue('MSG_DATA', res)

            if msg_type == MAV_EMG:
                if DEBUG: print('MAV_EMG')
                self.previous_mav_emg_data_counter = self.mav_emg_data_counter
                self.mav_emg_data_counter = data[0]
                
                if (self.previous_mav_emg_data_counter+1)%256 != self.mav_emg_data_counter and self.previous_mav_emg_data_counter > -1:
                    print("mav_emg_data_counter mismatch, packet loss? current counter: ", self.mav_emg_data_counter, "previous counter: ", self.previous_mav_emg_data_counter)
                
                data = data[1:]
                for i in range(self.electrode_count):
                    int_data = data[i*2:i*2 + 2]
                    int_data.extend(bytearray(b'\x00\x00'))
                    res.append(struct.unpack('<i', int_data)[0] / 32768)
                
                data = data[16:]
                self.current_classifier_output = data[0]

                temp = self._get_response_queue('MSG_DATA')
                if temp is not None:
                    temp['TYPE'].append('MAV_EMG')
                    temp['DATA'].append(res)
                    temp['CLASS'].append(self.current_classifier_output)
                    self._add_to_response_queue('MSG_DATA', temp)
                else:
                    res = {'TYPE': ['MAV_EMG'], 'DATA': [res], 'CLASS': [self.current_classifier_output]}
                    self._add_to_response_queue('MSG_DATA', res)

            if msg_type == ENVELOPE_EMG:
                if DEBUG: print('ENVELOPE_EMG')
                self.previous_envelope_emg_data_counter = self.envelope_emg_data_counter
                self.envelope_emg_data_counter = data[0]
                
                if (self.previous_envelope_emg_data_counter+1)%256 != self.envelope_emg_data_counter and self.previous_envelope_emg_data_counter > -1:
                    print("envelope_emg_data_counter mismatch, packet loss? current counter: ", self.envelope_emg_data_counter, "previous counter: ", self.previous_envelope_emg_data_counter)
                
                data = data[1:]
                for i in range(self.electrode_count):
                    int_data = data[i*2:i*2 + 2]
                    int_data.extend(bytearray(b'\x00\x00'))
                    res.append(struct.unpack('<i', int_data)[0] / 32768)
                
                data = data[16:]
                self.current_classifier_output = data[0]

                temp = self._get_response_queue('MSG_DATA')
                if temp is not None:
                    temp['TYPE'].append('ENVELOPE_EMG')
                    temp['DATA'].append(res)
                    temp['CLASS'].append(self.current_classifier_output)
                    self._add_to_response_queue('MSG_DATA', temp)
                else:
                    res = {'TYPE': ['ENVELOPE_EMG'], 'DATA': [res], 'CLASS': [self.current_classifier_output]}
                    self._add_to_response_queue('MSG_DATA', res)

            if msg_type == TD5_EMG:
                if DEBUG: print('TD5_EMG')
                self.previous_td5_emg_data_counter = self.td5_emg_data_counter
                self.td5_emg_data_counter = data[0]
                
                if (self.previous_td5_emg_data_counter+1)%256 != self.td5_emg_data_counter and self.previous_td5_emg_data_counter > -1:
                    print("td5_emg_data_counter mismatch, packet loss? current counter: ", self.td5_emg_data_counter, "previous counter: ", self.previous_td5_emg_data_counter)
                
                data = data[1:]
                for i in range(self.electrode_count * 5):
                    int_data = data[i*2 : i*2+2]
                    int_data.extend(bytearray(b'\x00\x00'))
                    res.append(struct.unpack('<i', int_data)[0] / 32768)

                data = data[80:]
                self.current_classifier_output = data[0]

                temp = self._get_response_queue('MSG_DATA')
                if temp is not None:
                    temp['TYPE'].append('TD5_EMG')
                    temp['DATA'].append(res)
                    temp['CLASS'].append(self.current_classifier_output)
                    self._add_to_response_queue('MSG_DATA', temp)
                else:
                    res = {'TYPE': ['TD5_EMG'], 'DATA': [res], 'CLASS': [self.current_classifier_output]}
                    self._add_to_response_queue('MSG_DATA', res)

            if msg_type == FULL_EMG:
                if DEBUG: print('FULL_EMG')
                self.previous_full_emg_data_counter = self.full_emg_data_counter
                self.full_emg_data_counter = data[0]
                
                if (self.previous_full_emg_data_counter+1)%256 != self.full_emg_data_counter and self.previous_full_emg_data_counter > -1:
                    print("full_emg_data_counter mismatch, packet loss? current counter: ", self.full_emg_data_counter, "previous counter: ", self.previous_full_emg_data_counter)
                
                data = data[1:]
                #print((len(data)-1)/16)
                for i in range(self.electrode_count * 21):
                    int_data = data[i*2 : i*2+2]
                    int_data.extend(bytearray(b'\x00\x00'))
                    res.append(struct.unpack('<i', int_data)[0] / 32768)
                
                self.current_classifier_output = data[-1]

                temp = self._get_response_queue('MSG_DATA')
                if temp is not None:
                    temp['TYPE'].append('FULL_EMG')
                    temp['DATA'].append(res)
                    temp['CLASS'].append(self.current_classifier_output)
                    self._add_to_response_queue('MSG_DATA', temp)
                else:
                    res = {'TYPE': ['FULL_EMG'], 'DATA': [res], 'CLASS': [self.current_classifier_output]}
                    self._add_to_response_queue('MSG_DATA', res)

            if msg_type == BETA_ROW:
                if DEBUG: print('BETA_ROW')
                beta_row_index = data[0]
                data = data[1:]
                for i in range(100):
                    float_data = data[i*4:i*4 + 4]
                    res.append(struct.unpack('<f', float_data)[0])

                return beta_row_index, res

            if msg_type == DICTIONARY_ROW:
                if DEBUG: print('DICTIONARY_ROW')
                int_data = data[0:2]
                int_data.extend(bytearray(b'\x00\x00'))
                dictionary_index = struct.unpack('<i', int_data)[0]
                data = data[2:]
                
                dictionary_label = data[0]
                data = data[1:]

                for i in range(288):
                    int_data = data[i*2 : i*2+2]
                    int_data.extend(bytearray(b'\x00\x00'))
                    res.append(struct.unpack('<i', int_data)[0] / 32768)

                return dictionary_index, res

            if msg_type == LEADOFF_INFO:
                if DEBUG: print('LEADOFF_INFO')
            if msg_type == GLIDE:
                if DEBUG: print('GLIDE')
            if msg_type == FILTERING_HISTORY:
                if DEBUG: print('FILTERING_HISTORY')
                int_data = data[0:4]
                filtering_history_timestamp = []
                filtering_history_timestamp = self.parse_u32(filtering_history_timestamp, int_data)[0]
                data = data[4:]

                filtering_history_classifier_was_skipped = data[0]
                data = data[1:]

                filtering_history_class_at_start = data[0]
                data = data[1:]

                filtering_history_speed_at_start = data[0]
                data = data[1:]

                filtering_history_class_at_end = data[0]
                data = data[1:]

                filtering_history_speed_at_end = data[0]

                res = { "filtering_history_timestamp": filtering_history_timestamp,
                        "filtering_history_classifier_was_skipped": filtering_history_classifier_was_skipped,
                        "filtering_history_class_at_start": filtering_history_class_at_start,
                        "filtering_history_speed_at_start": filtering_history_speed_at_start,
                        "filtering_history_class_at_end": filtering_history_class_at_end,
                        "filtering_history_speed_at_end": filtering_history_speed_at_end
                        }
                
                self._add_to_response_queue('FILTERING_HISTORY', res)

        pass

    def msg_plaintext(self, data):
        res = data.decode('utf-8')
        return res

    def msg_response(self, data):
        #print('msg_response', data)
        command = data[0]
        data = data[1:]
        self._command_distributor(command, data)

    def msg_event(self, data):

        if data[0] > 1:
            if data[0] == 2:
                if DEBUG: print('FLASH SAVE COMPLETED')
            elif data[0] == 1:
                if DEBUG: print('TEMPERATURE WARNING')

        else:

            res = { 'SEGMENTATION_STATE': 1}
            
            segmentation = data[0]
            data = data[1:]
            res['SEGMENTATION_RESPONSIBLE'] = data[0]
            data = data[1:]

            int_data = data[0:4]
            res['TIMESTAMP'] = struct.unpack('>I', int_data)[0]

            if segmentation == 0:
                self._add_to_response_queue('SEGMENTATION_ONSET', res)
            elif segmentation == 1:
                self._add_to_response_queue('SEGMENT_SAVED', res)
        

    def cmd_connect(self, data = []):
        if not len(data):
            data = bytearray(b'\x80')
            self._add_to_tx_interaction_queue(data)

    def cmd_disconnect(self, data = []):
        self.flush()
        if not len(data):
            data = bytearray(b'\x81')
            self._add_to_tx_interaction_queue(data)
            time.sleep(0.5)

    def cmd_request_time(self, data = []):
        if not len(data):
            data = bytearray(b'\x82')
            self._add_to_tx_interaction_queue(data)
        else:
            int_data = data[0:4]
            res = struct.unpack('>I', int_data)[0]
            self._add_to_response_queue('REQUEST_TIME', res)

    def cmd_edit_control_strategy(self, data):
        
        ## CONTROL FUNCTIONS
        PAUSE_CONTROL = 0x00
        START_CONTROL = 0x01
        CHANGE_SIZE = 0x02
        GET_FILTER_LIST = 0x03
        ADD_FILTER = 0x04
        MODIFY_FILTER = 0x05
        DELETE_FILTER = 0x06
        CLEAR_FILTERS = 0x07
        GET_PREVIEW_ORDER = 0x08
        SET_PREVIEW_ORDER = 0x09
        GET_PROCESS_ORDER = 0x0A
        SET_PROCESS_ORDER = 0x0B
        EASRC = 0x0C
        GLIDE = 0x0D
        RLDA = 0x0E
        STREAM_FILTER_ACTIVITY = 0x0F
        ## END

        command = data[0]
        data = data[1:]
        
        if command == PAUSE_CONTROL:
            if DEBUG:
                ut_data = data

            data_out = bytearray(b'\x83\x00')
            self._add_to_tx_interaction_queue(data_out)

            if DEBUG:
                res = []
                self.print_test(ut_data, res, 'PAUSE_CONTROL')

        elif command == START_CONTROL:
            if DEBUG:
                ut_data = data

            data_out = bytearray(b'\x83\x01')
            self._add_to_tx_interaction_queue(data_out)

            if DEBUG:
                res = []
                self.print_test(ut_data, res, 'START_CONTROL')

        elif command == CHANGE_SIZE:
            if DEBUG:
                ut_data = data

            data_out = bytearray(b'\x83\x02')
            data_out.extend(data)
            self._add_to_tx_interaction_queue(data_out)

            if DEBUG:
                res = []
                startpoint = 2
                res = self.parse_u8(res, data_out[startpoint:] )
                self.print_test(ut_data, res, 'CHANGE_SIZE')
        elif command == GET_FILTER_LIST:
            if not len(data):
                if DEBUG:
                    ut_data = data

                data_out = bytearray(b'\x83\x03')
                self._add_to_tx_interaction_queue(data_out)

                if DEBUG:
                    res = []
                    self.print_test(ut_data, res, 'GET_FILTER_LIST')
            else:
                res = {}
                num_filters = data[0]
                print(num_filters)
                data = data[1:]
                for i in range(num_filters):
                    res[self.UFT[data[i]]] = data[i + 1]
            
                '''if DEBUG:
                    print(res)'''

                self._add_to_response_queue('GET_FILTER_LIST', res)
        elif command == ADD_FILTER or command == MODIFY_FILTER:
            '''
            data should include UTF and order
            '''

            UFT = data[0]
            ORDER = data[1]

            if DEBUG:
                ut_data = []

            if command == MODIFY_FILTER:
                data_out = bytearray(b'\x83\x05')
                data_out = self.add_u8(data_out, UFT)
                data_out = self.add_u8(data_out, ORDER)
                debug_start = 4
            else:
                data_out = bytearray(b'\x83\x04')
                data_out = self.add_u8(data_out, UFT)
                debug_start = 3

            if self.UFT[UFT] == 'UnanimousVoting':
                
                data_out = self.add_u8(data_out, self.filter_data['UnanimousVoting'][ORDER]['Default'])
                data_out = self.add_u8(data_out, self.num_classes)

                if DEBUG:
                    ut_data.append(self.filter_data['UnanimousVoting'][ORDER]['Default'])
                    ut_data.append(self.num_classes)


                for i in range(self.num_classes):
                    data_out = self.add_u8(data_out, i)
                    data_out = self.add_u8(data_out, self.filter_data['UnanimousVoting'][ORDER][str(i)])

                    if DEBUG:
                        ut_data.append(i)
                        ut_data.append(self.filter_data['UnanimousVoting'][ORDER][str(i)])

                if DEBUG:
                    res = []
                    res = self.parse_u8(res, data_out[debug_start:] )
                    txt = 'MODIFY_FILTER' if command == MODIFY_FILTER else 'ADD_FILTER'
                    self.print_test(ut_data, res, 'UnanimousVoting', txt)

            elif self.UFT[UFT] == 'OnsetThreshold':
                data_out = self.add_q015(data_out, self.filter_data['OnsetThreshold'][ORDER]['Threshold'])
                data_out = self.add_u8(data_out, self.filter_data['OnsetThreshold'][ORDER]['Latching'])

                if DEBUG:
                    ut_data.append(self.filter_data['OnsetThreshold'][ORDER]['Threshold'])
                    ut_data.append(self.filter_data['OnsetThreshold'][ORDER]['Latching'])

                if DEBUG:
                    res = []
                    res = self.parse_q015(res, data_out[debug_start:debug_start+2] )
                    res = self.parse_u8(res, data_out[debug_start+2:] )
                    txt = 'MODIFY_FILTER' if command == MODIFY_FILTER else 'ADD_FILTER'
                    self.print_test(ut_data, res, 'OnsetThreshold', txt)

            elif self.UFT[UFT] == 'ProportionalControl':
                data_out = self.add_q015(data_out, self.filter_data['ProportionalControl'][ORDER]['Reference_MMAV'])

                if DEBUG:
                    ut_data.append(self.filter_data['ProportionalControl'][ORDER]['Reference_MMAV'])

                for key in self.filter_data['ProportionalControl'][ORDER]['General']:
                    data_out = self.add_float(data_out, self.filter_data['ProportionalControl'][ORDER]['General'][key])

                    if DEBUG:
                        ut_data.append(self.filter_data['ProportionalControl'][ORDER]['General'][key])
                
                data_out = self.add_u8(data_out, self.num_classes)

                if DEBUG:
                    ut_data.append(self.num_classes)

                for i in range(self.num_classes):
                    data_out = self.add_u8(data_out, i)

                    if DEBUG:
                        ut_data.append(i)

                    for key in self.filter_data['ProportionalControl'][ORDER][str(i)]:
                        data_out = self.add_float(data_out, self.filter_data['ProportionalControl'][ORDER][str(i)][key])

                        if DEBUG:
                            ut_data.append(self.filter_data['ProportionalControl'][ORDER][str(i)][key])

                if DEBUG:
                    
                    res = []
                    res = self.parse_q015(res, data_out[debug_start:debug_start+2] )

                    endpoint = len(self.filter_data['ProportionalControl'][ORDER]['General'].keys()) * 4

                    startpoint = debug_start + 2

                    res = self.parse_float(res, data_out[startpoint : startpoint + endpoint ] )

                    startpoint = startpoint + endpoint

                    res = self.parse_u8(res, data_out[startpoint : startpoint + 1 ] )

                    startpoint = startpoint + 1

                    for i in range(self.num_classes):
                        
                        res = self.parse_u8(res, data_out[startpoint: startpoint+1 ])

                        endpoint = len(self.filter_data['ProportionalControl'][ORDER][str(i)].keys()) * 4

                        res = self.parse_float(res, data_out[startpoint+1:startpoint+1 + endpoint] )

                        startpoint = startpoint + endpoint+1

                    txt = 'MODIFY_FILTER' if command == MODIFY_FILTER else 'ADD_FILTER'
                    self.print_test(ut_data, res, 'ProportionalControl', txt)
                
            elif self.UFT[UFT] == 'VelocityRamp':
                data_out = self.add_u8(data_out, self.filter_data['VelocityRamp'][ORDER]['Ramp_Length'])
                data_out = self.add_u8(data_out, self.filter_data['VelocityRamp'][ORDER]['Increment'])
                data_out = self.add_u8(data_out, self.filter_data['VelocityRamp'][ORDER]['Decrement'])

                data_out = self.add_float(data_out, self.filter_data['VelocityRamp'][ORDER]['Min_Speed_Multiplier'])
                data_out = self.add_float(data_out, self.filter_data['VelocityRamp'][ORDER]['Max_Speed_Multiplier'])

                if DEBUG:
                    ut_data.append(self.filter_data['VelocityRamp'][ORDER]['Ramp_Length'])
                    ut_data.append(self.filter_data['VelocityRamp'][ORDER]['Increment'])
                    ut_data.append(self.filter_data['VelocityRamp'][ORDER]['Decrement'])
                    ut_data.append(self.filter_data['VelocityRamp'][ORDER]['Min_Speed_Multiplier'])
                    ut_data.append(self.filter_data['VelocityRamp'][ORDER]['Max_Speed_Multiplier'])

                if DEBUG:
                    res = []
                    res = self.parse_u8(res, data_out[debug_start:debug_start+3] )
                    res = self.parse_float(res, data_out[debug_start+3:] )
                    txt = 'MODIFY_FILTER' if command == MODIFY_FILTER else 'ADD_FILTER'
                    self.print_test(ut_data, res, 'VelocityRamp', txt)
            
            self._add_to_tx_interaction_queue(data_out)                

        elif command == DELETE_FILTER:
            if DEBUG:
                ut_data = data

            data_out = bytearray(b'\x83\x06')
            data_out = self.add_u8(data_out, data)
            self._add_to_tx_interaction_queue(data_out)

            if DEBUG:
                res = []
                startpoint = 2
                res = self.parse_u8(res, data_out[startpoint:] )
                self.print_test(ut_data, res, 'DELETE_FILTER')
        elif command == CLEAR_FILTERS:
            if DEBUG:
                ut_data = data

            data_out = bytearray(b'\x83\x07')
            self._add_to_tx_interaction_queue(data_out)

            if DEBUG:
                res = []
                self.print_test(ut_data, res, 'CLEAR_FILTERS')
        elif command == GET_PREVIEW_ORDER:
            if not len(data):
                if DEBUG:
                    ut_data = data

                data_out = bytearray(b'\x83\x08')
                self._add_to_tx_interaction_queue(data_out)

                if DEBUG:
                    res = []
                    self.print_test(ut_data, res, 'GET_PREVIEW_ORDER')
            else:
                res = {}
                num_filters = data[0]
                data = data[1:]
                for i in range(0, len(data), 2):
                    res[self.UFT[data[i]]] = data[i + 1]
                self._add_to_response_queue('GET_PREVIEW_ORDER', res)
        elif command == SET_PREVIEW_ORDER:
            if DEBUG:
                ut_data = data

            data_out = bytearray(b'\x83\x09')
            data_out = self.add_u8(data_out, data)
            self._add_to_tx_interaction_queue(data_out)

            if DEBUG:
                res = []
                startpoint = 2
                res = self.parse_u8(res, data_out[startpoint:] )
                self.print_test(ut_data, res, 'SET_PREVIEW_ORDER')
        elif command == GET_PROCESS_ORDER:
            if not len(data):
                if DEBUG:
                    ut_data = data

                data_out = bytearray(b'\x83\x0A')
                self._add_to_tx_interaction_queue(data_out)

                if DEBUG:
                    res = []
                    self.print_test(ut_data, res, 'GET_PROCESS_ORDER')
            else:
                res = {}
                num_filters = data[0]
                data = data[1:]
                for i in range(0, len(data), 2):
                    res[self.UFT[data[i]]] = data[i + 1]
                self._add_to_response_queue('GET_PROCESS_ORDER', res)
        elif command == SET_PROCESS_ORDER:
            if DEBUG:
                ut_data = data

            data_out = bytearray(b'\x83\x0B')
            data_out = self.add_u8(data_out, data)
            self._add_to_tx_interaction_queue(data_out)

            if DEBUG:
                res = []
                startpoint = 2
                res = self.parse_u8(res, data_out[startpoint:] )
                self.print_test(ut_data, res, 'SET_PROCESS_ORDER')
        elif command == EASRC:
            if DEBUG: print('EASRC')
        elif command == GLIDE:
            if DEBUG: print('GLIDE')
        elif command == RLDA:            
            ## RLDA FUNCTIONS
            REPLACE_ALL_DATA = 0x00
            CHANGE_NUM_AXES = 0x01
            EDIT_AXIS = 0x02
            ## END

            subcommand = data[0]
            data = data[1:]
            
            if subcommand == REPLACE_ALL_DATA:
                if DEBUG:
                    ut_data = []

                data_out = bytearray(b'\x83\x0E\x00')
                Nc = data[0]
                data_out = self.add_u8(data_out, Nc)
                ut_data.append(Nc)
                data = data[1:]

                for i in range(288):
                    for j in range(Nc-1):
                        data_out = self.add_q015(data_out, data[i*(Nc-1) + j])

                        if DEBUG:
                            ut_data.append(data[i*(Nc-1) + j])

                data = data[288*(Nc-1):]
                for i in range(Nc-1):
                    for j in range(Nc):
                        data_out = self.add_float(data_out, data[i*Nc + j])

                        if DEBUG:
                            ut_data.append(data[i*Nc + j])

                if DEBUG:
                    res = []
                    startpoint = 3
                    res = self.parse_u8(res, data_out[startpoint:startpoint+1] )
                    startpoint = startpoint + 1
                    res = self.parse_q015(res, data_out[startpoint:startpoint+2*(Nc-1)*288] )
                    startpoint = startpoint+2*(Nc-1)*288
                    res = self.parse_float(res, data_out[startpoint:] )

                    self.print_test(ut_data, res, 'REPLACE_ALL_DATA')
                
                self._add_to_tx_interaction_queue(data_out)

            elif subcommand == CHANGE_NUM_AXES:
                if DEBUG: print('CHANGE_NUM_AXES')
                data_out = bytearray(b'\x83\x0E\x01')
                data_out = self.add_u8(data_out, data)
                self._add_to_tx_interaction_queue(data_out)
            # TODO
            elif subcommand == EDIT_AXIS:
                if DEBUG:
                    ut_data = data

                data_out = bytearray(b'\x83\x0E\x02')
                Nb = data[3]
                data_out = self.add_u8(data_out, data[0:4])
                data = data[4:]
                for i in range(Nb):
                    data_out = self.add_float(data_out, data[i])
                data = data[Nb:]
                for i in range(Nb+1):
                    data_out = self.add_u8(data_out, data[i])
                data = data[Nb+1:]
                for i in range(4):
                    data_out = self.add_float(data_out, data[i])

                if DEBUG:
                    res = []
                    startpoint = 3
                    res = self.parse_u8(res, data_out[startpoint:startpoint+4] )
                    startpoint = startpoint + 4
                    endpoint = startpoint + (Nb*4)
                    res = self.parse_float(res, data_out[startpoint:endpoint] )
                    startpoint = endpoint
                    res = self.parse_u8(res, data_out[startpoint:startpoint+Nb+1] )
                    startpoint = startpoint+Nb+1
                    res = self.parse_float(res, data_out[startpoint:] )
                    
                    self.print_test(ut_data, res, 'EDIT_AXIS')

                self._add_to_tx_interaction_queue(data_out)

        elif command == STREAM_FILTER_ACTIVITY:
            if DEBUG: print('STREAM_FILTER_ACTIVITY')
            data_out = bytearray(b'\x83\x0F')
            data_out.extend(data)# = self.add_u8(data_out, data)
            self._add_to_tx_interaction_queue(data_out)

    def cmd_data_monitor(self, data):
        
        ## DATA MONITOR FUNCTIONS
        EDIT_CACHE = 0x00
        RUN_TEMPORARY_CACHE = 0x01
        VIEW_CACHE_SUMMARY = 0x02
        VIEW_SEGMENT_DETAILS = 0x03
        GET_SEGMENT_FEATURE_VECTORS = 0x04
        REQUEST_EVENTS = 0x05
        ## END

        command = data[0]
        data = data[1:]
        
        if command == EDIT_CACHE:
            if DEBUG:
                ut_data = data

            data_out = bytearray(b'\x84\x00')
            data_out = self.add_u8(data_out, data[0:2])
            data = data[2:]
            data_out = self.add_q015(data_out, data[0])
            data = data[1:]
            data_out = self.add_u8(data_out, data[0])
            data = data[1:]
            data_out = self.add_u8(data_out, data[0])

            if DEBUG:
                res = []
                startpoint = 2
                res = self.parse_u8(res, data_out[startpoint:startpoint+2] )
                startpoint = startpoint + 2
                res = self.parse_q015(res, data_out[startpoint:startpoint+2] )
                startpoint = startpoint+2
                res = self.parse_u8(res, data_out[startpoint:] )
                self.print_test(ut_data, res, 'REPLACE_ALL_DATA')

            self._add_to_tx_interaction_queue(data_out)

        elif command == RUN_TEMPORARY_CACHE:
            if DEBUG:
                ut_data = data

            data_out = bytearray(b'\x84\x01')
            data_out = self.add_u32(data_out, data[0])
            self._add_to_tx_interaction_queue(data_out)

            if DEBUG:
                res = []
                startpoint = 2
                res = self.parse_u32(res, data_out[startpoint:] )
                self.print_test(ut_data, res, 'RUN_TEMPORARY_CACHE')

            
        elif command == VIEW_CACHE_SUMMARY:
            if DEBUG: print('VIEW_CACHE_SUMMARY')
            if len(data) == 1:
                data_out = bytearray(b'\x84\x02')
                data_out = self.add_u8(data_out, data[0])
                self._add_to_tx_interaction_queue(data_out)
            else:
                res = {}
                res['Cache being summarized'] = data[0]
                data = data[1:]
                Ns = data[0]
                res['Number of segments'] = Ns
                data = data[1:]
                res['Timestamps of segments'] = []
                res['Timestamps of segments'] = self.parse_u32(res['Timestamps of segments'], data)

                self._add_to_response_queue('VIEW_CACHE_SUMMARY', res)
        elif command == VIEW_SEGMENT_DETAILS:
            if len(data) == 2:
                if DEBUG:
                    ut_data = data

                data_out = bytearray(b'\x84\x03')
                data_out = self.add_u8(data_out, data[0])
                data = data[1:]
                data_out = self.add_u32(data_out, data[0])

                if DEBUG:
                    res = []
                    startpoint = 2
                    res = self.parse_u8(res, data_out[startpoint:startpoint+1] )
                    res = self.parse_u32(res, data_out[startpoint+1:] )
                    self.print_test(ut_data, res, 'VIEW_SEGMENT_DETAILS')

                self._add_to_tx_interaction_queue(data_out)
            else:
                res = {}
                res['Cache being viewed'] = data[0]
                data = data[1:]
                int_data = data[0:4]
                res['Timestamp of segment being viewed'] = []
                res['Timestamp of segment being viewed'] = self.parse_u32(res['Timestamp of segment being viewed'], int_data)[0]
                data = data[4:]
                int_data = data[0:2]
                res['Length (ms) of segment'] = []
                res['Length (ms) of segment'] = self.parse_u16(res['Length (ms) of segment'], int_data)[0]
                data = data[2:]
                res['Most common class label of segment'] = data[0]
                self._add_to_response_queue('VIEW_SEGMENT_DETAILS', res)
        elif command == GET_SEGMENT_FEATURE_VECTORS:
            if len(data) == 2:
                if DEBUG:
                    ut_data = data
                data_out = bytearray(b'\x84\x04')
                data_out = self.add_u8(data_out, data[0])
                data = data[1:]
                data_out = self.add_u32(data_out, data[0])

                if DEBUG:
                    res = []
                    startpoint = 2
                    res = self.parse_u8(res, data_out[startpoint:startpoint+1] )
                    res = self.parse_u32(res, data_out[startpoint+1:] )
                    self.print_test(ut_data, res, 'GET_SEGMENT_FEATURE_VECTORS')

                self._add_to_tx_interaction_queue(data_out)
            else:
                res = {}
                res['Response type'] = data[0]
                if data[0] == 0:
                    data = data[1:]
                    res['Cache to view'] = data[0]
                    data = data[1:]
                    int_data = data[0:4]
                    res['Timestamp of segment to view'] = []
                    res['Timestamp of segment to view'] = self.parse_u32(res['Timestamp of segment to view'], int_data)
                    data = data[4:]
                    res['Number of vectors to be sent'] = data[0]
                    self.Novtbs = res['Number of vectors to be sent']
                    self._get_segment_feature_vector_temp = []
                elif data[0] == 1:
                    data = data[1:]
                    res['Cache to view'] = data[0]
                    data = data[1:]
                    int_data = data[0:4]
                    res['Timestamp of segment to view'] = []
                    res['Timestamp of segment to view'] = self.parse_u32(res['Timestamp of segment to view'], int_data)[0]
                    data = data[4:]
                    int_data = data[0:2]
                    res['Number of milliseconds into segment of this vector'] = []
                    res['Number of milliseconds into segment of this vector'] = self.parse_u16(res['Number of milliseconds into segment of this vector'], int_data)[0]
                    data = data[2:]
                    res['Feature vector'] = []
                    res['Feature vector'] = self.parse_q015(res['Feature vector'], data)
                    self._get_segment_feature_vector_temp.append( res )
                elif data[0] == 2:
                    data = data[1:]
                    res['Cache to view'] = data[0]
                    data = data[1:]
                    int_data = data[0:4]
                    res['Timestamp of segment to view'] = []
                    res['Timestamp of segment to view'] = self.parse_u32(res['Timestamp of segment to view'], int_data)
                    data = data[4:]
                    res['Number of vectors sent'] = data[0]
                    if len(self._get_segment_feature_vector_temp) == res['Number of vectors sent'] and len(self._get_segment_feature_vector_temp) == self.Novtbs:
                        self._add_to_response_queue('GET_SEGMENT_FEATURE_VECTORS', self._get_segment_feature_vector_temp)
                    else:
                        self._add_to_response_queue('GET_SEGMENT_FEATURE_VECTORS', False)
                    self._get_segment_feature_vector_temp = []
                    self.Novtbs = -1
        elif command == REQUEST_EVENTS:
            if DEBUG:
                ut_data = data

            data_out = bytearray(b'\x84\x05')
            data_out = self.add_u16(data_out, data[0])
            data = data[1:]
            data_out = self.add_u16(data_out, data[0])
            data = data[1:]
            data_out = self.add_u16(data_out, data[0])
            data = data[1:]
            data_out = self.add_u16(data_out, data[0])

            if DEBUG:
                res = []
                startpoint = 2
                res = self.parse_u16(res, data_out[startpoint:] )
                self.print_test(ut_data, res, 'REQUEST_EVENTS')

            self._add_to_tx_interaction_queue(data_out)
        
    def cmd_configure_daughterboard(self, data):
        
        ## CONFIGURE DAUGHTERBOARD FUNCTIONS
        GET_PERIPHERALS = 0x00
        SET_STYLE = 0x01
        ## END
        
        if data[0] == GET_PERIPHERALS:
            if DEBUG: print('GET_PERIPHERALS')
            data = data[1:]
            if not len(data):
                data_out = bytearray(b'\x85\x00')
                self._add_to_tx_interaction_queue(data_out)
            else:
                res = {}
                res['Current style'] = data[0]
                data = data[1:]
                res['DAC'] = data[0]
                data = data[1:]
                res['Wrist PWM Driver'] = data[0]
                data = data[1:]
                res['Elbow PWM Driver'] = data[0]
                data = data[1:]
                res['Wrist Driver'] = data[0]
                data = data[1:]
                res['Elbow Driver'] = data[0]
                data = data[1:]
                res['IMU'] = data[0]
                data = data[1:]
                res['External UART'] = data[0]
                self._add_to_response_queue('GET_PERIPHERALS', res)
        elif data[0] == SET_STYLE:
            if DEBUG: print('SET_STYLE')
            data = data[1:]
            data_out = bytearray(b'\x85\x01')
            data_out = self.add_u8(data_out, data[0])
            self._add_to_tx_interaction_queue(data_out)

    def cmd_movement_engine(self, data):
        
        ## MOVEMENT ENGINE FUNCTIONS
        GET_TERMINUS_LIST = 0x00
        ADD_TERMINUS = 0x01
        MODIFY_TERMINUS = 0x02
        DELETE_TERMINUS = 0x03
        CLEAR_TERMINUSES = 0x04
        GET_ACTION_LIST = 0x05
        ADD_ACTION = 0x06
        MODIFY_ACTION = 0x07
        DELETE_ACTION = 0x08
        CLEAR_ACTIONS = 0x09
        TEST_ACTION = 0x0A
        SUSPEND_ACTIONS = 0x0B
        ## END

        command = data[0]
        data = data[1:]
                
        if command == GET_TERMINUS_LIST:
            if not len(data):
                if DEBUG:
                    ut_data = data
    
                data_out = bytearray(b'\x86\x00')
                self._add_to_tx_interaction_queue(data_out)

                if DEBUG:
                    res = []
                    self.print_test(ut_data, res, 'GET_TERMINUS_LIST')
            else:
                res = {}
                res['Number of terminuses'] = data[0]
                data = data[1:]
                for i in range(res['Number of terminuses']):
                    for key in UTIs:
                        if UTIs[key][0] == data[0]:
                            res[key] = {}
                            res[key]['instance'] = data[1]
                            res[key]['status'] = data[(res['Number of terminuses']-i)*2]                            
                            del data[(res['Number of terminuses']-i)*2]
                            data = data[2:]
                            break

                self._add_to_response_queue('GET_TERMINUS_LIST', res)

        elif command == ADD_TERMINUS:
            if DEBUG:
                ut_data = data

            data_out = bytearray(b'\x86\x01')
            data_out = self.add_u8(data_out, data)
            self._add_to_tx_interaction_queue(data_out)

            if DEBUG:
                res = []
                startpoint = 2
                res = self.parse_u8(res, data_out[startpoint:] )
                self.print_test(ut_data, res, 'ADD_TERMINUS')

        elif command == MODIFY_TERMINUS:
            if DEBUG: print('MODIFY_TERMINUS is deprecated')
            data = data[1:]
        elif command == DELETE_TERMINUS:
            if DEBUG:
                ut_data = data

            data_out = bytearray(b'\x86\x03')
            data_out = self.add_u8(data_out, data)
            self._add_to_tx_interaction_queue(data_out)

            if DEBUG:
                res = []
                startpoint = 2
                res = self.parse_u8(res, data_out[startpoint:] )
                self.print_test(ut_data, res, 'DELETE_TERMINUS')

        elif command == CLEAR_TERMINUSES:
            data_out = bytearray(b'\x86\x04')
            data_out = self.add_u8(data_out, data)

            self._add_to_tx_interaction_queue(data_out)
        elif command == GET_ACTION_LIST:
            if not len(data):
                if DEBUG:
                    ut_data = data
    
                data_out = bytearray(b'\x86\x05')
                self._add_to_tx_interaction_queue(data_out)

                if DEBUG:
                    res = []
                    self.print_test(ut_data, res, 'GET_ACTION_LIST')
            else:
                res = {}
                res['Number of actions'] = data[0]
                data = data[1:]
                for i in range(res['Number of actions']):
                    for key in UATs:
                        if UATs[key][0] == data[0]:
                            res[key] = {}
                            res[key]['instance'] = data[1]
                            res[key]['class'] = data[(res['Number of actions']-i)*2]                            
                            del data[(res['Number of actions']-i)*2]
                            data = data[2:]
                            break
                self._add_to_response_queue('GET_ACTION_LIST', res)
        elif command == ADD_ACTION:
            if DEBUG:
                ut_data = data

            data_out = bytearray(b'\x86\x06')
            data_out = self.add_u8(data_out, data)
            self._add_to_tx_interaction_queue(data_out)

            if DEBUG:
                res = []
                startpoint = 2
                res = self.parse_u8(res, data_out[startpoint:] )
                self.print_test(ut_data, res, 'ADD_ACTION')

        elif command == MODIFY_ACTION:
            if DEBUG: print('MODIFY_ACTION is deprecated')
            data = data[1:]
        elif command == DELETE_ACTION:
            if DEBUG:
                ut_data = data

            data_out = bytearray(b'\x86\x08')
            data_out = self.add_u8(data_out, data)
            self._add_to_tx_interaction_queue(data_out)

            if DEBUG:
                res = []
                startpoint = 2
                res = self.parse_u8(res, data_out[startpoint:] )
                self.print_test(ut_data, res, 'DELETE_ACTION')

        elif command == CLEAR_ACTIONS:
            data_out = bytearray(b'\x86\x09')
            self._add_to_tx_interaction_queue(data_out)
        elif command == TEST_ACTION:
            if DEBUG:
                ut_data = data

            data_out = bytearray(b'\x86\x0A')
            data_out = self.add_u8(data_out, data[:2])
            data_out = self.add_u16(data_out, data[2:])
            self._add_to_tx_interaction_queue(data_out)

            if DEBUG:
                res = []
                startpoint = 2
                res = self.parse_u8(res, data_out[startpoint:startpoint+2] )
                res = self.parse_u16(res, data_out[startpoint+2:] )
                self.print_test(ut_data, res, 'TEST_ACTION')
            
        elif command == SUSPEND_ACTIONS:
            if len(data)%2==0:
                if DEBUG:
                    ut_data = data

                data_out = bytearray(b'\x86\x0B')
                data_out = self.add_u8(data_out, data)
                self._add_to_tx_interaction_queue(data_out)

                if DEBUG:
                    res = []
                    startpoint = 2
                    res = self.parse_u8(res, data_out[startpoint:] )
                    self.print_test(ut_data, res, 'SUSPEND_ACTIONS')

            else:
                res = {}
                res['Number of actions suspended'] = data[0]
                data = data[1:]
                for i in range(res['Number of actions suspended']):
                    for key in UATs:
                        if UATs[key][0] == data[0]:
                            res[key] = {}
                            res[key]['instance'] = data[1]              
                            data = data[2:]
                            break

                self._add_to_response_queue('SUSPEND_ACTIONS', res)

    def cmd_stream_data(self, data):
        if DEBUG:
            ut_data = data
        
        data_out = bytearray(b'\x8C')
        data_out = self.add_u8(data_out, data[0])
        data = data[1:]
        data_out = self.add_u16(data_out, data[0])
        data = data[1:]
        data_out = self.add_u16(data_out, data[0])
        self._add_to_tx_interaction_queue(data_out)

        if DEBUG:
            res = []
            startpoint = 1
            res = self.parse_u8(res, data_out[startpoint:startpoint+1] )
            startpoint = startpoint + 1
            res = self.parse_u16(res, data_out[startpoint:startpoint+2] )
            startpoint = startpoint + 2
            res = self.parse_u16(res, data_out[startpoint:] )
            self.print_test(ut_data, res, 'STREAM_DATA')

    def cmd_stop_stream_data(self):
        data_out = bytearray(b'\x8D')
        self._add_to_tx_interaction_queue(data_out)
        self.flush()

    def cmd_adjust_electrodes(self, data):
        
        ## ADJUST ELECTRODES FUNCTIONS
        SET_GAIN = 0x00
        SET_SIMULATION = 0x01
        DISABLE_OR_ENABLE = 0x02
        GET_ENABLED = 0x03
        LEAD_OFF = 0x04
        GET_GAINS = 0x05
        SET_FEATURE_EXTRACT_BUFF_LEN = 0x06
        GET_FEATURE_EXTRACT_BUFF_LEN = 0x07
        ## END

        command = data[0]
        data = data[1:]
        
        if command == SET_GAIN:
            if DEBUG:
                ut_data = data

            data_out = bytearray(b'\x8E\x00')
            data_out = self.add_u8(data_out, data[0])
            data = data[1:]
            data_out = self.add_u8(data_out, data[0])
            print(data_out)
            self._add_to_tx_interaction_queue(data_out)

            if DEBUG:
                res = []
                startpoint = 2
                res = self.parse_u8(res, data_out[startpoint:startpoint+1] )
                startpoint = startpoint + 1
                res = self.parse_u8(res, data_out[startpoint:] )
                self.print_test(ut_data, res, 'SET_GAIN')

        elif command == SET_SIMULATION:
            ## NOT YET IMPLEMENTED
            pass
        elif command == DISABLE_OR_ENABLE:
            if DEBUG:
                ut_data = data

            data_out = bytearray(b'\x8E\x02')
            data_out = self.add_u8(data_out, data[0])
            self._add_to_tx_interaction_queue(data_out)

            if DEBUG:
                res = []
                startpoint = 2
                res = self.parse_u8(res, data_out[startpoint:] )
                self.print_test(ut_data, res, 'DISABLE_OR_ENABLE')
        elif command == GET_ENABLED:
            
            if not len(data):
                if DEBUG:
                    ut_data = data

                data_out = bytearray(b'\x8E\x03')
                self._add_to_tx_interaction_queue(data_out)
                
                if DEBUG:
                    res = []
                    self.print_test(ut_data, res, 'GET_ENABLED')
            else:
                res = {}
                temp = []
                val = '{:08b}'.format(data[0])
                for i in val:
                    temp.append( int(i) )
                res['Electrodes enabled'] = list(reversed(temp))

                data = data[1:]
                temp = []
                val = '{:08b}'.format(data[0])
                for i in val:
                    temp.append( int(i) )
                res['Electrodes operational'] = list(reversed(temp))
                
                self._add_to_response_queue('GET_ENABLED', res)
        elif command == LEAD_OFF:
            if DEBUG:
                ut_data = data

            data_out = bytearray(b'\x8E\x04')
            data_out = self.add_u8(data_out, data[0])
            self._add_to_tx_interaction_queue(data_out)

            if DEBUG:
                res = []
                startpoint = 2
                res = self.parse_u8(res, data_out[startpoint:] )
                self.print_test(ut_data, res, 'LEAD_OFF')
        elif command == GET_GAINS:
            if not len(data):
                if DEBUG:
                    ut_data = data

                data_out = bytearray(b'\x8E\x05')
                self._add_to_tx_interaction_queue(data_out)
                
                if DEBUG:
                    res = []
                    self.print_test(ut_data, res, 'GET_GAINS')
            else:
                res = {}
                res['Gain'] = []
                for i in range(0, 8):
                    res['Gain'].append( data[i] )
                self._add_to_response_queue('GET_GAINS', res)
        elif command == SET_FEATURE_EXTRACT_BUFF_LEN:
            if DEBUG:
                ut_data = data

            data_out = bytearray(b'\x8E\x06')
            data_out = self.add_u8(data_out, data[0])
            data = data[1:]
            data_out = self.add_u16(data_out, data[0:])
            self._add_to_tx_interaction_queue(data_out)

            if DEBUG:
                res = []
                startpoint = 2
                res = self.parse_u8(res, data_out[startpoint:startpoint+1] )
                startpoint = startpoint + 1
                res = self.parse_u16(res, data_out[startpoint:] )
                self.print_test(ut_data, res, 'SET_FEATURE_EXTRACT_BUFF_LEN')
        elif command == GET_FEATURE_EXTRACT_BUFF_LEN:
            if not len(data):
                if DEBUG:
                    ut_data = data

                data_out = bytearray(b'\x8E\x07')
                self._add_to_tx_interaction_queue(data_out)
                
                if DEBUG:
                    res = []
                    self.print_test(ut_data, res, 'GET_FEATURE_EXTRACT_BUFF_LEN')
            else:
                res = {}
                res['Length of buffer for electrode'] = []
                res['Length of buffer for electrode'] = self.parse_u16(res['Length of buffer for electrode'], data)
                self._add_to_response_queue('GET_FEATURE_EXTRACT_BUFF_LEN', res)

    def cmd_set_loglevel(self, data):
        data_out = bytearray(b'\x8F')
        data_out = self.add_u8(data_out, data[0])
        data = data[1:]
        data_out = self.add_u8(data_out, data[0])
        self._add_to_tx_interaction_queue(data_out)

    def cmd_debug_driver(self, data):
        
        ## DEBUG DRIVER FUNCTIONS
        DAC = 0x00
        EMG_FILTER_STREAMING = 0x01
        ## END

        command = data[0]
        data = data[1:]
        
        if command == DAC:
            if DEBUG:
                ut_data = data

            data_out = bytearray(b'\x90\x00')
            data_out = self.add_u8(data_out, data[0])
            data = data[1:]
            data_out = self.add_u8(data_out, data[0])
            self._add_to_tx_interaction_queue(data_out)

            if DEBUG:
                res = []
                startpoint = 2
                res = self.parse_u8(res, data_out[startpoint:startpoint+1] )
                startpoint = startpoint + 1
                res = self.parse_u8(res, data_out[startpoint:] )
                self.print_test(ut_data, res, 'DAC')
        
        if command == EMG_FILTER_STREAMING:

            data_out = bytearray(b'\x90\x01')
            data_out = self.add_u8(data_out, data[0])
            data = data[1:]
            data_out = self.add_u8(data_out, data[0])
            self._add_to_tx_interaction_queue(data_out)


    def cmd_configuration(self, data):
        
        ## CONFIGURATION FUNCTIONS
        GET_GENERAL_FIELDS = 0x00
        SET_GENERAL_FIELDS = 0x01
        SAVE_TO_FLASH = 0x02
        GET_GENERAL_PURPOSE_MAP_ENTRIES = 0x03
        ADD_GENERAL_PURPOSE_MAP_ENTRY = 0x04
        CLEAR_GENERAL_PURPOSE_MAP = 0x05
        RESET_ALL_STATE = 0x06
        GET_FIRMWARE_VERSION = 0x07
        ## END

        command = data[0]
        data = data[1:]
        
        if command == GET_GENERAL_FIELDS:
            if not len(data):
                if DEBUG:
                    ut_data = data

                data_out = bytearray(b'\x91\x00')
                self._add_to_tx_interaction_queue(data_out)

                if DEBUG:
                    res = []
                    self.print_test(ut_data, res, 'GET_GENERAL_FIELDS')
            else:
                res = {}

                #TODO parsing
                res['Motherboard style'] = data[0]
                data = data[1:]
                res['Motherboard revision'] = data[0]
                data = data[1:]
                res['CAN Baud'] = data[0]
                data = data[1:]
                '''res['Amputation side'] = data[0]
                data = data[1:]
                res['Amputation level'] = data[0]
                data = data[1:]'''
                res['Filter region'] = data[0]
                data = data[1:]
                res['Control strategy'] = data[0]
                data = data[1:]
                res['Mirroring flags'] = data[0]
                self._add_to_response_queue('GET_GENERAL_FIELDS', res)
        elif command == SET_GENERAL_FIELDS:
            if DEBUG:
                ut_data = data

            data_out = bytearray(b'\x91\x01')
            data_out = self.add_u8(data_out, data[0])
            data = data[1:]
            data_out = self.add_u8(data_out, data[0])
            data = data[1:]
            data_out = self.add_u8(data_out, data[0])
            data = data[1:]
            data_out = self.add_u8(data_out, data[0])
            data = data[1:]
            data_out = self.add_u8(data_out, data[0])
            data = data[1:]
            data_out = self.add_u8(data_out, data[0])
            data = data[1:]
            data_out = self.add_u8(data_out, data[0])
            data = data[1:]
            data_out = self.add_u8(data_out, data[0])
            self._add_to_tx_interaction_queue(data_out)

            if DEBUG:
                res = []
                startpoint = 2
                res = self.parse_u8(res, data_out[startpoint:startpoint+1] )
                startpoint = startpoint + 1
                res = self.parse_u8(res, data_out[startpoint:startpoint+1] )
                startpoint = startpoint + 1
                res = self.parse_u8(res, data_out[startpoint:startpoint+1] )
                startpoint = startpoint + 1
                res = self.parse_u8(res, data_out[startpoint:startpoint+1] )
                startpoint = startpoint + 1
                res = self.parse_u8(res, data_out[startpoint:startpoint+1] )
                startpoint = startpoint + 1
                res = self.parse_u8(res, data_out[startpoint:startpoint+1] )
                startpoint = startpoint + 1
                res = self.parse_u8(res, data_out[startpoint:startpoint+1] )
                startpoint = startpoint + 1
                res = self.parse_u8(res, data_out[startpoint:] )
                self.print_test(ut_data, res, 'SET_GENERAL_FIELDS')
        elif command == SAVE_TO_FLASH:
            if DEBUG:
                ut_data = data

            data_out = bytearray(b'\x91\x02')
            self._add_to_tx_interaction_queue(data_out)

            if DEBUG:
                res = []
                self.print_test(ut_data, res, 'SAVE_TO_FLASH')
        elif command == GET_GENERAL_PURPOSE_MAP_ENTRIES:
            if not len(data):
                if DEBUG:
                    ut_data = data

                data_out = bytearray(b'\x91\x03')
                self._add_to_tx_interaction_queue(data_out)

                if DEBUG:
                    res = []
                    self.print_test(ut_data, res, 'GET_GENERAL_PURPOSE_MAP_ENTRIES')
            else:
                res = {}
                res['Number of entries'] = data[0]
                data = data[1:]
                for i in range(0, res['Number of entries']*3,3):
                    int_data = data[i:i+2]
                    temp = []
                    temp = self.parse_u16(temp, int_data)
                    res[temp[0]] = data[i+2]
                self._add_to_response_queue('GET_GENERAL_PURPOSE_MAP_ENTRIES', res)
        elif command == ADD_GENERAL_PURPOSE_MAP_ENTRY:
            if DEBUG:
                ut_data = data

            data_out = bytearray(b'\x91\x04')
            data_out = self.add_u16(data_out, data[0])
            data = data[1:]
            data_out = self.add_u8(data_out, data[0])
            self._add_to_tx_interaction_queue(data_out)

            if DEBUG:
                res = []
                startpoint = 2
                res = self.parse_u16(res, data_out[startpoint:startpoint+2] )
                startpoint = startpoint + 2
                res = self.parse_u8(res, data_out[startpoint:] )                
                self.print_test(ut_data, res, 'ADD_GENERAL_PURPOSE_MAP_ENTRY')
        elif command == CLEAR_GENERAL_PURPOSE_MAP:
            if DEBUG:
                ut_data = data

            data_out = bytearray(b'\x91\x05')
            self._add_to_tx_interaction_queue(data_out)

            if DEBUG:
                res = []
                self.print_test(ut_data, res, 'CLEAR_GENERAL_PURPOSE_MAP')
        elif command == RESET_ALL_STATE:
            if DEBUG:
                ut_data = data

            data_out = bytearray(b'\x91\x06')
            self._add_to_tx_interaction_queue(data_out)

            if DEBUG:
                res = []
                self.print_test(ut_data, res, 'RESET_ALL_STATE')
        elif command == GET_FIRMWARE_VERSION:
            if not len(data):
                if DEBUG:
                    ut_data = data

                data_out = bytearray(b'\x91\x07')
                self._add_to_tx_interaction_queue(data_out)

                if DEBUG:
                    res = []
                    self.print_test(ut_data, res, 'GET_FIRMWARE_VERSION')
            else:
                res = {}
                res['Version string length'] = data[0]
                data = data[1:]
                res['Version string characters'] = []
                for i in range(0, res['Version string length']):
                    res['Version string characters'].append( data[i] )
                self._add_to_response_queue('GET_FIRMWARE_VERSION', res)

    ### PUBLIC FUNCTIONS
    # Filter modification
    def add_filter_data(self, UFT, data):
        """
        Add filter, one filter at a time
        
        Parameters
        ----------
        UFT: list of Strings
            Name of filter: "UnanimousVoting", "OnsetThreshold", "ProportionalControl", "VelocityRamp"

        data: Dictionary, context dependant
            UnanimousVoting :       { 'Default' : int, '0' : int, ..., '#ofclasses-1': int},
            OnsetThreshold :        { 'Threshold': float, 'Latching' : bool },
            ProportionalControl :   {   'Reference_MMAV': 0, 
                                        'General': {'Min_Speed_Multiplier': 0, 'Max_Speed_Multiplier': 0, 'MMAV_ratio_lower': 0, 'MMAV_ratio_upper': 0},
                                        '0' : {'Min_Speed_Multiplier': 0, 'Max_Speed_Multiplier': 0, 'MMAV_ratio_lower': 0, 'MMAV_ratio_upper': 0}, 
                                        ..., 
                                        '#ofclasses-1': {'Min_Speed_Multiplier': 0, 'Max_Speed_Multiplier': 0, 'MMAV_ratio_lower': 0, 'MMAV_ratio_upper': 0}
                                    },
            VelocityRamp : { 'Ramp_Length' : 0, 'Increment' : 0, 'Decrement' : 0,'Min_Speed_Multiplier' : 0,'Max_Speed_Multiplier' : 0 },
        
        """

        if UFT in self.filter_data.keys():
            next_idx = max(self.filter_data[UFT].keys())+1
            self.filter_data[UFT][next_idx] = data
        else:
            self.filter_data[UFT] = {}
            self.filter_data[UFT][0] = data
            next_idx = 0

        data = bytearray(b'\x04')
        data = self.add_u8(data, self.UFT.index(UFT) )
        data = self.add_u8(data, next_idx )

        self.cmd_edit_control_strategy(data)


    def set_filter_data(self, UFT, ORDER, data):
        """
        Change filter settings, one filter at a time (use get_filter_data to see existing filters)
        
        Parameters
        ----------
        UFT: String
            Name of filter: "UnanimousVoting", "OnsetThreshold", "ProportionalControl", "VelocityRamp"

        ORDER: index of filter type

        data: Dictionary, context dependant
            UnanimousVoting :       { 'Default' : int, '0' : int, ..., '#ofclasses-1': int},
            OnsetThreshold :        { 'Threshold': float, 'Latching' : bool },
            ProportionalControl :   {   'Reference_MMAV': 0, 
                                        'General': {'Min_Speed_Multiplier': 0, 'Max_Speed_Multiplier': 0, 'MMAV_ratio_lower': 0, 'MMAV_ratio_lower': 0},
                                        '0' : {'Min_Speed_Multiplier': 0, 'Max_Speed_Multiplier': 0, 'MMAV_ratio_lower': 0, 'MMAV_ratio_lower': 0}, 
                                        ..., 
                                        '#ofclasses-1': {'Min_Speed_Multiplier': 0, 'Max_Speed_Multiplier': 0, 'MMAV_ratio_lower': 0, 'MMAV_ratio_lower': 0}
                                    },
            VelocityRamp : { 'Ramp_Length' : 0, 'Increment' : 0, 'Decrement' : 0,'Min_Speed_Multiplier' : 0,'Max_Speed_Multiplier' : 0 },
        
        """

        try:
            if UFT == self.UFT[1]: 
                for key in data:
                    self.filter_data[UFT][ORDER][key] = data[key]                
            elif UFT == self.UFT[2]:    
                for key in data:
                    self.filter_data[UFT][ORDER][key] = data[key]  
            elif UFT == self.UFT[3]:    
                for key in data:
                    if key == 'Reference_MMAV':
                        self.filter_data[UFT][ORDER][key] = data[key]
                    else:
                        for inner_key in data[key]:
                            self.filter_data[UFT][ORDER][key][inner_key] = data[key][inner_key]
            elif UFT == self.UFT[4]:    
                for key in data:
                    self.filter_data[UFT][ORDER][key] = data[key]

            data = bytearray(b'\x05')
            data = self.add_u8(data, self.UFT.index(UFT) )
            data = self.add_u8(data, ORDER )

            self.cmd_edit_control_strategy(data)

        except:
                print('Dictionary entry with unmatched key')

    def get_filter_data(self):
        """
        Add filter settings
        
        Parameters
        ----------
        dictionary of 
        """
        return self.filter_data

    def replace_RLDA_all(self, RLDA, Means):
        """
        Replace all RLDA 
        
        Parameters
        ----------
        RLDA: np.array of RLDA transformation matrix n x m (n: feature ; m: # of classes-1), will convert to Q0.15

        Cluster_means: np.array of RLDA cluster means n x m (n: # of classes ; m: # of classes-1), will convert to float
        
        """

        if RLDA.shape[0] != self.num_classes-1:
            Warning("Number of classes and 2nd dimention of RLDA matrix must be equal")
        else:
            data = [14, 0, self.num_classes]
            for i in RLDA:
                for j in i:
                    data.append(j)
            
            for i in Means:
                for j in i:
                    data.append(j)

            self.cmd_edit_control_strategy(data)

    def change_num_axes_RLDA(self, num_axes):
        """
        Delete all existing axes and change the number of axes
        
        Parameters
        ----------
        num_axes: int (U8)
        
        """

        data = bytearray(b'\x0E\x01')
        data = self.add_u8( num_axes )

        self.cmd_edit_control_strategy(data)

    def edit_RLDA_axis(self, axis_number, start_cluster, end_cluster, boundary, class_indicies, proportional_ramp_start, proportional_ramp_stop, min_proportional_speed, max_proportional_speed):
        """
        Change the parameters of a particular axis. Note the number of class indexes is the number of boundaries + 1.
        
        Parameters
        ----------
        axis_number: int (U8)
        start_cluster: int (U8)
        end_cluster: int (U8)
        boundary: list of floats
        class_indicies: list of int (U8)
        proportional_ramp_start: float
        proportional_ramp_stop: float
        min_proportional_speed: float
        max_proportional_speed: float
        """

        data = [14, 2]

        if len(boundary) != len(class_indicies)-1:
            Warning("Length of boundary must be equal to length of class_indicies-1")
            return False
        
        data.append(axis_number)
        data.append(start_cluster)
        data.append(end_cluster)
        data.append(len(boundary))

        for i in boundary:
            data.append(i)

        for i in class_indicies:
            data.append(i)

        data.append(proportional_ramp_start)
        data.append(proportional_ramp_stop)
        data.append(min_proportional_speed)
        data.append(max_proportional_speed)

        self.cmd_edit_control_strategy(data)
        
    def stream_filter_activity(self, duration):
        """
        Controls whether to transmit MSG_DATA filtering history packets after each filtering operation. 
        Indefinite streaming is accomplished by sending a new STREAM_FILTER_ACTIVITY;        
        the expiry is seamlessly extended to the current time plus duration
        
        Parameters
        ----------
        duration: int (U8) ms
        
        """

        data = bytearray(b'\x0F')
        data = self.add_u8( data, duration )

        self.cmd_edit_control_strategy(data)

    def edit_segmenter_cache(self, cache_type, max_segments, segment_onset_threshold, minimum_samples, vectors_per_segment):
        """
        Change the settings of a particular cache, including its segmenter. This will delete any segments in the cache being edited.
        
        Parameters
        ----------
        cache_type: int (U8), 0 - permanent 1 - temporary
        max_segments: int (U8), Maximum number of segments to store
        segment_onset_threshold: float (q0.15) [0,1]
        minimum_samples:
        vectors_per_segment:
        """

        data = [0]
        data.append(cache_type)
        data.append(max_segments)
        data.append(segment_onset_threshold)
        data.append(minimum_samples)
        data.append(vectors_per_segment)

        self.cmd_data_monitor(data)

    def run_temporary_cache(self, runtime):
        """
        View a summary of the segments in a cache.
        
        Parameters
        ----------
        runtime: int (U32) ms
        """

        data = [1]
        data.append(runtime)

        self.cmd_data_monitor(data)
    
    def get_emg_data(self):
        """
        Get EMG data.
        """

        return self._get_from_response_queue('MSG_DATA')
    
    def get_filter_activity(self):
        """
        Get filter activity.
        """

        return self._get_from_response_queue('FILTERING_HISTORY')

    def request_time_data(self):
        """
        Request time data.
        """
        try:
            send = self.request_time_data_send
        except:
            self.request_time_data_send = True
            send = self.request_time_data_send

        if send:
            self.cmd_request_time()
            self.request_time_data_send = False
        else:
            res = self._get_from_response_queue('REQUEST_TIME')
            if res is not None:
                print('REQUEST_TIME', res)
            return res

    def get_filter_list(self):
        """
        Get filter list
        """
        try:
            send = self.get_filter_list_send
        except:
            self.get_filter_list_send = True
            send = self.get_filter_list_send

        if send:
            data = [3]

            self.cmd_edit_control_strategy(data)
            self.get_filter_list_send = False
        else:
            res = self._get_from_response_queue('GET_FILTER_LIST')
            if res is not None:
                print('GET_FILTER_LIST', res)
            return res

    def get_preview_order(self):
        """
        Get preview order
        """
        try:
            send = self.get_preview_order_send
        except:
            self.get_preview_order_send = True
            send = self.get_preview_order_send

        if send:
            data = [8]

            self.cmd_edit_control_strategy(data)
            self.get_preview_order_send = False
        else:
            return self._get_from_response_queue('GET_PREVIEW_ORDER')

    def get_process_order(self):
        """
        Get process order
        """
        try:
            send = self.get_process_order_send
        except:
            self.get_process_order_send = True
            send = self.get_process_order_send

        if send:
            data = [10]

            self.cmd_edit_control_strategy(data)
            self.get_process_order_send = False
        else:
            return self._get_from_response_queue('GET_PROCESS_ORDER')

    def view_cache_summary(self, cache):
        """
        Run the temporary cache for a pre-set time window. 
        
        Parameters
        ----------
        cache: int (U8) Cache to view
        """
        try:
            send = self.view_cache_summary_send
        except:
            self.view_cache_summary_send = True
            send = self.view_cache_summary_send

        if send:
            data = [2]
            data.append(cache)

            self.cmd_data_monitor(data)
            self.view_cache_summary_send = False
        else:
            return self._get_from_response_queue('VIEW_CACHE_SUMMARY')

    def view_segment_details(self, cache, timestamp):
        """
        View the details of a particular segment. 
        Takes timestamp as an input, and will return a reject if no such segment exists
        
        Parameters
        ----------
        cache: int (U8) Cache to view
        timestamp: int (U32) Timestamp of segment to view (from cache summary)
        """
        try:
            send = self.view_segment_details_send
        except:
            self.view_segment_details_send = True
            send = self.view_segment_details_send

        if send:
            data = [3]
            data.append(cache)
            data.append(timestamp)

            self.cmd_data_monitor(data)
            self.view_segment_details_send = False
        else:
            return self._get_from_response_queue('VIEW_SEGMENT_DETAILS')
    
    def get_segment_feature_vectors(self, cache, timestamp):
        """
        Request the feature vectors composing a segment. 
        Takes timestamp as an input, and will return a reject if no such segment exists
        
        Parameters
        ----------
        cache: int (U8) Cache to view
        timestamp: int (U32) Timestamp of segment to view (from cache summary)
        """
        try:
            send = self.get_segment_feature_vectors_send
        except:
            self.get_segment_feature_vectors_send = True
            send = self.get_segment_feature_vectors_send

        if send:
            data = [4]
            data.append(cache)
            data.append(timestamp)

            self.cmd_data_monitor(data)
            self.get_segment_feature_vectors_send = False
        else:
            return self._get_from_response_queue('GET_SEGMENT_FEATURE_VECTORS')

    def get_segment_events(self, perm_start, perm_save, temp_start, temp_save):
        """
        Request MSG_EVENTs to be sent for various conditions. Requests are cleared on Bluetooth disconnection. 
        For each event, 0 disables the event, and nonzero values set the expiry in milliseconds of the event request.
        
        Parameters
        ----------
        perm_start: int (U16) Request segment start events on permanent segmenter
        perm_save: int (U16) Request segment save events on permanent segmenter
        temp_start: int (U16) Request segment start events on temporary segmenter
        temp_save: int (U16) Request segment save events on temporary segmenter
        """
        try:
            send = self.get_segment_events_send
        except:
            self.get_segment_events_send = True
            send = self.get_segment_events_send

        if send:
            data = [5]
            data.append(perm_start)
            data.append(perm_save)
            data.append(temp_start)
            data.append(temp_save)

            self.cmd_data_monitor(data)
            self.get_segment_events_send = False
        else:
            res = self._get_from_response_queue('SEGMENTATION_ONSET')
            if res is not None:
                return res
            else:
                return self._get_from_response_queue('SEGMENT_SAVED')

    def get_peripherals(self):
        """
        View the embedded system’s current configuration of what peripherals are available, and which have initialized properly. 
        This is not a hardware search, it only reports what it has been told.
        
        """
        try:
            send = self.get_peripherals_send
        except:
            self.get_peripherals_send = True
            send = self.get_peripherals_send

        if send:
            data = [0]

            self.cmd_configure_daughterboard(data)
            self.get_peripherals_send = False
        else:
            return self._get_from_response_queue('GET_PERIPHERALS')

    def set_style(self, style):
        """
        Set the style of the daughterboard, which identifies which electrical components 
        are present on the physically attached daughterboard. 
        Causes a rebuild of the movement engine when switching to a new style.
        
        Parameters
        ----------
        style: int (U8) (0=NC, 1=OMNI)
        """

        data = [1]
        data.append(style)

        self.cmd_configure_daughterboard(data)

    ### TERMINAL DEVICE COMMANDS  
    def get_terminus_list(self):
        """
        Get a list of current Terminuses with information about whether they’re functioning correctly.
        
        """
        try:
            send = self.get_terminus_list_send
        except:
            self.get_terminus_list_send = True
            send = self.get_terminus_list_send

        if send:
            data = [0]
            self.cmd_movement_engine(data)
            self.get_terminus_list_send = False
        else:
            return self._get_from_response_queue('GET_TERMINUS_LIST')
    
    def clear_terminuses(self):
        """
        Remove all terminuses. Clears actions.
        
        """

        data = [4]
        self.cmd_movement_engine(data)
    
    def delete_terminus(self, UTI, instance):
        """
        Delete an existing terminus. Clears actions. A reject will be sent if the terminus does not exist.

        Parameters
        ----------
        UTI: int (U8) device ID
        instance: unique counter number 
        """

        data = [3]
        data.append(UTI)
        data.append(instance)
        self.cmd_movement_engine(data)

    def add_SimpleHand(self, instance = -1, open_DAC = 0, close_DAC = 1, min_out_DAC = 0, max_out_DAC = 1, dependent = 0):
        """
        Add (or overwrite) device to the controller with the following parameters
        
        Parameters
        ----------
        instance: the instance to overwrite, if new device, leave as default
        open_DAC: int (U8) DAC channel used for open 0-7
        close_DAC: int (U8) DAC channel used for close 0-7
        min_out_DAC: float Minimum DAC output 0-1 -> 0-5V
        max_out_DAC: float Maximum DAC output 0-1 -> 0-5V
        dependent: int (U8) Dependent on Espire elbow; 0=No, 1=Yes, DAC channels will be ignored, rejected if EspireElbow terminus does not exist
        """

        if instance > -1:
            self.delete_terminus(UTIs['SimpleHand'][0], instance)

        data = [1]
        data.append( UTIs['SimpleHand'][0] )
        data.append(open_DAC)
        data.append(close_DAC)
        data.append( int(min_out_DAC*255) )
        data.append( int(max_out_DAC*255))
        data.append(dependent)

        self.cmd_movement_engine(data)

    def add_MotorDrivenWrist(self, instance = -1, min_out_PWN = 0, max_out_PWN = 1, dependent = 0):
        """
        Add (or overwrite) device to the controller with the following parameters
        
        Parameters
        ----------
        instance: the instance to overwrite, if new device, leave as default
        min_out_PWN: float Minimum PWM output 0-1 -> 0-100%
        max_out_PWN: float Maximum PWM output 0-1 -> 0-100%
        dependent: int (U8) Dependent on Espire elbow; 0=No, 1=Yes
        """

        if instance > -1:
            self.delete_terminus(UTIs['MotorDrivenWrist'][0], instance)

        data = [1]
        data.append( UTIs['MotorDrivenWrist'][0] )
        data.append( int(min_out_PWN*255) )
        data.append( int(max_out_PWN*255))
        data.append(dependent)

        self.cmd_movement_engine(data)

    def add_MotorDrivenElbow(self, instance = -1, min_out_PWN = 0, max_out_PWN = 1):
        """
        Add (or overwrite) device to the controller with the following parameters
        
        Parameters
        ----------
        instance: the instance to overwrite, if new device, leave as default
        min_out_PWN: float Minimum PWM output 0-1 -> 0-100%
        max_out_PWN: float Maximum PWM output 0-1 -> 0-100%
        """

        if instance > -1:
            self.delete_terminus(UTIs['MotorDrivenElbow'][0], instance)

        data = [1]
        data.append( UTIs['MotorDrivenElbow'][0] )
        data.append( int(min_out_PWN*255) )
        data.append( int(max_out_PWN*255))

        self.cmd_movement_engine(data)

    def add_DacWrist(self, instance = -1, pronate_DAC = 0, supinate_DAC = 1, min_out_DAC = 0, max_out_DAC = 1):
        """
        Add (or overwrite) device to the controller with the following parameters
        
        Parameters
        ----------
        instance: the instance to overwrite, if new device, leave as default
        pronate_DAC: int (U8) DAC channel used for open 0-7
        supinate_DAC: int (U8) DAC channel used for close 0-7
        min_out_DAC: float Minimum DAC output 0-1 -> 0-5V
        max_out_DAC: float Maximum DAC output 0-1 -> 0-5V
        """

        if instance > -1:
            self.delete_terminus(UTIs['DacWrist'][0], instance)

        data = [1]
        data.append( UTIs['DacWrist'][0] )
        data.append(pronate_DAC)
        data.append(supinate_DAC)
        data.append( int(min_out_DAC*255) )
        data.append( int(max_out_DAC*255))

        self.cmd_movement_engine(data)

    def add_DacElbow(self, instance = -1, bend_DAC = 0, extend_DAC = 1, min_out_DAC = 0, max_out_DAC = 1):
        """
        Add (or overwrite) device to the controller with the following parameters
        
        Parameters
        ----------
        instance: the instance to overwrite, if new device, leave as default
        bend_DAC: int (U8) DAC channel used for open 0-7
        extend_DAC: int (U8) DAC channel used for close 0-7
        min_out_DAC: float Minimum DAC output 0-1 -> 0-5V
        max_out_DAC: float Maximum DAC output 0-1 -> 0-5V
        """

        if instance > -1:
            self.delete_terminus(UTIs['DacElbow'][0], instance)

        data = [1]
        data.append( UTIs['DacElbow'][0] )
        data.append(bend_DAC)
        data.append(extend_DAC)
        data.append( int(min_out_DAC*255) )
        data.append( int(max_out_DAC*255))

        self.cmd_movement_engine(data)

    def add_Taska(self, instance = -1, A_DAC = 0, B_DAC = 1, min_out_DAC = 0, max_out_DAC = 1, dependent = 0):
        """
        Add (or overwrite) device to the controller with the following parameters
        
        Parameters
        ----------
        instance: the instance to overwrite, if new device, leave as default
        A_DAC: int (U8) DAC channel used for open 0-7
        B_DAC: int (U8) DAC channel used for close 0-7
        min_out_DAC: float Minimum DAC output 0-1 -> 0-5V
        max_out_DAC: float Maximum DAC output 0-1 -> 0-5V
        dependent: int (U8) Dependent on Espire elbow; 0=No, 1=Yes, DAC channels will be ignored, rejected if EspireElbow terminus does not exist
        """

        if instance > -1:
            self.delete_terminus(UTIs['Taska'][0], instance)

        data = [1]
        data.append( UTIs['Taska'][0] )
        data.append(A_DAC)
        data.append(B_DAC)
        data.append( int(min_out_DAC*255) )
        data.append( int(max_out_DAC*255))
        data.append(dependent)

        self.cmd_movement_engine(data)

    def add_EspireElbow(self, instance = -1, CAN_baud = 0, min_elbow_speed = 0, max_elbow_speed = 1, min_wrist_speed = 0, max_wrist_speed = 1, min_hand_speed = 0, max_hand_speed = 1):
        """
        Add (or overwrite) device to the controller with the following parameters
        
        Parameters
        ----------
        instance: the instance to overwrite, if new device, leave as default
        CAN_baud: int (U8) CAN baud (0=125k, 1=250k, 2=500k, 3=1000k)
        min_elbow_speed: float Minimum elbow speed 0-1 -> 0-100%
        max_elbow_speed: float Maximum elbow speed 0-1 -> 0-100%
        min_wrist_speed: float Minimum wrist speed 0-1 -> 0-100%
        max_wrist_speed: float Maximum wrist speed 0-1 -> 0-100%
        min_hand_speed: float Minimum hand speed 0-1 -> 0-100%
        max_hand_speed: float Maximum hand speed 0-1 -> 0-100%
        """

        if instance > -1:
            self.delete_terminus(UTIs['EspireElbow'][0], instance)

        data = [1]
        data.append( UTIs['EspireElbow'][0] )
        data.append(CAN_baud)
        data.append( int(min_elbow_speed*255) )
        data.append( int(max_elbow_speed*255) )
        data.append( int(min_wrist_speed*255) )
        data.append( int(max_wrist_speed*255) )
        data.append( int(min_hand_speed*255) )
        data.append( int(max_hand_speed*255) )

        self.cmd_movement_engine(data)

    def add_CovviNexus(self, instance = -1, CAN_baud = 0, A_DAC = 0, B_DAC = 1, min_out_DAC = 0, max_out_DAC = 1, dependent = 0):
        """
        Add (or overwrite) device to the controller with the following parameters
        
        Parameters
        ----------
        instance: the instance to overwrite, if new device, leave as default
        CAN_baud: int (U8) CAN baud (0=125k, 1=250k, 2=500k, 3=1000k)
        A_DAC: int (U8) DAC channel used for open 0-7
        B_DAC: int (U8) DAC channel used for close 0-7
        min_out_DAC: float Minimum DAC output 0-1 -> 0-5V
        max_out_DAC: float Maximum DAC output 0-1 -> 0-5V
        dependent: int (U8) Dependent on Espire elbow; 0=No, 1=Yes, DAC channels will be ignored, rejected if EspireElbow terminus does not exist
        """

        if instance > -1:
            self.delete_terminus(UTIs['CovviNexus'][0], instance)

        data = [1]
        data.append( UTIs['CovviNexus'][0] )
        data.append(CAN_baud)
        data.append(A_DAC)
        data.append(B_DAC)
        data.append( int(min_out_DAC*255) )
        data.append( int(max_out_DAC*255))
        data.append(dependent)

        self.cmd_movement_engine(data)

    def add_IlimbQuantum(self, instance = -1, CAN_baud = 0, min_signal_speed = 0, max_signal_speed = 100, dependent = 0):
        """
        Add (or overwrite) device to the controller with the following parameters
        
        Parameters
        ----------
        instance: the instance to overwrite, if new device, leave as default
        CAN_baud: int (U8) CAN baud (0=125k, 1=250k, 2=500k, 3=1000k)
        min_signal_speed: int (U8) Minimum Signal Speed 0-100
        max_signal_speed: int (U8) Maximum Signal Speed 0-100
        dependent: int (U8) Dependent on Espire elbow; 0=No, 1=Yes, DAC channels will be ignored, rejected if EspireElbow terminus does not exist
        """

        if instance > -1:
            self.delete_terminus(UTIs['IlimbQuantum'][0], instance)

        data = [1]
        data.append( UTIs['IlimbQuantum'][0] )
        data.append(CAN_baud)
        data.append(min_signal_speed)
        data.append(max_signal_speed)
        data.append(dependent)

        self.cmd_movement_engine(data)

    def add_Michelangelo(self, instance = -1, open_DAC = 0, close_DAC = 1):
        """
        Add (or overwrite) device to the controller with the following parameters
        
        Parameters
        ----------
        instance: the instance to overwrite, if new device, leave as default
        open_DAC: int (U8) DAC channel used for open 0-7
        close_DAC: int (U8) DAC channel used for close 0-7
        """

        if instance > -1:
            self.delete_terminus(UTIs['Michelangelo'][0], instance)

        data = [1]
        data.append( UTIs['Michelangelo'][0] )
        data.append(open_DAC)
        data.append(close_DAC)

        self.cmd_movement_engine(data)

    ### ACTION functions
    def get_action_list(self):
        """
        Get a list of current Actions with their linked classifier class
        
        """
        try:
            send = self.get_action_list_send
        except:
            self.get_action_list_send = True
            send = self.get_action_list_send

        if send:
            data = [5]
            self.cmd_movement_engine(data)
            self.get_action_list_send = False
        else:
            return self._get_from_response_queue('GET_ACTION_LIST')

    def add_action(self, UAT, mvmt_class, device_ID, device_instance, instance = -1, grip_ID = -1):
        """
        Add an action. 

        Parameters
        ----------
        UAT: int or string of action (see dictionary of UATs)
        mvmt_class: Classifier class to link to
        device_ID: device unique device ID 
        device_instance: device unique counter number 
        instance: unique counter number 
        grip_ID: Use with AssumeGrip or CloseInGrip. An Action of this type will cause a grip hand to assume a certain grip (and close if CloseInGrip)
        """

        if type(UAT) == str:
            UAT_int = UATs[UAT][0]
            UAT_str = UAT
        else:
            UAT_int = UAT
            for key in UATs:
                if UATs[key][0] == UAT:
                    UAT_str = key
                    break


        if instance > -1:
            self.delete_action(UATs[UAT_str][0], instance)
        else:
            instance = 0

        data = [6]
        data.append( UAT_int )
        data.append( instance )
        data.append( mvmt_class )
        data.append( device_ID )
        data.append( device_instance )

        if grip_ID > -1:
            data.append(grip_ID)

        self.cmd_movement_engine(data)

    def clear_actions(self):
        """
        Remove all terminuses. Clears actions.
        
        """

        data = [9]
        self.cmd_movement_engine(data)
    
    def delete_action(self, UAT, instance):
        """
        Delete an existing action. A reject will be sent if the action doesn’t exist.

        Parameters
        ----------
        UAT: int (U8) device ID
        instance: unique counter number 
        """

        data = [8]
        data.append(UAT)
        data.append(instance)
        self.cmd_movement_engine(data)

    def test_action(self, mvmt_class, speed, duration):
        """
        Cause a specific action with a specific speed for a set amount of time. 
        The classifier will be ignored for the duration. 
        Sending any TEST_ACTION with a duration of 0 will exit any running test. 
        Sending a new TEST_ACTION before the previous one has completed will override it with the new one. 
        This can be used to extend an existing test.

        Parameters
        ----------
        mvmt_class: int (U8) Classifier class of action to test
        speed: float Speed 0-1 -> 0%-100%
        duration: int (U16) Duration (ms)
        """

        data = [10]
        data.append(mvmt_class)
        data.append( int(speed*255) )
        data.append( duration )
        self.cmd_movement_engine(data)

    def suspend_action(self, switch, num_actions, list_of_UAIs):
        """
        Set the suspension status of multiple actions. A suspended action will do not respond to normal movement requests. 
        It will respond to movement requests from the TEST_ACTION command. 
        This setting does not persist through power cycles – such a permanent suspension should be accomplished by deleting the actions.

        The response reflects the state of all actions following the operation, not just those modified by the most recent command. 
        Sending a SUSPEND_ACTIONS with 0 actions may be used to get the status without making any changes.


        Parameters
        ----------
        switch: int (U8) Suspend or allow (0=allow, 1=suspend)
        num_actions: int (U8) Number of actions to suspend/allow
        list_of_UAI: list of ints [UAT of action 0, instances of action 0; ...;UAT of action num_actions, instances of action num_actions ]
        """
        try:
            send = self.suspend_action_send
        except:
            self.suspend_action_send = True
            send = self.suspend_action_send

        if send:
            data = [11]
            data.append(switch)
            data.append(num_actions)
            try:
                flat_list = [item for sublist in list_of_UAIs for item in sublist]
            except: 
                flat_list = list_of_UAIs
            for i in flat_list:
                data.append(i)

            self.cmd_movement_engine(data)
            self.suspend_action_send = False
        else:
            return self._get_from_response_queue('SUSPEND_ACTIONS')

    ### Stream commands
    def stream_data_start(self, data_type, frequency, quantity):
        """
        host-to-controller command used to request the controller stream data in a certain way. 
        It takes type, frequency, and quantity, which tell it what to stream, how often to send a data packet (in milliseconds), and how many to send total. 
        For example, a type of 0x03, a frequency of 5, and a quantity of 200 would send the RMS of all electrodes every 5ms for one second.
        See signal_type dictionary for reference
        
        Parameters
        ----------
        type: int (U8) Uses subtypes for MSG_DATA, 0x00 through 0x05
        frequency: int (U16) Number of milliseconds per data packet
        quantity: int (U16), number of total data packets to send
        """

        if type(data_type) == str:
            data_type_int = signal_type[data_type]
        else:
            for key in signal_type:
                if signal_type[key] == data_type:
                    data_type_int = data_type

        data = []
        data.append(data_type_int)
        data.append(frequency)
        data.append(quantity)

        self.cmd_stream_data(data)

    def stream_data_stop(self):
        """
        Host-to-controller, Stops any in-progress data streaming even if its requested duration has not elapsed.
        
        """

        self.cmd_stop_stream_data()

    ### Adjust electrodes
    def set_gains(self, gains, electrodes = []):
        """
        Sets the gain of the addressed electrode.
        
        Parameters
        ----------
        gains: list of int or int (U16) if single values (non list), sets all electrodes. if list, length of list is 1-8, values 0-7
        electrodes: list of int (U8) if gain is a list, electrodes indicies correspond to gain indicies, values 0-7
        """
        gains_list = [1, 2, 3, 4, 6, 8, 12]
        if type(gains) != list:
            for i in range(8):
                data = [0]
                data.append(i)
                data.append(gains_list[gains-1])
                self.cmd_adjust_electrodes(data)

        else:
            if len(electrodes) != len(gains):
                Warning("The two input lists must be of same length")
                return False
            else:

                for key, val in enumerate(electrodes):
                    data = [0]
                    data.append(val)
                    data.append(gains_list[gains[key]-1])
                    self.cmd_adjust_electrodes(data)            

    def set_electrodes(self, enabled_electrode = [True,True,True,True,True,True,True,True]):
        """
        Sets which electrodes are enabled. Disabled electrodes will not be polled for data and feature extraction will be skipped. 
        Features which consider multiple electrodes will be computed as if disabled electrodes don’t exist – 
        e.g. the mean MAV will be computed as sum of MAVs divided by 6 if only 6 electrodes are enabled.
        
        Parameters
        ----------
        enabled_electrode: list of bool (U8) length of list is 8, index 0 corresponds to electrode 0
        """
        data = [2,0]
        for key, val in enumerate(enabled_electrode):
            if val: data[1] += 2**key
        
        self.cmd_adjust_electrodes(data)

    def get_electrodes(self):
        """
        Gets which electrodes are enabled. 
        """
        try:
            send = self.get_electrodes_send
        except:
            self.get_electrodes_send = True
            send = self.get_electrodes_send

        if send:
            data = [3]

            self.cmd_adjust_electrodes(data)
            self.get_electrodes_send = False
        else:
            return self._get_from_response_queue('GET_ENABLED')

    def set_lead_off(self, enabled = False):
        """
        Disable or enable regular lead-off updates. 
        When enabled, a lead-off MSG_DATA will be transmitted 4 times per second.
        
        Parameters
        ----------
        enabled: list of bool (U8) length of list is 8, index 0 corresponds to electrode 0
        """

        data = [4]
        enabled = 1 if enabled else 0
        data.append(data)
        
        self.cmd_adjust_electrodes(data)

    def get_gains(self):
        """
        Gets which electrodes are enabled. 
        """
        try:
            send = self.get_gains_send
        except:
            self.get_gains_send = True
            send = self.get_gains_send

        if send:
            data = [5]

            self.cmd_adjust_electrodes(data)
            self.get_gains_send = False
        else:
            return self._get_from_response_queue('GET_GAINS')

    def set_feature_window_length(self, length = 200, electrodes = []):
        """
        Sets the length of the time-domain feature extraction buffer for the addressed electrode. 
        Unit is ms, default is 200.
        
        Parameters
        ----------
        length: list of int or int (U16) if single values (non list), sets all electrodes. if list, length of list is 8, index 0 corresponds to electrode 0
        electrodes: list of int (U8) if length is a list, electrodes indicies correspond to length indicies
        """
        
        if type(length) == int:
            for i in range(8):
                data = [6,i, length]
                self.cmd_adjust_electrodes(data)

        else:
            if len(length) != len(electrodes) or len(length) != 8:
                Warning("The two input lists must be of same length")
                return False
            
            for i in range(8):
                data = [6,electrodes[i], length[i]]
                self.cmd_adjust_electrodes(data)

    def get_feature_window_length(self):
        """
        Gets the lengths of the time-domain feature extraction buffers for each electrode.

        """
        try:
            send = self.get_feature_window_length_send
        except:
            self.get_feature_window_length_send = True
            send = self.get_feature_window_length_send

        if send:
            data = [7]
            self.cmd_adjust_electrodes(data)
            self.get_feature_window_length_send = False
        else:
            return self._get_from_response_queue('GET_FEATURE_EXTRACT_BUFF_LEN')

    ### SET_LOGLEVEL
    def set_loglevel(self, channel, level):
        """
        host-to-controller command used to change which plaintext messages are output to the various log channels 
        (currently the debug UART and the Bluetooth connection.) 
        Each channel will ignore messages with a severity lower than that configured by this command. 
        For example, if this command sets the level to Error, then Warning, Info, and Debug level messages won’t appear on that channel. 
        If this command sets the level to None, no messages will appear on that channel.
        
        Parameters
        ----------
        channel: int (U8) 0=UART, 1=Bluetooth
        level: int (U8) 0=None, 1=Critical, 2=Error, 3=Warning, 4=Info, 5=Debug
        """

        data = [channel, level]
        self.cmd_set_loglevel(data)

    ### DEBUG_DRIVER
    def set_DAC(self, channel, value):
        """
        Allows manual configuration of DAC voltages on each channel
        
        Parameters
        ----------
        channel: int (U8) 0-7 – real DAC channel
        value: int (U8) 0-255 – real value to write
        """

        data = [0, channel, value]
        self.cmd_debug_driver(data)

    def set_debug_emg_streaming(self, override = True, step = 0):
        """
        Allows manual configuration of DAC voltages on each channel
        
        Parameters
        ----------
        channel: int (U8) 0-7 – real DAC channel
        value: int (U8) 0-255 – real value to write
        """

        data = [1, override, step]
        self.cmd_debug_driver(data)


    ### CONFIGURATION
    def get_general_fields(self):
        """
        Read the in-memory configuration file’s general fields; 
        these are not the entirety of a configuration, but the fields which are not controlled through other subsystems. 
        Fields marked with [Unvalidated] are not checked for data validity because they are not used by the controller. 
        Fields marked with [Reserved] are ignored.
        
        """
        try:
            send = self.get_general_fields_send
        except:
            self.get_general_fields_send = True
            send = self.get_general_fields_send

        if send:
            data = [0]
            self.cmd_configuration(data)
            self.get_general_fields_send = False
        else:
            return self._get_from_response_queue('GET_GENERAL_FIELDS')

    def set_general_fields(self, mb_style, mb_rev, CAN_baud, amputation_side, amputation_level, filter_region, control_strategy, mirroring_flags):
        """
        Modify the in-memory configuration file’s general fields.
        
        Parameters
        ----------
        mb_style: Motherboard style	[Reserved]
        mb_rev: Motherboard revision	[Reserved]
        CAN_baud: CAN Baud	(0=125k, 1=250k, 2=500k, 3=1000k)
        amputation_side: Amputation side 	[Unvalidated] 0=R, 1=L
        amputation_level: Amputation level	[Unvalidated] 0=Partial hand, 1=Wrist disartic, 2=Transradial, 3=Elbow disartic, 4=Transhumeral, 5=Shoulder disartic
        filter_region: Filter region	0=50hz, 1=60hz
        control_strategy: Control strategy	0=RESCU, 1=Glide
        mirroring_flags: Mirroring flags	For each bit, 1=mirror operations, 0=don’t mirror
                                            Bit 0 (LSB): Hand open/close
                                            Bit 1: Wrist pro/supinate
                                            Bit 2: Elbow bend/extend
        Note that the controller does NOT automatically mirror wrist based on amputation side 

        """

        data = [1, mb_style, mb_rev, CAN_baud, amputation_side, amputation_level, filter_region, control_strategy, mirroring_flags]
        self.cmd_configuration(data)

    def save_to_flash(self):
        """
        Save the entire in-memory configuration file to flash storage. 
        Note that there is no command for restoring from flash, because that would require a soft reset (and happens automatically on all soft resets.) 
        The prosthesis will be halted and Bluetooth will be unresponsive during the operation. 
        Operation will not begin until after the ACK and Accept for this command have been transmitted.
        
        """

        data = [2]
        self.cmd_configuration(data)
    
    def get_general_purpose_map_entry(self):
        """
        Read all entries in the general purpose map, 
        a data structure storing up to 256 key-value pairs that the host may use to store persistent configuration about a prosthesis user, 
        such as their permission to take certain actions in the host app, or what timings the user prefers for calibration sequences. 
        The controller ignores all data in the general purpose map. 
        Entries are reported from lowest to highest key.
        
        """
        try:
            send = self.get_general_purpose_map_entry_send
        except:
            self.get_general_purpose_map_entry_send = True
            send = self.get_general_purpose_map_entry_send

        if send:
            data = [3]
            self.cmd_configuration(data)
            self.get_general_purpose_map_entry_send = False
        else:
            return self._get_from_response_queue('GET_GENERAL_PURPOSE_MAP_ENTRIES')

    def set_general_purpose_map_entry(self, key, value):
        """
        Adds a key-value pair to the general purpose map, or deletes it if the value is 0 (default-value keys are unstored to save space). 
        If the key already exists, its value will be overwritten with the new value. 
        There is 1000 key total maximum enforced on the map, but this does not mean the maximum key is 1000 
        (e.g. you could add a value to key 2000, so long as there are not already 1000 entries in the map.)
        
        Parameters
        ----------
        key: int (U16)
        value: int (U8) (0 deletes entry)
        """

        data = [4, key, value]
        self.cmd_configuration(data)

    def clear_general_purpose_map(self):
        """
        Clears all entries in the general purpose map.
        
        """

        data = [5]
        self.cmd_configuration(data)

    def reset_all_state(self):
        """
        Restores all nonvolatile state to factory default. 
        The flash memory of the configuration will be blanked and the controller will soft reset, 
        causing it to assume default state due to lack of flash state. 
        The Bluetooth connection will be invalid after this command is issued.
        Restores all nonvolatile state to factory default. 
        The flash memory of the configuration will be blanked and the controller will soft reset, 
        causing it to assume default state due to lack of flash state. 
        The Bluetooth connection will be invalid after this command is issued.
        
        """

        data = [6]
        self.cmd_configuration(data)

    def get_firmware_version(self):
        """
        Reads the hardcoded firmware version.
        Version strings conform to SEMVER 2.0.0, where “API” refers to this Bluetooth API. 
        Version strings may have optional prerelease codes following a dash, 
        and will have an automatic hexadecimal build ID hash following a plus sign. 
        Release firmware lacks a prerelease code (i.e. no dash and no text that would follow it.) E.g.:
        “1.2.3-dev+fedcab9876” means the Major version is 1, the Minor version is 2, the Patch is 3, 
        the build is not released (with the prerelease code “dev”), and the unique build hash is 0xFEDCAB9876.
        """
        try:
            send = self.get_firmware_version_send
        except:
            self.get_firmware_version_send = True
            send = self.get_firmware_version_send

        if send:
            data = [7]
            self.cmd_configuration(data)
            self.get_firmware_version_send = False
        else:
            return self._get_from_response_queue('GET_FIRMWARE_VERSION')

    ### Multi processing functions
    def _add_to_rx_communication_queue(self, data):
        """
        Add byte array to RX communication queue
        
        Parameters
        ----------
        data: byte array
            byte array to put into the RX communication queue
        
        """
        existing = []
        if self.rx_communication_queue.qsize() > 0:
            try:
                buffer = self.rx_communication_queue.get( timeout = 1e-3 )
                for i in buffer:
                    existing.append( i )
            except queue.Empty:
                pass
        existing.append(data)
  
        try:
            self.rx_communication_queue.put( existing, timeout = 1e-3 )
        except queue.Full:
            pass

    def _get_from_rx_communication_queue(self):
        """
        Read byte array from RX communication queue

        Returns
        -------
        byte array
            byte array from RX communication queue
        """
        data = []
        while self.rx_communication_queue.qsize() > 0:
            try:
                data.append( self.rx_communication_queue.get( timeout = 1e-3 ) )
            except queue.Empty:
                pass
        return data
    
    def _add_to_rx_interaction_queue(self, data):
        """
        Add byte array to RX interaction queue
        
        Parameters
        ----------
        data: byte array
            byte array to put into the RX interaction queue
        
        """
        try:
            self.rx_interaction_queue.put( data, timeout = 1e-3 )
        except queue.Full:
            pass

    def _get_from_rx_interaction_queue(self):
        """
        Read byte array from RX interaction queue

        Returns
        -------
        byte array
            byte array from RX interaction queue
        """
        data = []
        while self.rx_interaction_queue.qsize() > 0:
            try:
                data.append( self.rx_interaction_queue.get( timeout = 1e-3 ) )
            except queue.Empty:
                pass
        return data

    def _add_to_tx_communication_queue(self, data):
        """
        Add byte array to TX communication queue
        
        Parameters
        ----------
        data: byte array
            byte array to put into the TX communication queue
        
        """
        try:
            self.tx_communication_queue.put( data, timeout = 1e-3 )
        except queue.Full:
            pass

    def _get_from_tx_communication_queue(self):
        """
        Read byte array from TX communication queue

        Returns
        -------
        byte array
            byte array from TX communication queue
        """
        data = []
        while self.tx_communication_queue.qsize() > 0:
            try:
                data.append( self.tx_communication_queue.get( timeout = 1e-3 ) )
            except queue.Empty:
                pass
        return data
    
    def _add_to_tx_interaction_queue(self, data):
        """
        Add byte array to TX interaction queue
        
        Parameters
        ----------
        data: byte array
            byte array to put into the TX interaction queue
        
        """
        existing = []
        if self.tx_interaction_queue.qsize() > 0:
            try:
                buffer = self.tx_interaction_queue.get( timeout = 1e-3 )
                for i in buffer:
                    existing.append( i )
            except queue.Empty:
                pass
        existing.append(data)
  
        try:
            self.tx_interaction_queue.put( existing, timeout = 1e-3 )
        except queue.Full:
            pass

    def _get_from_tx_interaction_queue(self):
        """
        Read byte array from TX interaction queue

        Returns
        -------
        byte array
            byte array from TX interaction queue
        """
        data = []
        while self.tx_interaction_queue.qsize() > 0:
            try:
                data.append( self.tx_interaction_queue.get( timeout = 1e-3 ) )
            except queue.Empty:
                pass
        return data

    def _add_to_response_queue(self, key, data):
        """
        Add dictionary entry to response queue
        
        Parameters
        ----------
        key: key to add dictionary entry to
        data: dictionary entry
        
        """
        
        try:
            response_buffer = self.response_queue.get( timeout = 1e-3 )
        except queue.Empty:
            response_buffer = {}
        
        response_buffer[key] = data
        self.response_queue.put( response_buffer, timeout = 1e-3 )

    def _get_from_response_queue(self, key):
        """
        Read dictionary entry to response queue

        Parameters
        ----------
        key: key to read dictionary entry from
        
        Returns
        -------
        res: dictionary entry or False
        """
        
        try:
            response_buffer = self.response_queue.get( timeout = 1e-3 )
        
            try:
                res = dc(response_buffer[key])

                try:
                    if type(res[ list( res.keys() )[0]]) == list:
                        
                        if list( res.keys() )[0] != 'TYPE':
                            return res
                        
                        
                        if len(res[ list( res.keys() )[0]]) > 1:
                            
                            temp_dict = {}

                            for key_iter in res.keys():
                                temp_dict[key_iter] = dc(res[key_iter][0])
                                del response_buffer[key][key_iter][0]
                                
                        else:
                            temp_dict = {}

                            for key_iter in res.keys():
                                temp_dict[key_iter] = dc(res[key_iter][0])
                                
                            del response_buffer[key]
                            
                        self.response_queue.put( response_buffer, timeout = 1e-3 )

                        return temp_dict
                    else:
                        return res
                except:
                    return res
            except Exception as e:
                self.response_queue.put( response_buffer, timeout = 1e-3 )
                return None

        except queue.Empty:
            pass

    def _get_from_response_queue2(self, key):
        """
        Read dictionary entry to response queue

        Parameters
        ----------
        key: key to read dictionary entry from
        
        Returns
        -------
        res: dictionary entry or False
        """
        
        try:
            response_buffer = self.response_queue.get( timeout = 1e-3 )
        
            try:
                res = dc(response_buffer[key])

                if len(res[ list( res.keys() )[0]]) > 1:
                    temp_dict = {}

                    for key_iter in res.keys():
                        temp_dict[key_iter] = dc(res[key_iter][0])
                        del response_buffer[key][key_iter][0]
                        
                else:
                    temp_dict = {}

                    for key_iter in res.keys():
                        temp_dict[key_iter] = dc(res[key_iter][0])
                        
                    del response_buffer[key]
                    

                self.response_queue.put( response_buffer, timeout = 1e-3 )

                return temp_dict
            except Exception as e:
                self.response_queue.put( response_buffer, timeout = 1e-3 )
                return None

        except queue.Empty:
            pass
    
    def _get_response_queue(self, key):
        """
        Read dictionary entry to response queue

        Parameters
        ----------
        key: key to read dictionary entry from
        
        Returns
        -------
        res: dictionary entry or False
        """
        
        try:
            response_buffer = self.response_queue.get( timeout = 1e-3 )
        
            try:
                res = response_buffer[key]

                return res

            except Exception as e:
                self.response_queue.put( response_buffer, timeout = 1e-3 )
                return None

        except queue.Empty:
            pass

    def _add_ack_queue(self, cmd):
        """
        Add dictionary entry to response queue
        
        Parameters
        ----------
        key: key to add dictionary entry to
        data: dictionary entry
        
        """
        try:
            ack_buffer = self.ack_queue.get( timeout = 1e-3 )
            ack_buffer.append(cmd[0])
            self.ack_queue.put( ack_buffer, timeout = 1e-3 )
        except:
            self.ack_queue.put( [cmd[0]], timeout = 1e-3 )
    
    def check_ack(self, cmd):
        """
        Add dictionary entry to response queue
        
        Parameters
        ----------
        key: key to add dictionary entry to
        data: dictionary entry
        
        """
        try:
            ack_buffer = self.ack_queue.get( timeout = 1e-3 )
            try:
                if cmd in ack_buffer:
                    del ack_buffer[ack_buffer.index(cmd)]
                    return True
                else:
                    return False
            except:
                return False
        except queue.Empty:
            return False

    def run_test(self, DummyData):

        '''self.key_recursion(DummyData)
        time.sleep(2)'''

        '''# Test filter
        for key in Test_data['Add_filter']:
            self.add_filter_data(key, Test_data['Add_filter'][key])

        for key in Test_data['Modify_filter']:
            self.set_filter_data(key, 0, Test_data['Modify_filter'][key])
        
        
        # Test RLDA
        for i in range(self.num_classes-1):
            try:
                Test_data['RLDA']['RLDA'] = np.vstack( (Test_data['RLDA']['RLDA'], np.ones(288)*( (i+1) * 0.1) ) )
            except:
                Test_data['RLDA']['RLDA'] = np.ones(288)*( (i+1) * 0.1 )

        for i in range(self.num_classes):
            try:
                Test_data['RLDA']['Means'] = np.vstack( (Test_data['RLDA']['Means'], np.ones(2)*( (i+1) * 0.1) ) )
            except:
                Test_data['RLDA']['Means'] = np.ones(2)*( (i+1) * 0.1)

        self.replace_RLDA_all(Test_data['RLDA']['RLDA'], Test_data['RLDA']['Means'])
        self.edit_RLDA_axis(axis_number=1, start_cluster=0, end_cluster=1, boundary=[0.1,0.9], class_indicies=[0,1,1], proportional_ramp_start=0, proportional_ramp_stop=1, min_proportional_speed=0.11, max_proportional_speed=0.92)
        # Test CMD_DATA_MONITOR
        input_data = self.get_test_data(Test_data['CMD_DATA_MONITOR']['EDIT_CACHE'])
        self.edit_segmenter_cache(input_data[0], input_data[1], input_data[2], input_data[3], input_data[4])

        self.run_temporary_cache(Test_data['CMD_DATA_MONITOR']['RUN_TEMPORARY_CACHE'])

        self.view_cache_summary(Test_data['CMD_DATA_MONITOR']['VIEW_CACHE_SUMMARY'])

        input_data = self.get_test_data(Test_data['CMD_DATA_MONITOR']['VIEW_SEGMENT_DETAILS'])
        self.view_segment_details(input_data[0], input_data[1])

        input_data = self.get_test_data(Test_data['CMD_DATA_MONITOR']['GET_SEGMENT_FEATURE_VECTORS'])
        self.get_segment_feature_vectors(input_data[0], input_data[1])

        input_data = self.get_test_data(Test_data['CMD_DATA_MONITOR']['REQUEST_EVENTS'])
        self.get_segment_events(input_data[0], input_data[1], input_data[2], input_data[3])

        # Test CMD_CONFIGURE_DAUGHTERBOARD
        self.get_peripherals()
        self.set_style(1)

        # Test CMD_MOVEMENT_ENGINE
        self.add_SimpleHand(min_out_DAC= 0.1, max_out_DAC= 0.9) 
        self.add_MotorDrivenWrist(instance=0, min_out_PWN=0.2, max_out_PWN=0.8, dependent=0)

        self.add_action('BasicOpen', 1, 1, 0)
        self.add_action(3, 2, 1, 0, 0)
        self.add_action('CloseInGrip', 3, 1, 0, 0, 2)

        self.test_action(3,0.6,1000)
        self.suspend_action(0,2,[[2,0],[3,0]])'''

        '''# Test CMD_STREAM_DATA
        self.stream_data_start(data_type=1,frequency=10,quantity=250)
        while True:
            data = self.get_emg_data()
            if data is not None:
                print(data)'''

        '''# Test CMD_ADJUST_ELECTRODES
        self.set_gains( gains=6)
        self.set_gains( gains=[6,6,6], electrodes=[0,4,8])

        self.set_electrodes( enabled_electrode = [True,True,False,True,True,False,True,True] )

        self.set_feature_window_length(length=150)
        self.set_feature_window_length(length=[150, 200, 250], electrodes=[0, 4, 6])

        # Test CMD_DEBUG_DRIVER
        self.set_DAC(4,122)

        # Test CMD_CONFIGURATION
        self.set_general_fields(1,2,3,4,4,0,0,0)
        self.set_general_purpose_map_entry(1500, 122)

        res = None
        typelist = ['RAW_EMG', 'FILTERED_EMG', 'MAV_EMG', 'ENVELOPE_EMG', 'TD5_EMG', 'FULL_EMG']
        while res is None:
            res = self.get_emg_data()
        print("DATA VALIDATION GET_EMG_DATA")
        for i, k in enumerate(typelist):
            vector = DummyData['MSG_DATA'][k]['data'][1:-1]

            for key, val in enumerate(res['DATA'][i]):
                res['DATA'][i][key] = round(val * 10000) / 10000
            for key, val in enumerate(vector):
                vector[key] = round(val * 10000) / 10000

            if res['DATA'][i] == vector:
                print(k, "   SUCCESS")
            else:
                print(k, "   FAILED", res['DATA'][i], vector)

        res = None
        while res is None:
            res = self.get_filter_activity()
        print("DATA VALIDATION FILTERING_HISTORY")
        temp = DummyData['MSG_DATA']['FILTERING_HISTORY']['data']
        for i, key in enumerate(res):
            temp_bool = res[key] == temp[i]
            if not temp_bool:
                break
        if temp_bool:
            print("   SUCCESS")
        else:
            print("   FAILED", res, temp)

        res = None
        while res is None:
            res = self.request_time_data()
        print("DATA VALIDATION CMD_REQUEST_TIME")
        if res == DummyData['CMD_REQUEST_TIME']['RESPONSE']['data'][0]:
            print("   SUCCESS")
        else:
            print("   FAILED", res, DummyData['CMD_REQUEST_TIME']['RESPONSE']['data'][0])

        res = None
        while res is None:
            res = self.get_filter_list()
        print("DATA VALIDATION GET_FILTER_LIST")
        temp = DummyData['CMD_EDIT_CONTROL_STRATEGY']['GET_FILTER_LIST']['data']
        try: 
            if res[self.UFT[temp[1]]] == temp[2]:
                print("   SUCCESS")
            else:
                print("   FAILED", res, temp)
        except:
            
            print("   FAILED", res, temp)
        
        res = None
        while res is None:
            res = self.get_preview_order()
        print("DATA VALIDATION GET_PREVIEW_ORDER")
        temp = DummyData['CMD_EDIT_CONTROL_STRATEGY']['GET_PREVIEW_ORDER']['data']
        try: 
            if res[self.UFT[temp[1]]] == temp[2] and res[self.UFT[temp[3]]] == temp[4]:
                print("   SUCCESS")
            else:
                print("   FAILED", res, temp)
        except:
            
            print("   FAILED", res, temp)
        
        res = None
        while res is None:
            res = self.get_process_order()
        print("DATA VALIDATION GET_PROCESS_ORDER")
        temp = DummyData['CMD_EDIT_CONTROL_STRATEGY']['GET_PROCESS_ORDER']['data']
        try: 
            if res[self.UFT[temp[1]]] == temp[2] and res[self.UFT[temp[3]]] == temp[4]:
                print("   SUCCESS")
            else:
                print("   FAILED", res, temp)
        except:
            
            print("   FAILED", res, temp)
        
        res = None
        while res is None:
            res = self.view_cache_summary(0)
        print("DATA VALIDATION VIEW_CACHE_SUMMARY")
        temp = DummyData['CMD_DATA_MONITOR']['VIEW_CACHE_SUMMARY']['data']
        if res['Cache being summarized'] == temp[0] and res['Number of segments'] == temp[1] and res['Timestamps of segments'] == temp[2:]:
            print("   SUCCESS")
        else:
            print("   FAILED", res, temp)
        
        res = None
        while res is None:
            res = self.view_segment_details(0, 1)
        print("DATA VALIDATION VIEW_SEGMENT_DETAILS")
        temp = DummyData['CMD_DATA_MONITOR']['VIEW_SEGMENT_DETAILS']['data']
        if res['Cache being viewed'] == temp[0] and res['Timestamp of segment being viewed'] == temp[1] and res['Length (ms) of segment'] == temp[2] and res['Most common class label of segment'] == temp[3]:
            print("   SUCCESS")
        else:
            print("   FAILED", res, temp)
        
        res = None
        while res is None:
            res = self.get_segment_feature_vectors(0, 1)
        print("DATA VALIDATION GET_SEGMENT_FEATURE_VECTORS")
        temp = DummyData['CMD_DATA_MONITOR']['GET_SEGMENT_FEATURE_VECTORS_VECTOR']['data']
        if temp[0] == res[0]['Response type'] and temp[1] == res[0]['Cache to view'] and temp[2] == res[0]['Timestamp of segment to view'] and temp[3] == res[0]['Number of milliseconds into segment of this vector']:
            temp_1 = True
        else:
            temp_1 = False
        temp = temp[4:]
        vector = res[0]['Feature vector']
        for key, val in enumerate(vector):
            vector[key] = round(val * 10000) / 10000
        for key, val in enumerate(temp):
            temp[key] = round(val * 10000) / 10000
        if temp_1 and vector == temp:
            print("   SUCCESS")
        else:
            print("   FAILED", res, temp)
        
        res = None
        while res is None:
            res = self.get_segment_events(0, 1, 0, 1)
        print("DATA VALIDATION GET_SEGMENT_EVENTS SEGMENTATION_ONSET")
        temp = DummyData['MSG_EVENT']['SEGMENTATION_ONSET']['data']
        if res['SEGMENTATION_RESPONSIBLE'] == temp[1] and res['TIMESTAMP'] == temp[2]:
            print("   SUCCESS")
        else:
            print("   FAILED", res, temp)
        
        res = None
        while res is None:
            res = self.get_segment_events(0, 1, 0, 1)
        print("DATA VALIDATION GET_SEGMENT_EVENTS SEGMENT_SAVED")
        temp = DummyData['MSG_EVENT']['SEGMENT_SAVED']['data']
        if res['SEGMENTATION_RESPONSIBLE'] == temp[1] and res['TIMESTAMP'] == temp[2]:
            print("   SUCCESS")
        else:
            print("   FAILED", res, temp)
        
        res = None
        while res is None:
            res = self.get_peripherals()
        print("DATA VALIDATION GET_PERIPHERALS")
        temp = DummyData['CMD_CONFIGURE_DAUGHTERBOARD']['GET_PERIPHERALS']['data']
        for i, key in enumerate(res):
            temp_bool = res[key] == temp[i]
            if not temp_bool:
                break
        if temp_bool:
            print("   SUCCESS")
        else:
            print("   FAILED", res, temp)
        
        res = None
        while res is None:
            res = self.get_terminus_list()
        print("DATA VALIDATION GET_TERMINUS_LIST")
        temp = DummyData['CMD_MOVEMENT_ENGINE']['GET_TERMINUS_LIST']['data']
        temp_bool = True
        if temp_bool and res['Number of terminuses'] != temp[0]:
            temp_bool = False

        k, _ = list(res.items())[1]
        if temp_bool and UTIs[k][0] != temp[1]:
            temp_bool = False
        if temp_bool and (res[k]['instance'] != temp[2] or res[k]['status'] != temp[-2]):
            temp_bool = False
        k, _ = list(res.items())[2]
        if temp_bool and UTIs[k][0] != temp[3]:
            temp_bool = False
        if temp_bool and (res[k]['instance'] != temp[4] or res[k]['status'] != temp[-1]):
            temp_bool = False

        if temp_bool:
            print("   SUCCESS")
        else:
            print("   FAILED", res, temp)
        
        res = None
        while res is None:
            res = self.get_action_list()
        print("DATA VALIDATION GET_ACTION_LIST")
        temp = DummyData['CMD_MOVEMENT_ENGINE']['GET_ACTION_LIST']['data']
        temp_bool = True
        if temp_bool and res['Number of actions'] != temp[0]:
            temp_bool = False

        k, _ = list(res.items())[1]
        if temp_bool and UATs[k][0] != temp[1]:
            temp_bool = False
        if temp_bool and (res[k]['instance'] != temp[2] or res[k]['class'] != temp[-2]):
            temp_bool = False
        k, _ = list(res.items())[2]
        if temp_bool and UATs[k][0] != temp[3]:
            temp_bool = False
        if temp_bool and (res[k]['instance'] != temp[4] or res[k]['class'] != temp[-1]):
            temp_bool = False

        if temp_bool:
            print("   SUCCESS")
        else:
            print("   FAILED", res, temp)
        
        res = None
        while res is None:
            res = self.suspend_action(0, 2, [1, 2])
        print("DATA VALIDATION SUSPEND_ACTIONS")
        temp = DummyData['CMD_MOVEMENT_ENGINE']['SUSPEND_ACTIONS']['data']
        temp_bool = True
        if temp_bool and res['Number of actions suspended'] != temp[0]:
            temp_bool = False

        k, _ = list(res.items())[1]
        if temp_bool and UATs[k][0] != temp[1]:
            temp_bool = False
        if temp_bool and res[k]['instance'] != temp[2]:
            temp_bool = False
        k, _ = list(res.items())[2]
        if temp_bool and UATs[k][0] != temp[3]:
            temp_bool = False
        if temp_bool and res[k]['instance'] != temp[4]:
            temp_bool = False

        if temp_bool:
            print("   SUCCESS")
        else:
            print("   FAILED", res, temp)
        
        res = None
        while res is None:
            res = self.get_electrodes()
        print("DATA VALIDATION GET_ENABLED")
        temp = DummyData['CMD_ADJUST_ELECTRODES']['GET_ENABLED']['data']
        res_ = {}
        temp_ = []
        val = bin(temp[0])
        for key, i in enumerate(val):
            if key > 1:
                temp_.append( int(i) )
        res_['Electrodes enabled'] = list(reversed(temp_))

        temp = temp[1:]
        temp_ = []
        val = bin(temp[0])
        for key, i in enumerate(val):
            if key > 1:
                temp_.append( int(i) )
        res_['Electrodes operational'] = list(reversed(temp_))
        temp_bool = True
        for i, key in enumerate(res):
            temp_bool = res[key] == res_[key]
            if not temp_bool:
                break
        if temp_bool:
            print("   SUCCESS")
        else:
            print("   FAILED", res, res_)
        
        res = None
        while res is None:
            res = self.get_gains()
        print("DATA VALIDATION GET_GAINS")
        temp = DummyData['CMD_ADJUST_ELECTRODES']['GET_GAINS']['data']
        if temp == res['Gain']:
            print("   SUCCESS")
        else:
            print("   FAILED", res, temp)
        
        res = None
        while res is None:
            res = self.get_feature_window_length()
        print("DATA VALIDATION GET_FEATURE_EXTRACT_BUFF_LEN")
        temp = DummyData['CMD_ADJUST_ELECTRODES']['GET_FEATURE_EXTRACT_BUFF_LEN']['data']
        if temp == res['Length of buffer for electrode']:
            print("   SUCCESS")
        else:
            print("   FAILED", res, temp)
        
        res = None
        while res is None:
            res = self.get_general_fields()
        print("DATA VALIDATION GET_GENERAL_FIELDS")
        temp = DummyData['CMD_CONFIGURATION']['GET_GENERAL_FIELDS']['data']
        for i, key in enumerate(res):
            temp_bool = res[key] == temp[i]
            if not temp_bool:
                break
        if temp_bool:
            print("   SUCCESS")
        else:
            print("   FAILED", res, temp)
        
        res = None
        while res is None:
            res = self.get_general_purpose_map_entry()
        print("DATA VALIDATION GET_GENERAL_PURPOSE_MAP_ENTRIES")
        temp = DummyData['CMD_CONFIGURATION']['GET_GENERAL_PURPOSE_MAP_ENTRIES']['data']
        temp_bool = True
        if res['Number of entries'] != temp[0]:
            temp_bool = False
        del res['Number of entries']
        temp = temp[1:]
        for i, key in enumerate(res):
            if key != temp[i*2]:
                temp_bool = False
                break
            if res[key] != temp[i*2+1]:
                temp_bool = False
                break
        if temp_bool:
            print("   SUCCESS")
        else:
            print("   FAILED", res, temp)
        
        res = None
        while res is None:
            res = self.get_firmware_version()
        print("DATA VALIDATION GET_FIRMWARE_VERSION")
        temp = DummyData['CMD_CONFIGURATION']['GET_FIRMWARE_VERSION']['data']
        temp_bool = True
        if res['Version string length'] != temp[0]:
            temp_bool = False
        for i, key in enumerate(res['Version string characters']):
            temp_bool = key == temp[i+1]
            if not temp_bool:
                break
        if temp_bool:
            print("   SUCCESS")
        else:
            print("   FAILED", res, temp)'''

            
    def get_test_data(self, data):
        input_data = []
        for i in data:
            input_data.append(i)
        return input_data
    
    def print_test(self, ut_data, res, module1, module2 = ''):
        if module2 == '':
            print( "Data conversion for ", module1," is:  ")
        else:
            print( "Data conversion for ", module1, "in module", module2," is:  ")
        
        for key, i in enumerate(res):
            res[key] = round(i*100)/100
        for key, i in enumerate(ut_data):
            ut_data[key] = round(i*100)/100

        if ut_data == res:
            print("SUCCESSFUL")
        else:
            print( "--from:  ", ut_data)
            print( "--to:    ", res)

    def add_float(self, data, x):
        if type(x) != list: x = [x]
        for i in x:
            data.extend(bytearray(struct.pack("f", float( i ))))
        return data

    def add_q015(self, data, x):
        if type(x) != list: x = [x]
        for i in x:
            data.extend(bytearray(struct.pack('<H', np.int16(i*32767))))
        return data

    def add_u32(self, data, x):
        if type(x) != list: x = [x]
        for i in x:
            h = bytearray(struct.pack('<H', np.int32(i)))
            c = len(h)
            while c < 4:
                h.extend(b'\x00')
                c += 1
            data.extend(h)
        return data

    def add_u16(self, data, x):
        if type(x) != list: x = [x]
        for i in x:
            data.extend(bytearray(struct.pack('<H', np.int16(i))))
        return data

    def add_u8(self, data, x):
        if type(x) != list: x = [x]
        for i in x:
            data.append(i)

        return data
    
    def parse_float(self, data, x):

        for i in range(0,len(x), 4):
            float_data = x[i:i + 4]
            data.append(struct.unpack('<f', float_data)[0])        
        return data

    def parse_q015(self, data, x):

        for i in range(0,len(x), 2):
            int_data = x[i : i+2]
            int_data.extend(bytearray(b'\x00\x00'))
            data.append(struct.unpack('<i', int_data)[0] / 32768)
        
        return data

    def parse_u32(self, data, x):
        for i in range(0,len(x), 4):
            int_data = x[i : i+4]
            data.append(struct.unpack('<i', int_data)[0])
        return data

    def parse_u16(self, data, x):
        for i in range(0,len(x), 2):
            int_data = x[i : i+2]
            int_data.extend(bytearray(b'\x00\x00'))
            data.append(struct.unpack('<i', int_data)[0])
        
        return data

    def parse_u8(self, data, x):
        for i in x:
            data.append(i)
        return data

    def flush(self):
        """
        Dispose of all previous data
        """
        self._flush_completed.clear()

        empty = False
        while not empty:
            try: self.response_queue.get( timeout = 1e-3 )
            except queue.Empty: empty = True
        empty = False
        while not empty:
            try: self.rx_communication_queue.get( timeout = 1e-3 )
            except queue.Empty: empty = True
        empty = False
        while not empty:
            try: self.tx_communication_queue.get( timeout = 1e-3 )
            except queue.Empty: empty = True
        empty = False
        while not empty:
            try: self.rx_interaction_queue.get( timeout = 1e-3 )
            except queue.Empty: empty = True
        empty = False
        while not empty:
            try: self.tx_interaction_queue.get( timeout = 1e-3 )
            except queue.Empty: empty = True
        empty = False
        while not empty:
            try: self.ack_queue.get( timeout = 1e-3 )
            except queue.Empty: empty = True

        self._flush_completed.set()

    @property
    def flush_completed(self):

        if self._flush_completed.is_set():
            return True
        else:
            return False

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

    def close(self):

        self.cmd_disconnect()

        self.core2driver.close()

        self._tx_exit_event.set()
        self._rx_exit_event.set()
        self._record_exit_event.set()

        self.flush()

        t = time.time()

        self._connection_process.join( timeout=5 )
        self._rx_process.join( timeout=5 )
        self._tx_process.join( timeout=5 )

        print('Core2 Controller process terminated in ', time.time()-t,'s')
        
    def key_recursion(self, d):

        try:
            self._add_to_rx_communication_queue(d['bytes'])
            time.sleep(0.1) 

        except:

            if len(d.keys()) > 0:
                for key in d:
                    self.key_recursion(d[key])

if __name__ == '__main__':
    import time
    import pickle


    with open('Core2DummyData.pkl', 'rb') as handle:
        DummyData = pickle.load(handle)

    core2 = Core2Adapt() 

    while not core2._connection_event.is_set():
        pass

    core2.run_test(DummyData)

    time.sleep(2) 

    time.sleep(2)
    core2.close()