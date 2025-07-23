import pickle
import struct
import numpy as np

MSG_DATA = 0x05
RAW_EMG = 0x00

def key_recursion(d):

    try:
        d['bytes'] = tx_packaging(d['bytes'])
        return d

    except:

        if len(d.keys()) > 0:
            for key in d:
                d[key] = key_recursion(d[key])

        return d

def tx_packaging(data):
    """
    Construct finalized byte array from mp queue and send to communication process mp queue
    """

    if len(data) > 0:

        tx_data = bytearray()
        tx_data.extend( b'\xff' )

        length = struct.pack('<H', np.int16(len(data) + 1))

        tx_data.extend( length )
        
        tx_data.extend( data )

        CRC = 0
        for b in tx_data:
            CRC += b 

        CRC %= 256

        tx_data.append( CRC )

        return tx_data

def add_float(data, num, rep = 1):
    for i in range(1,num+1):
        for j in range(rep):
            data['bytes'].extend(bytearray(struct.pack("f", float(i))))
            data['data'].append(float(i))


    return data

def add_float_norm(data, num, rep = 1):
    for i in range(1,num+1):
        for j in range(rep):
            data['bytes'].extend(bytearray(struct.pack("f", float( (i*0.1)/num ))))
            data['data'].append(float( (i*0.1)/num ))


    return data

def add_q015(data, num, rep = 1):
    for i in range(1,num+1):
        for j in range(rep):
            data['bytes'].extend(bytearray(struct.pack('<H', np.int16(i * 0.0001 *32768))))
            data['data'].append(i * 0.0001)

    return data

def add_u32(data, num, rep = 1):
    for i in range(1,num+1):
        for j in range(rep):
            h = bytearray(struct.pack('<H', np.int32(i)))
            c = len(h)
            while c < 4:
                h.extend(b'\x00')
                c += 1
            data['bytes'].extend(h)
            data['data'].append(np.int32(i))

    return data

def add_u24(data, num, rep = 1, val = False):
    for i in range(1,num+1):
        for j in range(rep):
            if val:
                h = bytearray(struct.pack('<H', np.int32(val)))
                data['data'].append(np.int32(val))
            else:
                h = bytearray(struct.pack('<H', np.int32(i)))
                data['data'].append(np.int32(i))
            c = len(h)
            while c < 3:
                h.extend(b'\x00')
                c += 1
            data['bytes'].extend(h)

    return data

def add_u16(data, num, rep = 1, val = False):
    for i in range(1,num+1):
        for j in range(rep):
            if val:
                data['bytes'].extend(bytearray(struct.pack('<H', np.int16(val))))
                data['data'].append(np.int16(val))
            else:
                data['bytes'].extend(bytearray(struct.pack('<H', np.int16(i))))
                data['data'].append(np.int16(i))

    return data

def add_u8(data, num, rep = 1, val = False):
    for i in range(1,num+1):
        for j in range(rep):
            if val:
                data['bytes'].append(val)
                data['data'].append(val)
            else:
                data['bytes'].append(i)
                data['data'].append(i)

    return data

if __name__ == '__main__':

    import binascii

    filter_data = {    'UnanimousVoting' : { 'Default' : {}},
                                'OnsetThreshold' : { 'Threshold': 0, 'Latching' : True },
                                'ProportionalControl' : {   'Reference_MMAV': 0, 
                                                            'General': {'Min_Speed_Multiplier': 0, 'Max_Speed_Multiplier': 0, 'MMAV_ratio_lower': 0, 'MMAV_ratio_lower': 0}
                                                        },
                                'VelocityRamp' : { 'Ramp_Length' : 0, 'Increment' : 0, 'Decrement' : 0,'Min_Speed_Multiplier' : 0,'Max_Speed_Multiplier' : 0 },    
        }

    filter_data['UnanimousVoting']['Default'][4] = 5
    filter_data['UnanimousVoting']['Default'][1] = 5
    print(max(filter_data['UnanimousVoting']['Default'].keys()))

    DummyData = {   

        'MSG_DATA' :            {
                                    'RAW_EMG' :             { 'bytes': bytearray(b'\x05\x00\x01\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x05\x00\x00\x00\x06\x00\x00\x00\x07\x00\x00\x00\x08\x00\x00\x00\x02'),
                                                              'data': [1,1,2,3,4,5,6,7,8,2]
                                                            },

                                    'FILTERED_EMG' :        { 'bytes': bytearray(b'\x05\x01\x01'),
                                                              'data': [1,]
                                                            },
                                    'MAV_EMG' :             { 'bytes': bytearray(b'\x05\x02\x01'),
                                                              'data': [1,]
                                                            },
                                    'ENVELOPE_EMG' :        { 'bytes': bytearray(b'\x05\x03\x01'),
                                                              'data': [1,]
                                                            },
                                    'TD5_EMG' :             { 'bytes': bytearray(b'\x05\x04\x01'),
                                                              'data': [1,]
                                                            },
                                    'FULL_EMG' :            { 'bytes': bytearray(b'\x05\x05\x01'),
                                                              'data': [1,]
                                                            },
                                    #'LEADOFF_INFO' :        bytearray(b'\x05\x08'),
                                    'FILTERING_HISTORY' :   { 'bytes': bytearray(b'\x05\x0A'),
                                                              'data': []
                                                            },
                                    
                                    },

        'MSG_PLAINTEXT' :                                   { 'bytes': bytearray(b'\x06\x46\x61\x53\x7A'),
                                                              'data': [1,70,97,83,122]
                                                            },

        'MSG_EVENT' :           {
                                   'SEGMENTATION_ONSET' :   { 'bytes': bytearray(b'\x08\x00\x00\x00\x00\x00\x80'),
                                                              'data': [0,0,128]
                                                            },
                                   'SEGMENT_SAVED' :        { 'bytes': bytearray(b'\x08\x01\x00\x00\x00\x00\x88'),
                                                              'data': [1,0,136]
                                                            },
                                    },

        'CMD_REQUEST_TIME':     {
                                    'RESPONSE' :            { 'bytes': bytearray(b'\x07\x82\x00\x00\x00\x80'),
                                                              'data': [128]
                                                            }
                                },

        'CMD_EDIT_CONTROL_STRATEGY':{
                                    'GET_FILTER_LIST':          {   'bytes': bytearray(b'\x07\x83\x03\x01\x01\x00'),
                                                                    'data': [1,1,0]
                                                                    },
                                    'GET_PREVIEW_ORDER':        {   'bytes': bytearray(b'\x07\x83\x08\x02\x01\x00\x02\x05'),
                                                                    'data': [2,1,0,2,5]
                                                                    },
                                    'GET_PROCESS_ORDER':        {   'bytes': bytearray(b'\x07\x83\x0A\x02\x01\x00\x02\x05'),
                                                                    'data': [2,1,0,2,5]
                                                                    }        
                                    },

        'CMD_DATA_MONITOR':{
                                    'VIEW_CACHE_SUMMARY' :           {  'bytes': bytearray(b'\x07\x84\x02\x00\x02'),
                                                                        'data': [0,2]
                                                                        },
                                    'VIEW_SEGMENT_DETAILS' :           {    'bytes': bytearray(b'\x07\x84\x03\x00'),
                                                                            'data': [0]
                                                                            },
                                    'GET_SEGMENT_FEATURE_VECTORS_START' :           {   'bytes': bytearray(b'\x07\x84\x04\x00\x00'),
                                                                                        'data': [0,0]
                                                                                        },
                                    'GET_SEGMENT_FEATURE_VECTORS_VECTOR' :           {  'bytes': bytearray(b'\x07\x84\x04\x01\x00'),
                                                                                        'data': [1,0]
                                                                                        },
                                    'GET_SEGMENT_FEATURE_VECTORS_STOP' :           {    'bytes': bytearray(b'\x07\x84\x04\x02\x00'),
                                                                                        'data': [2,0]
                                                                                        }
                                    },


        'CMD_CONFIGURE_DAUGHTERBOARD':{
                                    'GET_PERIPHERALS' :           { 'bytes': bytearray(b'\x07\x85\x00'),
                                                                    'data': []
                                                                    }
                                    },

        'CMD_MOVEMENT_ENGINE':{
                                    'GET_TERMINUS_LIST' :           {   'bytes': bytearray(b'\x07\x86\x00\x02'),
                                                                        'data': [2]
                                                                        },
                                    'GET_ACTION_LIST' :           {   'bytes': bytearray(b'\x07\x86\x05\x02'),
                                                                        'data': [2]
                                                                        },
                                    'SUSPEND_ACTIONS' :           {   'bytes': bytearray(b'\x07\x86\x0B\x02'),
                                                                        'data': [2]
                                                                        }
                                    },

        'CMD_ADJUST_ELECTRODES':{
                                    'GET_ENABLED' :           { 'bytes': bytearray(b'\x07\x8E\x03'),
                                                                'data': []
                                                                },
                                    'GET_GAINS' :           {   'bytes': bytearray(b'\x07\x8E\x05'),
                                                                'data': []
                                                                },
                                    'GET_FEATURE_EXTRACT_BUFF_LEN' :           {    'bytes': bytearray(b'\x07\x8E\x07'),
                                                                                    'data': []
                                                                                    }
                                    },

        'CMD_CONFIGURATION':{
                                    'GET_GENERAL_FIELDS' :    { 'bytes': bytearray(b'\x07\x91\x00'),
                                                                'data': []
                                                                },
                                    'GET_GENERAL_PURPOSE_MAP_ENTRIES' :   { 'bytes': bytearray(b'\x07\x91\x03\x03'),
                                                                            'data': [3]
                                                                            },
                                    'GET_FIRMWARE_VERSION' :          { 'bytes': bytearray(b'\x07\x91\x07\x04'),
                                                                        'data': [4]
                                                                        }
                                    }


    }
    # MSG_DATA
    DummyData['MSG_DATA']['FILTERED_EMG'] = add_q015(DummyData['MSG_DATA']['FILTERED_EMG'], 8)
    DummyData['MSG_DATA']['FILTERED_EMG'] = add_u8(DummyData['MSG_DATA']['FILTERED_EMG'], 1, val=2)

    DummyData['MSG_DATA']['MAV_EMG'] = add_q015(DummyData['MSG_DATA']['MAV_EMG'], 8)
    DummyData['MSG_DATA']['MAV_EMG'] = add_u8(DummyData['MSG_DATA']['MAV_EMG'], 1, val=2)

    DummyData['MSG_DATA']['ENVELOPE_EMG'] = add_q015(DummyData['MSG_DATA']['ENVELOPE_EMG'], 8)
    DummyData['MSG_DATA']['ENVELOPE_EMG'] = add_u8(DummyData['MSG_DATA']['ENVELOPE_EMG'], 1, val=2)

    for i in range(8):
        DummyData['MSG_DATA']['TD5_EMG'] = add_q015(DummyData['MSG_DATA']['TD5_EMG'], 5)
    DummyData['MSG_DATA']['TD5_EMG'] = add_u8(DummyData['MSG_DATA']['TD5_EMG'], 1, val=2)

    for i in range(8):
        DummyData['MSG_DATA']['FULL_EMG'] = add_q015(DummyData['MSG_DATA']['FULL_EMG'], 68)
    DummyData['MSG_DATA']['FULL_EMG'] = add_u8(DummyData['MSG_DATA']['FULL_EMG'], 1, val=2)

    DummyData['MSG_DATA']['FILTERING_HISTORY'] = add_u32(DummyData['MSG_DATA']['FILTERING_HISTORY'], 1)
    DummyData['MSG_DATA']['FILTERING_HISTORY'] = add_u8(DummyData['MSG_DATA']['FILTERING_HISTORY'], 5)

    # CMD_DATA_MONITOR
    DummyData['CMD_DATA_MONITOR']['VIEW_CACHE_SUMMARY'] = add_u32(DummyData['CMD_DATA_MONITOR']['VIEW_CACHE_SUMMARY'], 2)

    DummyData['CMD_DATA_MONITOR']['VIEW_SEGMENT_DETAILS'] = add_u32(DummyData['CMD_DATA_MONITOR']['VIEW_SEGMENT_DETAILS'], 1)
    DummyData['CMD_DATA_MONITOR']['VIEW_SEGMENT_DETAILS'] = add_u16(DummyData['CMD_DATA_MONITOR']['VIEW_SEGMENT_DETAILS'], 1)
    DummyData['CMD_DATA_MONITOR']['VIEW_SEGMENT_DETAILS'] = add_u8(DummyData['CMD_DATA_MONITOR']['VIEW_SEGMENT_DETAILS'], 1)

    DummyData['CMD_DATA_MONITOR']['GET_SEGMENT_FEATURE_VECTORS_START'] = add_u32(DummyData['CMD_DATA_MONITOR']['GET_SEGMENT_FEATURE_VECTORS_START'], 1)
    DummyData['CMD_DATA_MONITOR']['GET_SEGMENT_FEATURE_VECTORS_START'] = add_u8(DummyData['CMD_DATA_MONITOR']['GET_SEGMENT_FEATURE_VECTORS_START'], 1, val=1)

    DummyData['CMD_DATA_MONITOR']['GET_SEGMENT_FEATURE_VECTORS_VECTOR'] = add_u32(DummyData['CMD_DATA_MONITOR']['GET_SEGMENT_FEATURE_VECTORS_VECTOR'], 1)
    DummyData['CMD_DATA_MONITOR']['GET_SEGMENT_FEATURE_VECTORS_VECTOR'] = add_u16(DummyData['CMD_DATA_MONITOR']['GET_SEGMENT_FEATURE_VECTORS_VECTOR'], 1)
    DummyData['CMD_DATA_MONITOR']['GET_SEGMENT_FEATURE_VECTORS_VECTOR'] = add_q015(DummyData['CMD_DATA_MONITOR']['GET_SEGMENT_FEATURE_VECTORS_VECTOR'], 288)

    DummyData['CMD_DATA_MONITOR']['GET_SEGMENT_FEATURE_VECTORS_STOP'] = add_u32(DummyData['CMD_DATA_MONITOR']['GET_SEGMENT_FEATURE_VECTORS_STOP'], 1)
    DummyData['CMD_DATA_MONITOR']['GET_SEGMENT_FEATURE_VECTORS_STOP'] = add_u8(DummyData['CMD_DATA_MONITOR']['GET_SEGMENT_FEATURE_VECTORS_STOP'], 1, val=1)

    # CMD_CONFIGURE_DAUGHTERBOARD
    DummyData['CMD_CONFIGURE_DAUGHTERBOARD']['GET_PERIPHERALS'] = add_u8(DummyData['CMD_CONFIGURE_DAUGHTERBOARD']['GET_PERIPHERALS'], 8)

    # CMD_MOVEMENT_ENGINE
    DummyData['CMD_MOVEMENT_ENGINE']['GET_TERMINUS_LIST'] = add_u8(DummyData['CMD_MOVEMENT_ENGINE']['GET_TERMINUS_LIST'], 6)

    DummyData['CMD_MOVEMENT_ENGINE']['GET_ACTION_LIST'] = add_u8(DummyData['CMD_MOVEMENT_ENGINE']['GET_ACTION_LIST'], 6)

    DummyData['CMD_MOVEMENT_ENGINE']['SUSPEND_ACTIONS'] = add_u8(DummyData['CMD_MOVEMENT_ENGINE']['SUSPEND_ACTIONS'], 4)

    # CMD_ADJUST_ELECTRODES
    DummyData['CMD_ADJUST_ELECTRODES']['GET_ENABLED'] = add_u8(DummyData['CMD_ADJUST_ELECTRODES']['GET_ENABLED'], 2, val=170)

    DummyData['CMD_ADJUST_ELECTRODES']['GET_GAINS'] = add_u8(DummyData['CMD_ADJUST_ELECTRODES']['GET_GAINS'], 8, val=6)

    DummyData['CMD_ADJUST_ELECTRODES']['GET_FEATURE_EXTRACT_BUFF_LEN'] = add_u16(DummyData['CMD_ADJUST_ELECTRODES']['GET_FEATURE_EXTRACT_BUFF_LEN'], 8, val=400)

    # CMD_CONFIGURATION
    DummyData['CMD_CONFIGURATION']['GET_GENERAL_FIELDS'] = add_u8(DummyData['CMD_CONFIGURATION']['GET_GENERAL_FIELDS'], 8, val=1)

    DummyData['CMD_CONFIGURATION']['GET_GENERAL_PURPOSE_MAP_ENTRIES'] = add_u16(DummyData['CMD_CONFIGURATION']['GET_GENERAL_PURPOSE_MAP_ENTRIES'], 1, val=1)
    DummyData['CMD_CONFIGURATION']['GET_GENERAL_PURPOSE_MAP_ENTRIES'] = add_u8(DummyData['CMD_CONFIGURATION']['GET_GENERAL_PURPOSE_MAP_ENTRIES'], 1)
    DummyData['CMD_CONFIGURATION']['GET_GENERAL_PURPOSE_MAP_ENTRIES'] = add_u16(DummyData['CMD_CONFIGURATION']['GET_GENERAL_PURPOSE_MAP_ENTRIES'], 1, val=2)
    DummyData['CMD_CONFIGURATION']['GET_GENERAL_PURPOSE_MAP_ENTRIES'] = add_u8(DummyData['CMD_CONFIGURATION']['GET_GENERAL_PURPOSE_MAP_ENTRIES'], 1)
    DummyData['CMD_CONFIGURATION']['GET_GENERAL_PURPOSE_MAP_ENTRIES'] = add_u16(DummyData['CMD_CONFIGURATION']['GET_GENERAL_PURPOSE_MAP_ENTRIES'], 1, val=3)
    DummyData['CMD_CONFIGURATION']['GET_GENERAL_PURPOSE_MAP_ENTRIES'] = add_u8(DummyData['CMD_CONFIGURATION']['GET_GENERAL_PURPOSE_MAP_ENTRIES'], 1)

    DummyData['CMD_CONFIGURATION']['GET_FIRMWARE_VERSION'] = add_u8(DummyData['CMD_CONFIGURATION']['GET_FIRMWARE_VERSION'], 1, val=70)
    DummyData['CMD_CONFIGURATION']['GET_FIRMWARE_VERSION'] = add_u8(DummyData['CMD_CONFIGURATION']['GET_FIRMWARE_VERSION'], 1, val=97)
    DummyData['CMD_CONFIGURATION']['GET_FIRMWARE_VERSION'] = add_u8(DummyData['CMD_CONFIGURATION']['GET_FIRMWARE_VERSION'], 1, val=83)
    DummyData['CMD_CONFIGURATION']['GET_FIRMWARE_VERSION'] = add_u8(DummyData['CMD_CONFIGURATION']['GET_FIRMWARE_VERSION'], 1, val=122)

    DummyData = key_recursion(DummyData)

    with open('Core2DummyData.pkl', 'wb') as handle:
        pickle.dump(DummyData, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('Core2DummyData.pkl', 'rb') as handle:
        loaded = pickle.load(handle)
    
    pass

# CMD_EDIT_CONTROL_STRATEGY
    '''DummyData['CMD_EDIT_CONTROL_STRATEGY']['ADD_UNANIMOUS'] = add_u8(DummyData['CMD_EDIT_CONTROL_STRATEGY']['ADD_UNANIMOUS'], 3, 2)

    DummyData['CMD_EDIT_CONTROL_STRATEGY']['MODIFY_UNANIMOUS'] = add_u8(DummyData['CMD_EDIT_CONTROL_STRATEGY']['MODIFY_UNANIMOUS'], 3, 2)


    DummyData['CMD_EDIT_CONTROL_STRATEGY']['ADD_ONSET'] = add_q015(DummyData['CMD_EDIT_CONTROL_STRATEGY']['ADD_ONSET'], 1)
    DummyData['CMD_EDIT_CONTROL_STRATEGY']['ADD_ONSET'] = add_u8(DummyData['CMD_EDIT_CONTROL_STRATEGY']['ADD_ONSET'], 1)

    DummyData['CMD_EDIT_CONTROL_STRATEGY']['MODIFY_ONSET'] = add_q015(DummyData['CMD_EDIT_CONTROL_STRATEGY']['MODIFY_ONSET'], 1)
    DummyData['CMD_EDIT_CONTROL_STRATEGY']['MODIFY_ONSET'] = add_u8(DummyData['CMD_EDIT_CONTROL_STRATEGY']['MODIFY_ONSET'], 1)


    DummyData['CMD_EDIT_CONTROL_STRATEGY']['ADD_PROP'] = add_q015(DummyData['CMD_EDIT_CONTROL_STRATEGY']['ADD_PROP'], 1)
    DummyData['CMD_EDIT_CONTROL_STRATEGY']['ADD_PROP'] = add_float_norm(DummyData['CMD_EDIT_CONTROL_STRATEGY']['ADD_PROP'], 4)
    DummyData['CMD_EDIT_CONTROL_STRATEGY']['ADD_PROP'] = add_u8(DummyData['CMD_EDIT_CONTROL_STRATEGY']['ADD_PROP'], 1, val=2)
    DummyData['CMD_EDIT_CONTROL_STRATEGY']['ADD_PROP'] = add_u8(DummyData['CMD_EDIT_CONTROL_STRATEGY']['ADD_PROP'], 1, val=0)
    DummyData['CMD_EDIT_CONTROL_STRATEGY']['ADD_PROP'] = add_float_norm(DummyData['CMD_EDIT_CONTROL_STRATEGY']['ADD_PROP'], 4)
    DummyData['CMD_EDIT_CONTROL_STRATEGY']['ADD_PROP'] = add_u8(DummyData['CMD_EDIT_CONTROL_STRATEGY']['ADD_PROP'], 1, val=1)
    DummyData['CMD_EDIT_CONTROL_STRATEGY']['ADD_PROP'] = add_float_norm(DummyData['CMD_EDIT_CONTROL_STRATEGY']['ADD_PROP'], 4)

    DummyData['CMD_EDIT_CONTROL_STRATEGY']['MODIFY_PROP'] = add_q015(DummyData['CMD_EDIT_CONTROL_STRATEGY']['MODIFY_PROP'], 1)
    DummyData['CMD_EDIT_CONTROL_STRATEGY']['MODIFY_PROP'] = add_float_norm(DummyData['CMD_EDIT_CONTROL_STRATEGY']['MODIFY_PROP'], 4)
    DummyData['CMD_EDIT_CONTROL_STRATEGY']['MODIFY_PROP'] = add_u8(DummyData['CMD_EDIT_CONTROL_STRATEGY']['MODIFY_PROP'], 1, val=2)
    DummyData['CMD_EDIT_CONTROL_STRATEGY']['MODIFY_PROP'] = add_u8(DummyData['CMD_EDIT_CONTROL_STRATEGY']['MODIFY_PROP'], 1, val=0)
    DummyData['CMD_EDIT_CONTROL_STRATEGY']['MODIFY_PROP'] = add_float_norm(DummyData['CMD_EDIT_CONTROL_STRATEGY']['MODIFY_PROP'], 4)
    DummyData['CMD_EDIT_CONTROL_STRATEGY']['MODIFY_PROP'] = add_u8(DummyData['CMD_EDIT_CONTROL_STRATEGY']['MODIFY_PROP'], 1, val=1)
    DummyData['CMD_EDIT_CONTROL_STRATEGY']['MODIFY_PROP'] = add_float_norm(DummyData['CMD_EDIT_CONTROL_STRATEGY']['MODIFY_PROP'], 4)


    DummyData['CMD_EDIT_CONTROL_STRATEGY']['ADD_VR'] = add_u8(DummyData['CMD_EDIT_CONTROL_STRATEGY']['ADD_VR'], 3)
    DummyData['CMD_EDIT_CONTROL_STRATEGY']['ADD_VR'] = add_float_norm(DummyData['CMD_EDIT_CONTROL_STRATEGY']['ADD_VR'], 2)

    DummyData['CMD_EDIT_CONTROL_STRATEGY']['MODIFY_VR'] = add_u8(DummyData['CMD_EDIT_CONTROL_STRATEGY']['MODIFY_VR'], 3)
    DummyData['CMD_EDIT_CONTROL_STRATEGY']['MODIFY_VR'] = add_float_norm(DummyData['CMD_EDIT_CONTROL_STRATEGY']['MODIFY_VR'], 2)'''