import socket
import subprocess
import struct
import random
import os

class MyoTrain:
    """ A Python implementation of a MyoTrain interface """
    MOVEMENT_DICTIONARY = { 'rest'         :  0, 'open'         :  1, 'trigger'       :  2,
                            'column'       :  3, 'index'        :  4, 'key'           :  33,
                            'mouse'        :  6, 'park_thumb'   :  7, 'power'         :  29,
                            'pinch_closed' :  26, 'pinch_open'   : 10, 'tripod'        : 32,
                            'tripod_open'  : 12, 'palm down'    : 35, 'palm up'       : 36,
                            'elbow bend'   : 45, 'elbow extend' : 46  }

    @staticmethod
        
    def create_packet( movement, velocity, pause_byte ):
        # movement          --> movement class to perform
        # velocity          --> speed at which to do the movement
        # compound_movement --> if false, returns to rest after move (set 1)
        # active            --> can the hand move (set to 1)
        # max_cue_length    --> how long each cue can last for (set to 15)
        return bytes( [ movement, 0, 0, 0, velocity, 1, 1, 15, pause_byte ] )
    
    def __init__( self, path = os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ), 'myotrain', 'MyoTrain_R.exe' ), ip = '127.0.0.1', port = 9027, running = False ):
        self.__addr = ( ip, port )
        self.__udp = socket.socket( socket.AF_INET, socket.SOCK_DGRAM )
        try:
            self.__udp.bind( (ip, 9097) )
        except:
            pass
        if running == False:
            self.__venv = subprocess.Popen( [ path ], shell = False,
                                            stdin = subprocess.DEVNULL,
                                            stdout = subprocess.DEVNULL,
                                            stderr = subprocess.DEVNULL )
        self.__udp.setblocking(False)

    def __del__( self ):
        try: self.__venv.kill()
        except AttributeError: pass

    def publish( self, msg, speed = 1, pause_byte = 1 ):
        #if self.__venv.poll() is None: # virtual environment is on
        try:
            move = MyoTrain.MOVEMENT_DICTIONARY[ msg.lower() ]
            speed = int( 100 * min( [ max( [ 0.0, speed ] ), 1.0 ] ) ) 
            pkt = MyoTrain.create_packet( move, speed, pause_byte )
            self.__udp.sendto( pkt, self.__addr )
        except KeyError:
            pass

    def read( self ):
        #if self.__venv.poll() is None: # virtual environment is on
        try:
            data = self.__udp.recv( 1024 )
            return data
        except BlockingIOError:
            pass
        return None

if __name__ == '__main__':
    import inspect
    import argparse
    import time

    # helper function for booleans
    def str2bool( v ):
        if v.lower() in [ 'yes', 'true', 't', 'y', '1' ]: return True
        elif v.lower() in [ 'no', 'false', 'n', 'f', '0' ]: return False
        else: raise argparse.ArgumentTypeError( 'Boolean value expected!' )

    # parse commandline entries
    class_init = inspect.getargspec( MyoTrain.__init__ )
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

    vhand = MyoTrain( args.path, args.ip, args.port )
    moves = [ 'rest', 'open', 'trigger', 'column', 'index_point',
              'key', 'mouse', 'park_thumb', 'power', 'pinch_closed',
              'pinch_open', 'tripod', 'tripod_open', 'pronate', 'supinate',
              'elbow_bend', 'elbow_extend' ]
    
    print( '------------ Movement Commands ------------' )
    print( '| 00  -----  REST                         |' )
    print( '| 01  -----  HAND OPEN                    |' )
    print( '| 02  -----  TRIGGER GRASP                |' )
    print( '| 03  -----  COLUMN GRASP                 |' )
    print( '| 04  -----  INDEX_POINT                  |' )
    print( '| 05  -----  KEY GRASP                    |' )
    print( '| 06  -----  MOUSE GRASP                  |' )
    print( '| 07  -----  PARK THUMB                   |' )
    print( '| 08  -----  POWER GRASP                  |' )
    print( '| 09  -----  FINE PINCH CLOSED            |' )
    print( '| 10  -----  FINE PINCH OPEN              |' )
    print( '| 11  -----  TRIPOD PINCH CLOSED          |' )
    print( '| 12  -----  TRIPOD PINCH OPEN            |' )
    print( '| 13  -----  WRIST PRONATION              |' )
    print( '| 14  -----  WRIST SUPINATION             |' )
    print( '| 15  -----  ELBOW BEND                   |' )
    print( '| 16  -----  ELBOW EXTEND                 |' )
    print( '-------------------------------------------' )
    print( '| Press [Q] to quit!                      |' )
    print( '-------------------------------------------' )

    time.sleep(10)    
    vhand = MyoTrain( args.path, args.ip, args.port )

    done = False  
    while not done:
        cmd = input( 'Command: ' )
        if cmd.lower() == 'q':
            done = True
        else:
            try:
                idx = int( cmd )
                if idx in range( 0, len( moves ) ):
                    vhand.publish( moves[ idx ], 1, 1 )
            except ValueError:
                pass
    print( 'Bye-bye!' )