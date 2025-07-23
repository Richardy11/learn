import sys

if sys.platform == 'win32': import bluetooth
elif sys.platform == 'linux': import socket
else: raise RuntimeError( 'Bluetooth not supported for this OS:' , sys.platform )

#from . import AbstractBaseOutput

class Bebionic3:
    """ Python implementation of a Bebionic3 prosthetic hand driver using the IBT control board """
    def __init__( self, mac = 'ec:fe:7e:1a:57:35', elbow = False ):
        """
        Constructor

        Parameters
        ----------
        mac : str
            The MAC address of the controller board for the Bebionic3
        elbow : bool
            True if a powered elbow is connected, False else

        Returns
        -------
        obj
            A Bebionic3 interface object
        """
        self._elbow = elbow
        10
        if sys.platform == 'win32':
            self._bt = bluetooth.BluetoothSocket( bluetooth.RFCOMM )
        else:
            self._bt = socket.socket( socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM )
        
        self._bt.connect( ( mac, 1 ) )
        
        self._init_bt()
        self._bt.send( b'\xff\x02\x9e\x9f' )   # stop movement command
        self._bt.send( b'\xff\x02\x9b\x9c' )   # clear movement command
        self._move_dict = { 'tripod'       : b'\xff\x04\x9c\x09\x01\xa9',
                            'power'        : b'\xff\x04\x9c\x11\x01\xb1',
                            'pinch_open'   : b'\xff\x04\x9c\x0e\x01\xae',
                            'active_index' : b'\xff\x04\x9c\x10\x01\xb0',
                            'pinch_closed' : b'\xff\x04\x9c\x12\x01\xb2',
                            'key_lateral'  : b'\xff\x04\x9c\x13\x01\xb3',
                            'index_point'  : b'\xff\x04\x9c\x14\x01\xb4',
                            'mouse'        : b'\xff\x04\x9c\x15\x01\xb5',
                            'column'       : b'\xff\x04\x9c\x16\x01\xb6',
                            'relaxed'      : b'\xff\x04\x9c\x17\x01\xb7',
                            'rest'         : b'\xff\x02\x9b\x9c',
                            'open'         : b'\xff\x04\x9c\x01\x01\xa1',
                            'pronate'      : b'\xff\x04\x9c\x24\x01\xc4',
                            'supinate'     : b'\xff\x04\x9c\x23\x01\xc3',
                            'elbow_flex'   : b'\xff\x04\x9c\x2d\x01\xcd',
                            'elbow_extend' : b'\xff\x04\x9c\x2e\x01\xce',
                            'close'        : b'\xff\x04\x9c\x02\x01\xa2' }
        self._last_move = None

    def __del__( self ):
        """
        Destructor

        Stops the Bebionic3 from any movements its currently doing and closes communication.
        """
        try:
            self._bt.send( b'\xff\x02\x9e\x9f' )   # stop movement command
            self._bt.send( b'\xff\x02\x9b\x9c' )   # clear movement command
            self._bt.close()                       # close communication
        except (AttributeError, OSError):
            # did not open the bluetooth communication
            pass

    def _init_bt( self ):
        """
        Initializes the bluetooth connection and registers all available grips
        """

        self._bt.send( b'\xff\x06\x80\x00\x00\x00\x00\x85' )                           # connect
        self._bt.send( b'\xff\x02\x00\x01' )
        self._bt.send( b'\xff\x02\x83\x84' )
        self._bt.send( b'\xff\x02\x00\x01' )
        
        # add grips
        self._bt.send( b'\xff\x16\x93\x01\xff\x01\x00\x00\x00\xff\x01\x64\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x0d' )    # open
        self._bt.send( b'\xff\x16\x93\x02\xff\x02\x00\x00\x00\xff\x01\x64\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x0f' )    # 
        self._bt.send( b'\xff\x16\x93\x11\xff\x0c\x00\x00\x00\x01\x01\x64\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x2a' )    # power
        self._bt.send( b'\xff\x16\x93\x09\xff\x0c\x00\x00\x00\x00\x01\x64\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x21' )    # tripod
        self._bt.send( b'\xff\x16\x93\x0e\xff\x0c\x00\x00\x00\x02\x01\x64\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x28' )    # pinch_open
        self._bt.send( b'\xff\x16\x93\x10\xff\x0c\x00\x00\x00\x03\x01\x64\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x2b' )    # pinch_closed
        self._bt.send( b'\xff\x16\x93\x12\xff\x0c\x00\x00\x00\x04\x01\x64\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x2e' )    # active_index
        self._bt.send( b'\xff\x16\x93\x13\xff\x0c\x00\x00\x00\x05\x01\x64\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x30' )    # key
        self._bt.send( b'\xff\x16\x93\x14\xff\x0c\x00\x00\x00\x06\x01\x64\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x32' )    # index_point
        self._bt.send( b'\xff\x16\x93\x15\xff\x0c\x00\x00\x00\x07\x01\x64\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x34' )    # mouse
        self._bt.send( b'\xff\x16\x93\x16\xff\x0c\x00\x00\x00\x08\x01\x64\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x36' )    # column
        self._bt.send( b'\xff\x16\x93\x17\xff\x0c\x00\x00\x00\x09\x01\x64\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x38' )    # relaxed
        if not self._elbow:
            '''self._bt.send( b'\xff\x16\x93\x23\xff\x09\x00\x00\x00\x01\x01\x46\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x1b' )    # supinate
            self._bt.send( b'\xff\x16\x93\x24\xff\x09\x00\x00\x00\x02\x01\x46\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x1d' )    # pronate'''
        else:
            raise RuntimeError( "Elbow actuation is currently not supported!" )

        
    def _send_movement_command( self, move ):
        """
        Send a movement command to the Bebionic3

        Parameters
        ----------
        move : str
            Name of the desired movement class

        Raises
        ------
        RuntimeError
            Invalid movement class name is given
        """
        if move in self._move_dict:
            self._bt.send( b'\xff\x02\x9e\x9f' )        # stop last movement command
            self._bt.send( self._move_dict['rest'] )    # clear last movement command
            self._bt.send( self._move_dict[ move ] )    # send current movement command
            self._bt.send( self._move_dict[ move ] )    # must send active movement commands twice
        else:
            raise RuntimeError( 'Invalid movement class for the Bebionic3: ', move )

    def publish( self, msg ):
        """
        Publish output to the Bebionic3

        Parameters
        ----------
        msg : str
            The name of the output movement class to send

        Notes
        -----
        If the Bebionic3 is already doing the desired movement class, no command is sent to save on bandwidth
        """
        # if msg is not self._last_move:
        self._send_movement_command( msg )
        self._last_move = msg

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
    class_init = inspect.getargspec( Bebionic3.__init__ )
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

    bb3 = Bebionic3( args.mac, args.elbow )
    moves = [ 'tripod', 'power', 'pinch_open', 'active_index', 'pinch_closed',
              'key_lateral', 'index_point', 'mouse', 'column', 'relaxed',
              'rest', 'open', 'pronate', 'supinate', 'elbow_flex', 'elbow_extend', 'close' ]
    
    print( '------------ Movement Commands ------------' )
    print( '| 00  -----  STANDARD TRIPOD CLOSED       |' )
    print( '| 01  -----  POWER                        |' )
    print( '| 02  -----  THUMB PRECISION OPEN         |' )
    print( '| 03  -----  ACTIVE INDEX                 |' )
    print( '| 04  -----  THUMB PRECISION CLOSED       |' )
    print( '| 05  -----  KEY LATERAL                  |' )
    print( '| 06  -----  FINGER INDEX POINT           |' )
    print( '| 07  -----  MOUSE                        |' )
    print( '| 08  -----  COLUMN                       |' )
    print( '| 09  -----  RELAXED                      |' )
    print( '| 10  -----  NOGRIP                       |' )
    print( '| 11  -----  OPEN                         |' )
    print( '| 12  -----  PRONATE                      |' )
    print( '| 13  -----  SUPINATE                     |' )
    print( '| 14  -----  ELBOW FLEXION                |' )
    print( '| 15  -----  ELBOW EXTENSION              |' )    
    print( '-------------------------------------------' )
    print( '| Press [Q] to quit!                      |' )
    print( '-------------------------------------------' )

    done = False  
    while not done:
        cmd = input( 'Command: ' )
        if cmd.lower() == 'q':
            done = True
        else:
            try:
                idx = int( cmd )
                if idx in range( 0, len( moves ) ):
                    bb3.publish( moves[ idx ] )
            except ValueError:
                pass
    print( 'Bye-bye!' )