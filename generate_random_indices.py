import os

import numpy as np

np.random.seed( 0 ) # optional

METHODS = [ 1 ]
POSITIONS = [ 1, 5, 6]

WRIST = [ 'PALM UP', 'PALM DOWN' ]
HAND = [ 'OPEN','POWER' ]
#HAND = [ 'OPEN','POWER', 'TRIPOD', 'KEY' ]
#HAND = [ 'OPEN','POWER', 'TRIPOD', 'KEY', 'INDEX', 'PINCH' ]

# Palm up/palm down in a even situation choos es one half of the HAND classes (in an ODD, it doubles the # of trials and Palm up chooses the last half/Palm down chooses first half\


# what we think is better: even distribution of palm up/palm down between all HAND classes

# FittsLawTask.py and made generate_random_indices.py

N_TASKS_REPEAT = 3 # number of times a task type will appear in a position for a given method

TASKS = [ i for i in range( ( 2 * len( HAND ) ) if ( len( HAND ) % 2 ) else len( HAND ) ) ]
print( TASKS )

method_index = []
position_index = []
task_index = []

for method in METHODS:
    for position in POSITIONS:
        for task in TASKS:
            for _ in range( N_TASKS_REPEAT ):

                method_index.append( method )
                position_index.append( position ) #print to screen
                task_index.append( task )

idx = np.random.permutation( np.arange( len( method_index ) ) )

method_index = [ method_index[i] for i in idx ]
position_index = [ position_index[i] for i in idx ]
task_index = [ task_index[i] for i in idx ]

print( 'TOTAL NUMBER OF TASKS: %d' % len( method_index ) )
print( method_index )
print( position_index )
print( task_index )

with open( 'fittslawrandom.txt', 'w' ) as f:
    for i in idx:
        f.write( '%d\n' % i )

with open( 'fittslawpositions.txt', 'w' ) as f:
    for position in position_index:
        f.write( '%d\n' % position )

with open( 'fittslawmethods.txt', 'w' ) as f:
    for method in method_index:
        f.write( '%d\n' % method )

# can randomize list AFTER choose targets is generated.
# FittslawTask line 219 to call method 1/2