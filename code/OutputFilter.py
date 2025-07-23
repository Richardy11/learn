from collections import deque
import itertools
from random import uniform
import numpy as np
import matplotlib.pyplot as plt
import copy

class OutputFilter:
    """ A Python implementation of a uniform vote output filter """
    def __init__( self, classes, sensitivities, continuous, uniform_size ):
        """
        Constructor

        Parameters
        ----------
        size : int
            The length of the output filter

        Returns
        -------
        obj
            A UniformVoteFilter object
        """
        self.__rest_buffer = deque()
        self.__class_buffer = deque()
        '''
        0 - power
        1 - tripod
        2 - key
        3 - index
        4 - wrist
        '''
        self.__continuous = continuous
        self.__classes = classes
        temp_classes = copy.deepcopy(classes)

        if self.__continuous['TRANSIENT_WRIST']:
            self.__cont_rot = ['REST', 'PALM DOWN', 'PALM UP']
        else:
            self.__cont_rot = ['REST']

        '''try:
            temp_classes.remove('REST')
            temp_classes.remove('PALM DOWN')
            temp_classes.remove('PALM UP')
        except:
            pass'''

        if self.__continuous['TRANSIENT_ELBOW']:
            self.__cont_elb = ['REST', 'ELBOW BEND', 'ELBOW EXTEND']
        else:
            self.__cont_elb = ['REST']

        '''try:
            temp_classes.remove('ELBOW BEND')
            temp_classes.remove('ELBOW EXTEND')
        except:
            pass

        try:
            temp_classes.remove('OPEN')
        except:
            pass'''

        self.__cont_open = ['REST']
        for cl in classes:
            try:
                if self.__continuous['TRANSIENT_'+cl]:
                    self.__cont_open.append(cl)
            except: pass

        self.__current_pred = 'REST'

        '''self.__cont_open = ['REST']
        for i, cl in enumerate(temp_classes):
            if cl.upper().find("ELBOW ") == -1 and cl.upper().find("PALM ") == -1 and cl.upper() not in ['REST', 'OPEN']:
                if (self.__continuous[i]):
                    self.__cont_open.append(cl)'''

        '''self.__state = 'REST'
        self.__active_state = False'''

        self.sensitivities = sensitivities

        self.__class_buffer_size = max(self.sensitivities.values())
        self.__rest = self.sensitivities['REST']

        self.__size = uniform_size
        self.__buffer = deque()

    def filter( self, pred, filter ):
        """
        """
        if filter == 'First Over Filter':
            self.__rest_buffer.append(pred)
            self.__class_buffer.append(pred)

            if len(self.__rest_buffer) > self.__rest:
                self.__rest_buffer.popleft()

            if len(self.__class_buffer) > self.__class_buffer_size:
                self.__class_buffer.popleft()

            #if self.__continuous is True:
            #tmp_current_pred = self.__current_pred
            if len(self.__rest_buffer) == self.__rest:
                rest_out = np.unique(self.__rest_buffer)
                if rest_out.shape[0] == 1 and rest_out[-1] == 'REST':
                    self.__current_pred = 'REST'
                    #self.__state = 'REST'
                    #self.__class_buffer = deque()

            if len(self.__class_buffer) == self.__class_buffer_size:

                movement_queues = {}
                for cl in self.__classes:
                    if cl != 'REST':
                        movement_queues[cl] = deque(itertools.islice(self.__class_buffer, self.__class_buffer_size-self.sensitivities[cl], self.__class_buffer_size))

                        if cl in self.__cont_open:
                            if len(np.unique(movement_queues[cl])) == 1 and np.unique(movement_queues[cl]) == cl and (self.__current_pred == 'REST' or self.__current_pred == 'OPEN'): #(self.__state == cl and self.__current_pred == 'OPEN') ) : 
                                self.__current_pred = cl
                    
                        elif cl in self.__cont_rot:
                            if len(np.unique(movement_queues[cl])) == 1 and np.unique(movement_queues[cl]) == cl and (self.__current_pred in self.__cont_rot): self.__current_pred = cl

                        elif cl in self.__cont_elb:
                            if len(np.unique(movement_queues[cl])) == 1 and np.unique(movement_queues[cl]) == cl and (self.__current_pred in self.__cont_elb): self.__current_pred = cl
                        # non transient states
                        else:
                            if len(np.unique(movement_queues[cl])) == 1 and np.unique(movement_queues[cl]) == cl and (self.__current_pred == 'REST' or cl in self.__cont_open or( self.__current_pred in self.__cont_open and cl == 'OPEN') ): self.__current_pred = cl #and self.__state == 'REST' : self.__current_pred = cl #and (self.__current_pred in self.__cont_open): self.__current_pred = cl


            '''if tmp_current_pred != self.__current_pred and tmp_current_pred == 'REST':
                self.__state = self.__current_pred'''

        elif filter == 'Uniform Filter':

            self.__buffer.append(pred)    # circular buffer
            if len(self.__buffer) > self.__size:
                self.__buffer.popleft()
            
            if np.unique(self.__buffer).shape[0] == 1:     # all elements in buffer are the same
                return np.unique(self.__buffer)[0]
            else:
                return 'REST'

        return self.__current_pred

if __name__ == '__main__':
    N_CLASSES = [0,1,0,2,1,1,2,2,3,3,0,3,4,3,4,0,5,1,5,0,6,1,6,0,7,1,7,0,8,9,8,9,0]
    UNIFORM_SIZE = 15
    classes = ['REST', 'OPEN', 'POWER', 'PALM DOWN', 'PALM UP', 'TRIPOD', 'KEY', 'INDEX', 'ELBOW BEND', 'ELBOW EXTEND']
    predictions = []
    sensitivities = {}
    for cl in classes:
        if cl == 'REST':
            sensitivities[cl] = 5
        else:
            sensitivities[cl] = 10
    sensitivities['PALM DOWN'] = 30

    for i in N_CLASSES:
        predictions.extend( [ i ] * UNIFORM_SIZE )

    output = []
    firstover = OutputFilter(continuous=[True, False, False, False, False, False], sensitivities = sensitivities,classes = ['REST', 'OPEN', 'POWER', 'PALM DOWN', 'PALM UP', 'TRIPOD', 'KEY', 'INDEX', 'ELBOW BEND', 'ELBOW EXTEND'], uniform_size= 10)
    for i, pred in enumerate(predictions):
        output.append( classes.index(firstover.filter( classes[pred], filter = 'First Over Filter'  )) )

    plt.plot(range(len(predictions)),predictions)
    plt.plot(range(len(predictions)),output)
    plt.show()
    print( 'Predictions:', predictions )
    print( 'Output:     ', output )