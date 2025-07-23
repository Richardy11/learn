
import multiprocessing as mp
import queue

import matplotlib

from RunClassifier import RunClassifier
matplotlib.use("QT5Agg")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from matplotlib.widgets import Button

import sys

import time
import random

import pickle
import os
from os import path

import copy

import csv

from scipy.io import loadmat

from collections import deque

from OnsetThreshold import Onset
from RealTimeDataPlot import RealTimeDataPlot

from TimeDomainFilter import TimeDomainFilter
from FourierTransformFilter import FourierTransformFilter
from SenseController import SenseController
from MyoArmband import MyoArmband
from RunClassifier import RunClassifier
from ProportionalControl import Proportional_Control


sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ))

from local_addresses import address


#set file path for Parameters and get Sense address
SUBJECT = 10

if len(sys.argv) > 1:
    #print( 'SUBJECT:', str(sys.argv[1]) )
    SUBJECT = sys.argv[1]

CUE_FLDR = os.path.join( 'cues' )

PARAM_FILE = os.path.join('data', 'Subject_%s' % SUBJECT, 'RESCU_Protocol_Subject_%s.pkl' % SUBJECT )

with open( PARAM_FILE, 'rb' ) as pkl:
    Parameters = pickle.load( pkl )

EMG_SCALING_FACTOR = 10000.0

cue_count = ( Parameters['General']['SAMPLING_RATE'] * Parameters['Calibration']['CUE_DURATION'] - Parameters['General']['WINDOW_SIZE'] ) / Parameters['General']['WINDOW_STEP']

local_surce_info = address()


class SignalLevel:
    def __init__(self, classes = ('rest','open','power','pronate','supinate','tripod'), method = "Unspecified", feat_smoothing = 0, vel_smoothing = 0, num_channels = 1, sampling_rate = 1000, buffer_time = 5, title=None):
        self._classes = classes
        self._num_channels = num_channels
        self._sampling_rate = sampling_rate
        self.show_raw = True
        self._buffer_time = buffer_time

        self.method = method
        self.feat_smoothing = feat_smoothing
        self.vel_smoothing = vel_smoothing

        self._title = title

        self.doubleparentdict = {}

        self._exit_event = mp.Event()
        self._calib_exit = mp.Event()

        self._queue = mp.Queue()
        self._calibration_queue = mp.Queue()

        self.trialcount = 0

        self._plotter = mp.Process(target = self._plot)

        self._plotter.start()


    def _change_color_max(self):
        self.target_bar.set_color('g')
    
    def _revert_color(self):
        self.target_bar.set_color('r')

    def _distance_update(self):
        self.timeractive = True
        self.index = random.randint(0,len(self.dist)-1)
        self.distance = self.dist[self.index]
        self.target_width = self.width[self.index]
        self.offset = self.target_width/2
    
    def _rest_update(self):
        self.timeractive = False
        del self.dist[self.index]
        del self.width[self.index]
        self.distance = self.threshold
        self.offset = 0.0025

    def _target_bar_update(self):
        if self.distance == self.threshold:
            self.target_bar, = self.axis1.bar(0,self.target_width,1,self.distance-self.offset,color='r',alpha=0)
        elif self.distance > self.threshold:
            self.target_bar, = self.axis1.bar(0,self.target_width,1,self.distance-self.offset,color='r',alpha=0.6)

    def _clear_fig(self):
        self.axis1.cla()
        self.axis1.axis('off')

    def _restart(self):
        plt.pause(0.001)
        plt.waitforbuttonpress()
        try:
            self.prompt.remove()
            self.trialtext.remove()
            self.targettext.remove()
        except:
            pass
        self._signallevelplot(12)

    def _signallevelplot(self,input): 
        self.axis1.axis('on')
        self.axis1.tick_params(which='both',bottom=False,top=False,labelbottom=False,right=True)
        self.axis1.set_title('Mean Value Contraction')
        self.threshold = self.lower
        self.maxval = self.upper
        self.axis1.set_xlim(-0.5,0.5)
        self.axis1.set_ylim(self.threshold,self.maxval)
        self.buttonbox.cla() 
        self.buttonbox.axis('off')
        try:
            self.settings.remove()
        except: 
            pass


        inputloc = os.path.join('Input', self.filename)


        self.distlist = []
        self.widthlist = []
        self.tempdist = []
        self.tempwidth = []
        with open(inputloc) as csv_file:
            csv_reader = csv.reader(csv_file,delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    for i in row:
                        try:
                            self.distlist.append(float(i))
                            self.tempdist.append(float(i))
                        except:
                            pass
                elif line_count == 1:
                    for i in row:
                        try:
                            self.widthlist.append(float(i))
                            self.tempwidth.append(float(i))
                        except:
                            pass
                line_count += 1
        
                

        self.distlist = [x/100*(self.maxval-self.threshold)+self.threshold for x in self.distlist]
        self.widthlist = [x/100*(self.maxval-self.threshold) for x in self.widthlist]
        

        self.dist = []
        self.width = []
        for i in self.distlist:
            for j in range(len(self.widthlist)):
                self.dist.append(i)
        
        self.width = self.widthlist*len(self.distlist)
            
        self.index = random.randint(0,len(self.dist)-1)
        self.distance = self.dist[self.index]
        self.target_width = self.width[self.index] 

        def _exit_callback(event):
            self.close()

        exax = plt.axes([0.75, 0.05, 0.2, 0.025])
        exitbutton = Button(exax, 'EXIT')
        exitbutton.on_clicked(_exit_callback)

        #draw target bar
        self.offset = self.target_width/2
        self.target_bar, = self.axis1.bar(x = 0,height = self.target_width,width = 1,bottom = self.distance-self.offset,color='r',alpha=0.6)
        
        # draw bar plot
        height = 0
        self.barplot, = self.axis1.bar(0,height,1,color='b',alpha=0.2)

        self.targetcount = 0

        datadict = {'Movement': '','Distance': 0, 'Width': 0, 'Target Range': (0,0),'Completed':False,'Completion Time':0,'Movement Time': 0,'data':[]}
        parentdict = {}

        self.idlist = []
        self.timelist = []

        self.numtargets = len(self.dist)

        timerbool = True
        runthis = True
        self.timeractive = True

        self.targettext = self.fig1.text(0.9,0.95,'Target: %d/%d' % (self.targetcount+1,self.numtargets),size=20,ha='center',va='center',bbox=dict(boxstyle = 'square',color='gray',alpha=0.3))
        self.trialtext = self.fig1.text(0.1,0.95,'Trial: %d/%d' % (self.trialcount+1,self.numtrials),size=20,ha='center',va='center',bbox=dict(boxstyle = 'square',color='gray',alpha=0.3))

        while not self._exit_event.is_set():
            
            data = []
                    # pull samples from queue
            while self._queue.qsize() > 0:
                ctrl = self._queue.get() # ctrl is a tuple of str
                data.append( ctrl )
            
            
            if data:
                self._update_bar(data[0])
                datadict['data'].append((data[0],time.perf_counter()))
                if timerbool == True:
                    timeout = time.perf_counter()
                if runthis == True:
                    timer = time.perf_counter()
                
                timerbool = False
                runthis = False

                if self.height >= (self.distance-self.offset) and self.height <= (self.distance+self.offset) and not self.target_bar._facecolor[1]:  
                    start = time.perf_counter()
                    active = True
                    self._change_color_max()
                    
                elif (self.height < (self.distance-self.offset) or self.height > (self.distance+self.offset)):
                    self._revert_color()
                    active = False
                
                if active == True:
                    if (time.perf_counter()-start) > self.holdtime:
                        finished = time.perf_counter()-timer
                            
                        self.target_bar.remove()

                        active = False
                        runthis = True
                        
                        if not self.targetcount == self.numtargets:
                            if self.distance == self.threshold:
                                self.prompt3.remove()
                                if self.rest_time > 0:
                                    rest_second = self.rest_time
                                    for i in range(int(self.rest_time)):
                                        self.prompt2 = self.axis1.text(0,3*(self.maxval-self.threshold)/4+self.threshold,'Next Target in %.0f' % rest_second,size=20,ha='center',va='center',bbox=dict(boxstyle = 'square',color='gray',alpha=0.3),color='g')
                                        rest_second -= 1
                                        plt.pause(1)
                                        if self.prompt2:
                                            self.prompt2.remove()
                                elif self.rest_time == 0:
                                    self.prompt4 = self.axis1.text(0,3*(self.maxval-self.threshold)/4+self.threshold,'Press ENTER for Next Target',size=20,ha='center',va='center',bbox=dict(boxstyle='square',color='gray',alpha=0.3),color='g')
                                    plt.pause(0.001)
                                    plt.waitforbuttonpress()
                                    self.prompt4.remove()
                                timerbool = True
                                plt.pause(0.001)
                                self._distance_update()
                                self._target_bar_update()
                            elif self.distance > self.threshold:
                                datadict['Movement'] = self.movement
                                datadict['Distance'] = self.distance
                                datadict['Width'] = self.target_width
                                datadict['Target Range'] = (self.distance-self.target_width/2,self.distance+self.target_width/2)
                                datadict['Completed'] = True
                                movement = finished - self.holdtime
                                datadict['Completion Time'] = finished
                                datadict['Movement Time'] = movement


                                datadictcopy = copy.deepcopy(datadict)
                                parentdict[self.targetcount] = datadictcopy

                                datadict['data'] = []
                                self.targetcount+=1
                                self.prompt3 = self.axis1.text(0,3*(self.maxval-self.threshold)/4+self.threshold,'Completed! \nReturn to REST',size=20,ha='center',va='center',bbox=dict(boxstyle = 'square',color='gray',alpha=0.3),color='g')
                                self._rest_update()
                                self._target_bar_update()
                                self._update_target()
                                
                        if self.targetcount == self.numtargets:
                                self._clear_fig()
                                donetext = self.axis1.text(0.5,0.5,'Done! Saving data...',size=35,ha='center',va='center',bbox=dict(boxstyle = 'round',color='palegreen',alpha=0.3))
                                plt.pause(1)
                                donetext.remove()


                                filename = 'SLData_Subject_%d_T%s_%s_FS%s_VS%s' % (int(SUBJECT), (self.trialcount+1), self.method, self.feat_smoothing, self.vel_smoothing)
                                self.doubleparentdict[filename] = parentdict

                                '''folder = '.\Data\Subject_%d' % int(SUBJECT)
                                folder2 = 'SignalLevelData'
                                filename = 'SLData_Subject_%d_T%s_%s_FS%s_VS%s.mat' % (int(SUBJECT), (self.trialcount+1), self.method, self.feat_smoothing, self.vel_smoothing)
                                if not path.exists(folder):
                                    os.mkdir(folder)
                                if not path.exists(folder + '/' + folder2):
                                    os.mkdir(folder + '/' + folder2)
                                filepath = folder + '/' + folder2 + '/' + filename
                                if not path.exists(filepath):
                                    f = open(filepath, "w")
                                    f.write("")
                                    f.close()

                                with open(filepath, 'wb') as f:
                                    pickle.dump(parentdict,f)
                                with open(filepath, 'rb') as f:
                                        mydict = pickle.load(f)
                                print(mydict)'''

                                self.trialcount += 1
                                if self.trialcount < self.numtrials:
                                    text1 = '\n\n'.join(('Save Complete','Press ENTER to Continue to Next Trial'))
                                    self.prompt = self.axis1.text(0.5,0.7,text1,size=25,ha='center',va='center',bbox=dict(boxstyle='round',color='lightgreen',alpha=0.3))
                                    self._restart()
                                elif self.trialcount == self.numtrials:
                                    folder = '.\Data\Subject_%d' % int(SUBJECT)
                                    folder2 = 'SignalLevelData'
                                    filename = 'SLData_%s_FS%s_VS%s.pkl' % (self.method, self.feat_smoothing, self.vel_smoothing)
                                    if not path.exists(folder):
                                        os.mkdir(folder)
                                    if not path.exists(folder + '/' + folder2):
                                        os.mkdir(folder + '/' + folder2)
                                    filepath = folder + '/' + folder2 + '/' + filename
                                    if not path.exists(filepath):
                                        f = open(filepath, "w")
                                        f.write("")
                                        f.close()

                                    with open(filepath, 'wb') as f:
                                        pickle.dump(self.doubleparentdict,f)
                                    with open(filepath, 'rb') as f:
                                            mydict = pickle.load(f)
                                    print(mydict)

                                    self._exit_event.set()
                                plt.pause(1)

                if self.timeractive == True:
                    if timerbool == False:
                        if (time.perf_counter()-timeout) >= self.timeout:
                            self.target_bar.remove()

                            active = False
                            runthis = True
                            
                            if not self.targetcount == self.numtargets:
                                if self.distance == self.threshold:
                                    self.prompt3.remove()
                                    if self.rest_time > 0:
                                        rest_second = self.rest_time
                                        for i in range(int(self.rest_time)):
                                            self.prompt2 = self.axis1.text(0,3*(self.maxval-self.threshold)/4+self.threshold,'Next Target in %.0f' % rest_second,size=20,ha='center',va='center',bbox=dict(boxstyle = 'square',color='gray',alpha=0.3),color='g')
                                            rest_second -= 1
                                            plt.pause(1)
                                            if self.prompt2:
                                                self.prompt2.remove()
                                    elif self.rest_time == 0:
                                        self.prompt4 = self.axis1.text(0,3*(self.maxval-self.threshold)/4+self.threshold,'Press ENTER for Next Target',size=20,ha='center',va='center',bbox=dict(boxstyle='square',color='gray',alpha=0.3),color='g')
                                        plt.pause(0.001)
                                        plt.waitforbuttonpress()
                                        self.prompt4.remove()
                                    timerbool = True
                                    plt.pause(0.001)
                                    self._distance_update()
                                    self._target_bar_update()
                                elif self.distance > self.threshold:
                                    datadict['Movement'] = self.movement
                                    datadict['Distance'] = self.distance
                                    datadict['Width'] = self.target_width
                                    datadict['Completed'] = False
                                    datadict['Target Range'] = (self.distance-self.target_width/2,self.distance+self.target_width/2)
                                    finished = time.perf_counter()-timer
                                    movement = finished
                                    datadict['Completion Time'] = finished
                                    datadict['Movement Time'] = movement

                                    datadictcopy = copy.deepcopy(datadict)
                                    parentdict[self.targetcount] = datadictcopy

                                    datadict['data'] = []

                                    self.targetcount+=1
                                    self.prompt3 = self.axis1.text(0,3*(self.maxval-self.threshold)/4+self.threshold,'Failed! \nReturn to REST',size=20,ha='center',va='center',bbox=dict(boxstyle = 'square',color='gray',alpha=0.3),color='r')
                                    self._rest_update()
                                    self._target_bar_update()
                                    self._update_target()
                                    
                            if self.targetcount == self.numtargets:
                                    self._clear_fig()
                                    donetext = self.axis1.text(0.5,0.5,'Done! Saving data...',size=35,ha='center',va='center',bbox=dict(boxstyle = 'round',color='palegreen',alpha=0.3))
                                    plt.pause(1)
                                    donetext.remove()

                                    filename = 'SLData_Subject_%d_T%s_%s_FS%s_VS%s' % (int(SUBJECT), (self.trialcount+1), self.method, self.feat_smoothing, self.vel_smoothing)                                    
                                    self.doubleparentdict[filename] = parentdict

                                    '''folder = '.\Data\Subject_%d' % int(SUBJECT)
                                    folder2 = 'SignalLevelData'
                                    filename = 'SLData_Subject_%d_T%s_%s_FS%s_VS%s.mat' % (int(SUBJECT), (self.trialcount+1), self.method, self.feat_smoothing, self.vel_smoothing)
                                    if not path.exists(folder):
                                        os.mkdir(folder)
                                    if not path.exists(folder + '/' + folder2):
                                        os.mkdir(folder + '/' + folder2)
                                    filepath = folder + '/' + folder2 + '/' + filename
                                    if not path.exists(filepath):
                                        f = open(filepath, "w")
                                        f.write("")
                                        f.close()

                                    with open(filepath, 'wb') as f:
                                        pickle.dump(parentdict,f)
                                    with open(filepath, 'rb') as f:
                                        mydict = pickle.load(f)
                                    print(mydict)'''

                                    self.trialcount += 1
                                    if self.trialcount < self.numtrials:
                                        text1 = '\n\n'.join(('Save Complete','Press ENTER to Continue to Next Trial'))
                                        self.prompt = self.axis1.text(0.5,0.7,text1,size=25,ha='center',va='center',bbox=dict(boxstyle='round',color='lightgreen',alpha=0.3))
                                        self._restart()
                                    elif self.trialcount == self.numtrials:
                                        folder = '.\Data\Subject_%d' % int(SUBJECT)
                                        folder2 = 'SignalLevelData'
                                        filename = 'SLData_%s_FS%s_VS%s.pkl' % (self.method, self.feat_smoothing, self.vel_smoothing)
                                        if not path.exists(folder):
                                            os.mkdir(folder)
                                        if not path.exists(folder + '/' + folder2):
                                            os.mkdir(folder + '/' + folder2)
                                        filepath = folder + '/' + folder2 + '/' + filename
                                        if not path.exists(filepath):
                                            f = open(filepath, "w")
                                            f.write("")
                                            f.close()

                                        with open(filepath, 'wb') as f:
                                            pickle.dump(self.doubleparentdict,f)
                                        with open(filepath, 'rb') as f:
                                                mydict = pickle.load(f)
                                        print(mydict)

                                        self._exit_event.set()
                                    plt.pause(1)


                    
            plt.pause(0.001)
        plt.close()
        
    def _update_bar(self, height ):
        self.barplot.set_height(height)
        self.height = height

    def _update_target(self):
        try:
            self.targettext.remove()
        except:
            pass
        if self.targetcount != self.numtargets:
            self.targettext = self.fig1.text(0.9,0.95,'Target: %d/%d' % (self.targetcount+1,self.numtargets),size=20,ha='center',va='center',bbox=dict(boxstyle = 'square',color='gray',alpha=0.3),color='g')

    def _plot(self):
        self.input = Parameters['Signal Level']['Input']
        self.holdtime = Parameters['Signal Level']['Hold Time']
        self.timeout = Parameters['Signal Level']['Timeout']
        self.filename = Parameters['Signal Level']['File']
        self.numtrials = Parameters['Signal Level']['Trials']
        self.rest_time = Parameters['Signal Level']['Rest Time']
        self.movement = Parameters['Signal Level']['Movement']
        self.buffsize = Parameters['Signal Level']['Smoothing']

        try:
            '''if Parameters['Signal Level']['PROPORTIONAL_METHOD'] == 'Amplitude':
                x = loadmat('./matdata/calibratesignallevel.mat')
                self.lower = x['lower'][0][0]
                self.upper = x['upper'][0][0]
            elif Parameters['Signal Level']['PROPORTIONAL_METHOD'] == 'Spatial':
                self.lower = 0
                self.upper = 1'''

            self.lower = 0
            self.upper = 1
        except:
            print('No Calibration File Found')

        self.fig1 = plt.figure()
        mngr = plt.get_current_fig_manager()

        self.axis1 = self.fig1.add_subplot(111)
        plt.xlim(-0.5,0.5)
        plt.ylim(-1,1)
        mngr.window.setGeometry(500,100,900,900)
        mngr.window.setWindowTitle( "Signals" )

        self.axis1.axis('off')

        textstr = '\n\n'.join((
                            'Input: %s' % self.input,'Movement: %s' % self.movement,
                            'Hold Time: %.2f sec(s)' % self.holdtime,
                            'Timeout: %i sec(s)' % self.timeout,
                            'Trials: %i' % self.numtrials,
                            'Rest Time: %i sec(s)' % self.rest_time,
                            'Smoothing Level: %i' % self.buffsize,
                            'File Name: %s' % self.filename
                            ))

        self.settings = self.axis1.text(0,1,textstr,fontsize=20,horizontalalignment = 'center',verticalalignment = 'top', bbox=dict(boxstyle='round',facecolor='wheat',alpha=0.5))


        #create a button and make it respond to click
        self.buttonbox = plt.axes([0.25, 0.15, 0.52, 0.075])
        button = Button(self.buttonbox, 'START')
        button.on_clicked(self._signallevelplot)

        plt.show()

    def add(self,ctrl):
        try:
            self._queue.put(ctrl,timeout=1e-3)
        except queue.Full:
            pass

    def close(self):
        self._exit_event.set()
        while self._queue.qsize() > 0:
            try:
                self._queue.get(timeout = 1e-3)
            except queue.Empty:
                pass


if __name__ == '__main__':

    #create data filters
    print( 'Creating EMG feature filters...', end = '', flush = True )
    td5 = TimeDomainFilter()
    fft = FourierTransformFilter( fftlen = Parameters['General']['FFT_LENGTH'] )
    print( 'Done!' )

    DIRECTORY = os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ), 'data', 'Subject_%s' % SUBJECT, './calibration_data/calibratesignallevel_subject_%s.mat' % SUBJECT)
    x = loadmat(DIRECTORY)
    Xtrain = x['Xtrain']
    ytrain = x['ytrain'].flatten()
    onset_threshold = x['lower'][0][0]
    print( 'Training classifier...', end = '', flush = True )
    class_subdict_size = int( x['ytrain'].shape[1] / 2)
    run_classifier = RunClassifier(   class_subdict_size = class_subdict_size, 
                                            Xtrain = Xtrain, 
                                            ytrain = ytrain, 
                                            classes = ['REST',Parameters['Signal Level']['Movement']],
                                            perc_lo = [0 for i in range(2)], 
                                            perc_hi = [0.75 for i in range(2)],
                                            threshold_ratio = Parameters['Classification']['THRESHOLD_RATIO'] )
    print( 'Done!' )

    print( 'Creating proportional control filters...', end = '', flush = True )

    prop = Proportional_Control(    X = Xtrain.T, 
                                    L = ytrain, 
                                    onset_threshold = onset_threshold,
                                    classes = ['REST',Parameters['Signal Level']['Movement']] )
    print( 'Done!' )

    # create EMG interface
    print( 'Creating device interface...', end = '', flush = True )
    if Parameters['General']['SOURCE'] == 'Sense':
        emg = SenseController( name = local_surce_info.sense_name,
                                    mac = local_surce_info.sense_mac,
                                    gain = Parameters['General']['ELECTRODE_GAIN'],
                                    num_electrodes = Parameters['General']['DEVICE_ELECTRODE_COUNT'],
                                    srate = Parameters['General']['SAMPLING_RATE'] )
    elif Parameters['General']['SOURCE'] == 'Myo Band':
        emg = MyoArmband( name = local_surce_info.myo_name, mac = local_surce_info.myo_mac )
    else:
        raise RuntimeError( 'Unsupported EMG interface provided' )
    print( 'Done!' )

    emg.run()

    SL = SignalLevel(method = Parameters['Signal Level']['PROPORTIONAL_METHOD'], feat_smoothing = Parameters['Signal Level']['FILTER_LENGTH'], vel_smoothing = Parameters['Signal Level']['Smoothing'])

    emg_window = []

    buffsize = Parameters['Signal Level']['FILTER_LENGTH']
    prop_buffsize = Parameters['Signal Level']['Smoothing']

    inputmode = Parameters['Signal Level']['Input']
    n = Parameters['Signal Level']['Electrode']
    
    feat_buffer = deque()
    prop_buffer = deque()
    
    while not SL._exit_event.is_set():
        data = emg.state

        if data is not None:
            emg_window.append( data[:Parameters['General']['DEVICE_ELECTRODE_COUNT']].copy() / EMG_SCALING_FACTOR )
        
        if len( emg_window ) == Parameters['General']['WINDOW_SIZE']:
            win = np.vstack( emg_window )
            feat = np.hstack( [ td5.filter( win ), fft.filter( win ) ] )[:-Parameters['General']['DEVICE_ELECTRODE_COUNT']]                 
            emg_window = emg_window[Parameters['General']['WINDOW_STEP']:]

            if inputmode == 'Pattern Recognition':
                feat_buffer.append(feat)
                if len(feat_buffer) > buffsize:
                    while len(feat_buffer) > buffsize:
                        feat_buffer.popleft()

                feat_filtered = np.mean(feat_buffer, axis=0).reshape(1,-1)

                pred, prop_vel = run_classifier.emg_classify(feat_filtered, x['lower'], 'Spatial', Parameters['Classification']['ONSET_THRESHOLD'], return_aux_data = True)

                if Parameters['Signal Level']['PROPORTIONAL_METHOD'] == 'Amplitude':

                    mean_mav = np.mean(feat_filtered[:,:8])

                    if mean_mav < onset_threshold:
                        prop_vel = 0
                    else:
                        prop_vel = prop.Proportional( feat_filtered, pred, 1 )

                elif Parameters['Signal Level']['PROPORTIONAL_METHOD'] == 'Spatial':
                    
                    if len(prop_vel) == 0:
                        prop_vel = 0
                        obj_prop_vel = 0
                    else:
                        obj_prop_vel = prop_vel[1]
                        prop_vel = prop_vel[0]
                    
                    if type(prop_vel) != int:

                        prop_vel = prop_vel[pred-1]
                        obj_prop_vel = obj_prop_vel[pred-1]

                prop_buffer.append(prop_vel)
                while len(prop_buffer) > prop_buffsize:
                    prop_buffer.popleft()
                
                SL.add(np.mean(prop_buffer))
            elif inputmode == 'Single Electrode':
                feat_buffer.append(feat)
                if len(feat_buffer) > buffsize:
                    while len(feat_buffer) > buffsize:
                        feat_buffer.popleft()
                    feat_filtered = np.mean(feat_buffer, axis=0).reshape(1,-1)
                    SL.add(np.mean(copy.deepcopy(feat_filtered[:,n])))
    emg.close()