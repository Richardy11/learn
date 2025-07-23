import csv
import os
import matplotlib
import pickle
import matplotlib.pyplot as plt
from matplotlib.widgets import Button,TextBox
from mpl_interactions.widgets import RangeSlider
import multiprocessing as mp
import numpy as np
from scipy.io import savemat,loadmat
import random
import copy
import math

SUBJECT = 1

PARAM_FILE = os.path.join('data', 'Subject_%s' % SUBJECT, 'RESCU_Protocol_Subject_%s.pkl' % SUBJECT )

with open( PARAM_FILE, 'rb' ) as pkl:
    Parameters = pickle.load( pkl )

class ChooseTargets:
    def __init__(self):
        self.filename = Parameters['Signal Level']['File']
        
        self._plotter = mp.Process(target = self._plot)

        self._plotter.start()
        

    def _get_target_dist(self,val):
        distances = val
        if ',' in distances:
            self.distances = distances.split(',')
            counter = 0
            for i in self.distances:
                i = float(i)
                self.distances[counter] = i
                counter += 1
            self.distok = True
        elif '-' in distances:
            self.distances = distances.split('-')
            start = float(self.distances[0])
            stop = float(self.distances[1])
            step = float(self.distances[2])
            samples = int((stop-start)/step+1)
            if samples != ((stop-start)/step+1):
                self.distok = False
            else:
                self.distances = np.linspace(start,stop,samples)
                self.distok = True

    def _get_target_width(self,val2):
        widths = val2
        if ',' in widths:
            self.widths = widths.split(',')
            counter2 = 0
            for i in self.widths:
                i = float(i)
                self.widths[counter2] = i
                counter2 += 1
            self.widthok = True
        elif '-' in widths:
            self.widths = widths.split('-')
            start = float(self.widths[0])
            stop = float(self.widths[1])
            step = float(self.widths[2])
            samples = int((stop-start)/step+1)
            self.widths = np.linspace(start,stop,samples)
            if samples != ((stop-start)/step+1):
                self.widthok = False
            else:
                self.widthok = True
                self.widths = np.linspace(start,stop,samples)

    
    def _save_values(self,event):
        '''try:
            if self.distok == False or self.widthok == False:
                self._error_check()
            else:
                inputloc = os.path.join('Input',self.filename)
                rows = [self.distances,self.widths]
                with open(inputloc, mode='w',newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerows(rows)
                plt.close()
        except AttributeError:
            self.error = plt.text(-1.5,3.2,'Error! Missing Inputs',va='top',color='red')'''

        if self.distok == False or self.widthok == False:
            self._error_check()
        else:
            inputloc = os.path.join('Input',self.filename)
            rows = [self.distances,self.widths]
            with open(inputloc, mode='w',newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerows(rows)
            plt.close()


    def _save_as_default(self,event):
        if self.distok == False or self.widthok == False:
            self._error_check()
        else:
            savemat('./matdata/signalleveldefault.mat',mdict = {'distances': self.distances,'widths': self.widths})
            self._clear_text()
            self.saved = plt.text(-1.5,3.35,'Targets Saved! \nDistances: {0} \nWidths: {1} \nPress SUBMIT to use these targets'.format(self.distances,self.widths),va='top',color='red')
        

    def _load_default(self,event):
        x = loadmat('./matdata/signalleveldefault.mat')
        self.distances = x['distances'][0]
        self.widths = x['widths'][0]
        self._clear_text()
        self.loaded = plt.text(-1.5,3.35,'Targets Loaded! \nDistances: {0} \nWidths: {1}\nPress SUBMIT to use these targets'.format(self.distances,self.widths),va='top',color='red')

    def _error_check(self):
        self._clear_text()
        if self.distok == False:
            self.error = plt.text(-1.5,3.3,'Error! Distances Step is not compatible with range',va='top',color='red')
        if self.widthok == False:
            self.error2 = plt.text(-1.5,3.1,'Error! Widths Step is not compatible with range',va='top',color='red')

    def _clear_text(self):
        try: self.error.remove()
        except: pass
        try: self.error2.remove()
        except: pass
        try: self.saved.remove()
        except: pass
        try: self.loaded.remove()
        except: pass

    def _plot(self):
        fig = plt.figure()
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
        mngr = plt.get_current_fig_manager()
        axis = fig.add_subplot(111)
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        axis.axis('off')
        mngr.window.setGeometry(500,100,900,900)

        text = plt.text(0,1,'Signal Level Setup',fontsize = 30,horizontalalignment = 'center',verticalalignment = 'top', bbox=dict(boxstyle='round',facecolor='wheat',alpha=0.5))

        text1 = plt.text(-0.81,0.63,'Enter All Desired Target Distances in % of MCV (e.g. 40,50,60)\nRanges can be entered with "-" (e.g. 40-60-10 = 40,50,60)',va='top')
        distbox = plt.axes([0.2,0.6,0.6,0.1])
        distance = TextBox(distbox,'')
        distance.on_submit(self._get_target_dist)

        text2 = plt.text(0,-0.63,'Enter All Desired Target Widths in % of MCV (e.g. 10,12.5,15)\nRanges can be entered with "-" (e.g.: 10-15-2.5 = 10,12.5,15)',va='top')
        widthbox = plt.axes([0.2,0.4,0.6,0.1])
        width = TextBox(widthbox,'')
        width.on_submit(self._get_target_width)

        submitbox = plt.axes([0.1, 0.1, 0.4, 0.18])
        submitbutton = Button(submitbox, 'SUBMIT')
        submitbutton.on_clicked(self._save_values)

        defaultsave = plt.axes([0.55,0.2,0.3,0.08])
        savedefault = Button(defaultsave, 'SAVE AS DEFAULT')
        savedefault.on_clicked(self._save_as_default)

        defaultload = plt.axes([0.55,0.1,0.3,0.08])
        loaddefault = Button(defaultload,'LOAD DEFAULT')
        loaddefault.on_clicked(self._load_default)

        plt.show()

class ChooseTargetsFL:
    def __init__(self):
        self.filename = Parameters['Fitts Law']['File']

        self.rows = []

        inputloc = './Input/fittslawtargets.pkl'

        self.DoF_mov_list = {'Targets': {
                                        'xcoordinate': (0,1),
                                        'ycoordinate': (0,1),
                                        'target_rangex': (0.04,0.08),
                                        'target_rangey': (0.04,0.08),
                                        'radius': (0.075,0.5),
                                        'radius_offset': (0.02,0.03),
                                        'rim_target': (-180,180),
                                        'rim_target_offset': (15,45)
        } }

        try:
            with open( inputloc, 'rb' ) as f:
                    self.DoF_mov_list = pickle.load(f)
        except:
            pass

        self.axis_side = [0,0,0,0]

        
        if 'OPEN' in Parameters['Fitts Law']['CUE_LIST']:
            self.griplist = copy.deepcopy(Parameters['Fitts Law']['GRIP_CUE_LIST'])
        else:
            self.griplist = []


        self._plotter = mp.Process(target = self._plot)

        self._plotter.start()

    def _update_xcoord_values(self,val):
        self.DoF_mov_list['Targets']['xcoordinate']  = val

    def _update_ycoord_values(self,val):
        self.DoF_mov_list['Targets']['ycoordinate'] = val

    def _update_radius(self,val):
        self.DoF_mov_list['Targets']['radius'] = val
    
    def _update_radius_offset(self,val):
        self.DoF_mov_list['Targets']['radius_offset'] = val
    
    def _update_target_rangex(self,val):
        self.DoF_mov_list['Targets']['target_rangex'] = val
    
    def _update_target_rangey(self,val):
        self.DoF_mov_list['Targets']['target_rangey'] = val
    
    def _update_rim_target(self,val):
        self.DoF_mov_list['Targets']['rim_target'] = val

    def _update_rim_target_offset(self,val):
        self.DoF_mov_list['Targets']['rim_target_offset'] = val

    def lcm(self, x, y):
        if x > y:
            greater = x
        else:
            greater = y

        while(True):
            if ((greater % x == 0) and (greater % y == 0)):
                lcm = greater
                break
            greater += 1

        return lcm

    def _generate_targets(self):

        num_act_hand = 0
        num_act = 0
        self.non_grip = []

        for i in range(4):
            self.DoF_mov_list[i] = {    'Movements': [],
                                        'Target Position': [],
                                        'Target Range': [],
                                        'Target ID': [] }

            if Parameters['Fitts Law']['DoF'][i]['MOVEMENT_TYPE'] == 'Hand':
                num_act_hand = len(self.griplist)

            elif Parameters['Fitts Law']['DoF'][i]['MOVEMENT_TYPE'] != 'Off':
                num_act += +2
                if Parameters['Fitts Law']['DoF'][i]['MOVEMENT_TYPE'] == 'Wrist':
                    if Parameters['Fitts Law']['DoF'][i]['WRIST_DOF'] == 'ROTATION':
                        self.non_grip.append('PALM DOWN')
                        self.non_grip.append('PALM UP')
                    elif Parameters['Fitts Law']['DoF'][i]['WRIST_DOF'] == 'FLEXION':
                        self.non_grip.append('FLEXION')
                        self.non_grip.append('EXTENSION')
                elif Parameters['Fitts Law']['DoF'][i]['MOVEMENT_TYPE'] == 'Elbow':
                    self.non_grip.append('ELBOW BEND')
                    self.non_grip.append('ELBOW EXTEND')

        lcm = self.lcm(num_act_hand, 2)

        grip_iter = 0
        paired_mov_iter = 0
        wrist_movs = [['PALM DOWN', 'PALM UP'], ['FLEXION', 'EXTENSION']]
        elbow_movs = ['ELBOW BEND', 'ELBOW EXTEND']

        for i in range(Parameters['Fitts Law']['Trials']):
            
            for j in range(lcm):

                for k in range(4):
                    if Parameters['Fitts Law']['DoF'][k]['MOVEMENT_TYPE'] == 'Hand':
                        self.DoF_mov_list[k]['Movements'].append(self.griplist[grip_iter%num_act_hand])

                    elif Parameters['Fitts Law']['DoF'][k]['MOVEMENT_TYPE'] != 'Off':
                        
                        if Parameters['Fitts Law']['DoF'][k]['MOVEMENT_TYPE'] == 'Wrist':
                            if Parameters['Fitts Law']['DoF'][k]['WRIST_DOF'] == 'ROTATION':
                                self.DoF_mov_list[k]['Movements'].append(wrist_movs[0][paired_mov_iter%2])
                            elif Parameters['Fitts Law']['DoF'][k]['WRIST_DOF'] == 'FLEXION':
                                self.DoF_mov_list[k]['Movements'].append(wrist_movs[1][paired_mov_iter%2])
                        elif Parameters['Fitts Law']['DoF'][k]['MOVEMENT_TYPE'] == 'Elbow':
                            self.DoF_mov_list[k]['Movements'].append(elbow_movs[paired_mov_iter%2]) 

                    if k == 0:
                        start = (self.rad[0]+self.rad[1])/2
                        self.DoF_mov_list[k]['Target Position'].append((self.rad[0]+self.rad[1])/2)
                        self.DoF_mov_list[k]['Target Range'].append(random.uniform(self.radoffset[0],self.radoffset[1]))
                        midpoint = 0.2875#0.2125
                    elif k == 1:
                        start = (self.rim_target_angle[0]+self.rim_target_angle[1])/2
                        self.DoF_mov_list[k]['Target Position'].append(0)
                        self.DoF_mov_list[k]['Target Range'].append(random.uniform(self.rim_tolerance[0],self.rim_tolerance[1]))
                        midpoint = 0
                    elif k == 2:
                        start = (self.x_cent[0]+self.x_cent[1])/2
                        self.DoF_mov_list[k]['Target Position'].append((self.x_cent[0]+self.x_cent[1])/2)
                        self.DoF_mov_list[k]['Target Range'].append(random.uniform(self.x_target_range[0],self.x_target_range[1]))
                        midpoint = 0.5
                    elif k == 3:
                        start = (self.y_cent[0]+self.y_cent[1])/2
                        self.DoF_mov_list[k]['Target Position'].append((self.y_cent[0]+self.y_cent[1])/2)
                        self.DoF_mov_list[k]['Target Range'].append(random.uniform(self.y_target_range[0],self.y_target_range[1]))
                        midpoint = 0.5

                    #TODO no grip case
                    while self.DoF_mov_list[k]['Target Position'][-1] + self.DoF_mov_list[k]['Target Range'][-1]*2 > midpoint and self.DoF_mov_list[k]['Target Position'][-1] - self.DoF_mov_list[k]['Target Range'][-1]*2 < midpoint:
                        
                        if Parameters['Fitts Law']['DoF'][k]['MOVEMENT_TYPE'] == 'Hand':
                            
                            if self.DoF_mov_list[k]['Movements'][-1] == 'OPEN':
                                if k == 0:
                                    self.DoF_mov_list[k]['Target Position'][-1] = random.uniform(midpoint+self.DoF_mov_list[k]['Target Range'][-1]*2,self.rad[1])
                                elif k == 1:
                                    self.DoF_mov_list[k]['Target Position'][-1] = random.uniform(start,self.rim_target_angle[1])
                                elif k == 2:
                                    self.DoF_mov_list[k]['Target Position'][-1] = random.uniform(midpoint,self.x_cent[1])
                                elif k == 3:
                                    self.DoF_mov_list[k]['Target Position'][-1] = random.uniform(midpoint,self.y_cent[1])
                            else:
                                if k == 0:
                                    self.DoF_mov_list[k]['Target Position'][-1] = random.uniform(self.rad[0],midpoint-self.DoF_mov_list[k]['Target Range'][-1]*2)
                                elif k == 1:
                                    self.DoF_mov_list[k]['Target Position'][-1] = random.uniform(self.rim_target_angle[0],start)
                                elif k == 2:
                                    self.DoF_mov_list[k]['Target Position'][-1] = random.uniform(self.x_cent[0],midpoint)
                                elif k == 3:
                                    self.DoF_mov_list[k]['Target Position'][-1] = random.uniform(self.y_cent[0],midpoint)
                        else:
                            if self.axis_side[0]:
                                if k == 0:
                                    self.DoF_mov_list[k]['Target Position'][-1] = random.uniform(self.rad[0],midpoint-self.DoF_mov_list[k]['Target Range'][-1]*2)
                                elif k == 1:
                                    self.DoF_mov_list[k]['Target Position'][-1] = random.uniform(self.rim_target_angle[0],start)
                                elif k == 2:
                                    self.DoF_mov_list[k]['Target Position'][-1] = random.uniform(midpoint,self.x_cent[1])
                                elif k == 3:
                                    self.DoF_mov_list[k]['Target Position'][-1] = random.uniform(midpoint,self.y_cent[1])
                                    
                            else:
                                if k == 0:
                                    self.DoF_mov_list[k]['Target Position'][-1] = random.uniform(midpoint+self.DoF_mov_list[k]['Target Range'][-1]*2,self.rad[1])
                                elif k == 1:
                                    self.DoF_mov_list[k]['Target Position'][-1] = random.uniform(start,self.rim_target_angle[1])
                                elif k == 2:
                                    self.DoF_mov_list[k]['Target Position'][-1] = random.uniform(self.x_cent[0],midpoint)
                                elif k == 3:
                                    self.DoF_mov_list[k]['Target Position'][-1] = random.uniform(self.y_cent[0],midpoint)

                    if k == 1:
                        self.DoF_mov_list[k]['Target ID'].append( math.log2( ((self.DoF_mov_list[k]['Target Position'][-1]+180)/360) /(self.DoF_mov_list[k]['Target Range'][-1]/360) + 1) )
                    else:
                        self.DoF_mov_list[k]['Target ID'].append( math.log2(self.DoF_mov_list[k]['Target Position'][-1]/self.DoF_mov_list[k]['Target Range'][-1] + 1) )

                    if Parameters['Fitts Law']['DoF'][k]['MOVEMENT_TYPE'] == 'Hand':
                        self.axis_side[k] = (self.axis_side[k]+1)%len(self.griplist)
                    else:
                        self.axis_side[k] = (self.axis_side[k]+1)%2
                
                grip_iter += 1
                paired_mov_iter += 1
                
        print('center:', self.DoF_mov_list[0]['Target ID'], 'offset:', self.DoF_mov_list[1]['Target ID'])
        print(len(self.DoF_mov_list[0]['Target ID']))

    def _submit_values(self,event):
        self._save()
        inputloc = './Input/fittslawtargets.pkl'
        with open( inputloc, 'wb' ) as f:
                pickle.dump(self.DoF_mov_list, f)
        plt.close()

    def _reset(self,event):
        self.xcoord.reset()
        self.ycoord.reset()
        self.xrange.reset()
        self.yrange.reset()
        self.rad.reset()
        self.radoff.reset()
        self.rim.reset()
        self.rimoff.reset()

    def _save(self):
        self.bool = False
        try:
            self.errortext.remove()
        except:
            pass

        self.x_cent = []
        self.y_cent = []
        self.rad = []
        self.radoffset = []
        self.x_target_range = []
        self.y_target_range = []
        self.rim_target_angle = []
        self.rim_tolerance = []

        self.x_cent.append(round(self.DoF_mov_list['Targets']['xcoordinate'][0],3) )
        self.x_cent.append(round(self.DoF_mov_list['Targets']['xcoordinate'][1],3) )

        self.y_cent.append(round(self.DoF_mov_list['Targets']['ycoordinate'][0],3) )
        self.y_cent.append(round(self.DoF_mov_list['Targets']['ycoordinate'][1],3) )

        self.rad.append(round(self.DoF_mov_list['Targets']['radius'][0],3) )
        self.rad.append(round(self.DoF_mov_list['Targets']['radius'][1],3) )

        self.radoffset.append(round(self.DoF_mov_list['Targets']['radius_offset'][0],3) )
        self.radoffset.append(round(self.DoF_mov_list['Targets']['radius_offset'][1],3) )

        self.x_target_range.append(round(self.DoF_mov_list['Targets']['target_rangex'][0],3) )
        self.x_target_range.append(round(self.DoF_mov_list['Targets']['target_rangex'][1],3) )

        self.y_target_range.append(round(self.DoF_mov_list['Targets']['target_rangey'][0],3) )
        self.y_target_range.append(round(self.DoF_mov_list['Targets']['target_rangey'][1],3) )

        self.rim_target_angle.append(round(self.DoF_mov_list['Targets']['rim_target'][0],3) )
        self.rim_target_angle.append(round(self.DoF_mov_list['Targets']['rim_target'][1],3) )

        self.rim_tolerance.append(round(self.DoF_mov_list['Targets']['rim_target_offset'][0],3) )
        self.rim_tolerance.append(round(self.DoF_mov_list['Targets']['rim_target_offset'][1],3) )

        if self.x_cent[0] >= 0.5:
            self.errortext = plt.text(-2,3,'Error! X Coordinate Min must be less than 0.5',ha='center',va='top',fontsize=10,color = 'r')
        elif self.x_cent[1] <= 0.5:
            self.errortext = plt.text(-2,3,'Error! X Coordinate Max must be greater than 0.5',ha='center',va='top',fontsize=10,color = 'r')
        elif self.y_cent[0] >= 0.5:
            self.errortext = plt.text(-2,3,'Error! Y Coordinate Min must be less than 0.5',ha='center',va='top',fontsize=10,color = 'r')
        elif self.y_cent[1] <= 0.5:
            self.errortext = plt.text(-2,3,'Error! Y Coordinate Max must be greater than 0.5',ha='center',va='top',fontsize=10,color = 'r')
        elif self.rad[0] >= 0.2125:
            self.errortext = plt.text(-2,3,'Error! Radius Min must be less than 0.2125',ha='center',va='top',fontsize=10,color = 'r')
        elif self.rad[1] <= 0.2125:
            self.errortext = plt.text(-2,3,'Error! Radius Max must be greater than 0.2125',ha='center',va='top',fontsize=10,color = 'r')
        elif self.rim_target_angle[0] >= 0:
            self.errortext = plt.text(-2,3,'Error! Radius Min must be less than 0',ha='center',va='top',fontsize=10,color = 'r')
        elif self.rim_target_angle[1] <= 0:
            self.errortext = plt.text(-2,3,'Error! Radius Max must be greater than 0',ha='center',va='top',fontsize=10,color = 'r')
        else:
            self.bool = True
            self.rows.append([self.x_cent[0],self.x_cent[1],self.y_cent[0],self.y_cent[1],self.x_target_range[0],self.x_target_range[1],self.y_target_range[0],self.y_target_range[1]
                            ,self.rad[0],self.rad[1],self.radoffset[0],self.radoffset[1],self.rim_target_angle[0],self.rim_target_angle[1],self.rim_tolerance[0],self.rim_tolerance[1] ])
        
        self._generate_targets()

    def _load(self,event):
        try:
            inputloc = './Input/fittslaw.csv'
            x = loadmat('./matdata/fittslawtargets.mat')
            rows = x['rows']
            with open(inputloc, mode='w',newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerows(rows)
            plt.close()
        except:
            print('No File to Load')

    
    def _plot(self):
        fig = plt.figure()
        mngr = plt.get_current_fig_manager()
        plt.axis('off')

        def _close(event):
            plt.close()

        mngr.window.setGeometry(500,100,900,900)

        text = plt.text(0.5,1.1,'Fitts Law Test Setup',ha='center',va='top',fontsize=30,bbox=dict(boxstyle='round',facecolor='wheat',alpha=0.5))

        xcoordslider = plt.axes([0.1,0.8,0.75,0.05])
        self.xcoord = RangeSlider(xcoordslider,'X-Axis \nTarget',valmin = 0,valmax = 1,valinit = (self.DoF_mov_list['Targets']['xcoordinate'][0],self.DoF_mov_list['Targets']['xcoordinate'][1]),valfmt = '%0.3f')
        self.xcoord.on_changed(self._update_xcoord_values)

        ycoordslider = plt.axes([0.1,0.72,0.75,0.05])
        self.ycoord = RangeSlider(ycoordslider,'Y-Axis \nTarget',valmin=0,valmax=1,valinit = (self.DoF_mov_list['Targets']['ycoordinate'][0],self.DoF_mov_list['Targets']['ycoordinate'][1]),valfmt = '%0.3f')
        self.ycoord.on_changed(self._update_ycoord_values)

        xrangeslider = plt.axes([0.1,0.64,0.75,0.05])
        self.xrange = RangeSlider(xrangeslider,'X-Axis \nTarget \nRange',valmin=0.02,valmax=0.1,valinit = (self.DoF_mov_list['Targets']['target_rangex'][0],self.DoF_mov_list['Targets']['target_rangex'][1]),valfmt = '%0.3f')
        self.xrange.on_changed(self._update_target_rangex)

        yrangeslider = plt.axes([0.1,0.56,0.75,0.05])
        self.yrange = RangeSlider(yrangeslider,'Y-Axis \nTarget \nRange',valmin=0.02,valmax=0.1,valinit = (self.DoF_mov_list['Targets']['target_rangey'][0],self.DoF_mov_list['Targets']['target_rangey'][1]),valfmt = '%0.3f')
        self.yrange.on_changed(self._update_target_rangey)

        radslider = plt.axes([0.1,0.48,0.75,0.05])
        self.rad = RangeSlider(radslider,'Circle \nTarget \nRadius',valmin=0.075,valmax=0.5,valinit = (self.DoF_mov_list['Targets']['radius'][0],self.DoF_mov_list['Targets']['radius'][1]),valfmt = '%0.3f')
        self.rad.on_changed(self._update_radius)

        radoffslider = plt.axes([0.1,0.4,0.75,0.05])
        self.radoff = RangeSlider(radoffslider,'Circle \nTarget \nRadius \nOffset',valmin=0.01,valmax=0.05,valinit = (self.DoF_mov_list['Targets']['radius_offset'][0],self.DoF_mov_list['Targets']['radius_offset'][1]),valfmt = '%0.3f')
        self.radoff.on_changed(self._update_radius_offset)

        rimslider = plt.axes([0.1,0.32,0.75,0.05])
        self.rim = RangeSlider(rimslider,'Rim \nTarget \nAngle',valmin=-180,valmax=180,valinit = (self.DoF_mov_list['Targets']['rim_target'][0],self.DoF_mov_list['Targets']['rim_target'][1]),valfmt = '%0.0f')
        self.rim.on_changed(self._update_rim_target)

        rimoffslider = plt.axes([0.1,0.24,0.75,0.05])
        self.rimoff = RangeSlider(rimoffslider,'Rim \nTarget \nOffset',valmin=5,valmax=90,valinit = (self.DoF_mov_list['Targets']['rim_target_offset'][0],self.DoF_mov_list['Targets']['rim_target_offset'][1]),valfmt = '%0.0f')
        self.rimoff.on_changed(self._update_rim_target_offset)

        submitbox = plt.axes([0.03,0.02,0.18,0.05])
        submit = Button(submitbox,'SAVE')
        submit.on_clicked(self._submit_values)

        resetbox = plt.axes([0.6,0.02,0.18,0.05])
        reset = Button(resetbox,'RESET')
        reset.on_clicked(self._reset)

        '''loadbox = plt.axes([0.41,0.02,0.18,0.05])
        load = Button(loadbox,'LOAD FROM FILE')
        load.on_clicked(self._load)'''

        exitbox = plt.axes([0.79,0.02,0.18,0.05])
        exit = Button(exitbox,'EXIT')
        exit.on_clicked(_close)

        plt.show()

if __name__ == '__main__':
    if Parameters['General']['FL'] == True:
        CT = ChooseTargetsFL()
    else:
        CT = ChooseTargets()
