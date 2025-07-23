import pickle as pkl
import csv
from typing import Tuple

import numpy as np

import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button

import multiprocessing as mp

import copy

FS = 40

VS = 50

class DataAnalysis:
    def __init__(self,SUBJECT):
        self.subject = SUBJECT
        folder = '.\Data\Subject_%d' % int(self.subject)
        folder2 = 'SignalLevelData'
        filename = 'SLData_%s_FS%s_VS%s.pkl' % ('Spatial', FS, VS)
        with open (folder + '\\' + folder2 + '\\' + filename,'rb') as f:
            self.Data = pkl.load(f)

        self.indexlist = []
        self.indexlist2 = []
        self.timelist = []
        self.timelist2 = []

        folder = '.\Data\Subject_%d' % int(self.subject)
        folder2 = 'SignalLevelData'
        filename = 'SLData_%s_FS%s_VS%s.pkl' % ('Amplitude', FS, VS)
        with open (folder + '\\' + folder2 + '\\' + filename,'rb') as f:
            self.Data2 = pkl.load(f)
        
        self.indexlist3 = []
        self.indexlist4 = []
        self.timelist3 = []
        self.timelist4 = []
        self.totalindex1 = []
        self.totaltime1 = []
        self.totalindex2 = []
        self.totaltime2 = []
     
        self.shootdict = {}
        self.shootdict2 = {}
        self.velolist = []
        self.velodict = {}
        self.velodict2 = {}
        self.amvelolist = []
        self.amvelodict = {}
        self.amvelodict2 = {}
        self.amshootdict2 = {}
        self.amshootdict = {}
        self.spshootcount2 = {}
        self.spshootcount = {}
        self.amshootcount2 = {}
        self.amshootcount = {}


    def _calc_id(self):
        for key in self.Data:
            for key2 in self.Data[key]:
                dist = self.Data[key][key2]['Distance']
                width = self.Data[key][key2]['Width']
                id = np.log2(2*dist/width)
                mytuple = (self.Data[key][key2]['Completion Time'],self.Data[key][key2]['Completed'])
                if mytuple[1] == True:
                    self.indexlist.append(id)
                    self.timelist.append(mytuple[0])
                    self.totalindex1.append(id)
                    self.totaltime1.append(mytuple[0])
                else:
                    self.indexlist2.append(id)
                    self.timelist2.append(mytuple[0])
                    self.totalindex1.append(id)
                    self.totaltime1.append(mytuple[0])

        for key in self.Data2:
            for key2 in self.Data2[key]:
                dist = self.Data2[key][key2]['Distance']
                width = self.Data2[key][key2]['Width']
                id = np.log2(2*dist/width)
                mytuple = (self.Data2[key][key2]['Completion Time'],self.Data2[key][key2]['Completed'])
                if mytuple[1] == True:
                    self.indexlist3.append(id)
                    self.timelist3.append(mytuple[0])
                    self.totalindex2.append(id)
                    self.totaltime2.append(mytuple[0])
                else:
                    self.indexlist4.append(id)
                    self.timelist4.append(mytuple[0])
                    self.totalindex2.append(id)
                    self.totaltime2.append(mytuple[0])


    def _calc_velocity(self):
        for key in self.Data:
            for key2 in self.Data[key]:
                data = self.Data[key][key2]['data']
                self.velolist = []
                for i in range(len(data)-1):
                    if type(data[i]) is tuple:
                        velo = (data[i+1][0]-data[i][0])/(data[i+1][1]-data[i][1])
                    elif type(data[i]) is not tuple:
                        velo = (data[i+1]-data[i])/0.045
                    self.velolist.append(velo)
                self.velodict2[key2] = copy.deepcopy(self.velolist)
            self.velodict[key] = copy.deepcopy(self.velodict2)

        for key in self.Data2:
            for key2 in self.Data2[key]:
                data = self.Data2[key][key2]['data']
                self.amvelolist = []
                for i in range(len(data)-1):
                    if type(data[i]) is tuple:
                        velo = (data[i+1][0]-data[i][0])/(data[i+1][1]-data[i][1])
                    elif type(data[i]) is not tuple:
                        velo = (data[i+1]-data[i])/0.045
                    self.amvelolist.append(velo)
                self.amvelodict2[key2] = copy.deepcopy(self.amvelolist)
            self.amvelodict[key] = copy.deepcopy(self.amvelodict2)


    def _calc_shoot(self):
        for key in self.velodict:
            for i in self.velodict[key]:
                target_range = self.Data[key][i]['Target Range']
                distance = self.Data[key][i]['Distance']
                velocity = self.velodict[key][i]
                data = self.Data[key][i]['data'] #start work here!
                for j in range(len(velocity)-1):
                    if velocity[j+1] < 0 and velocity[j] >= 0:
                        if type(data[j]) is not tuple:
                            if data[j] > target_range[1]:
                                shoot = (data[j]-target_range[1])/distance
                            elif data[j] < target_range[0]:
                                shoot = (data[j]-target_range[0])/distance
                            else:
                                shoot = 0
                            self.shootdict2[i] = shoot
                            break
                        else:
                            if data[j][0] > target_range[1]:
                                shoot = (data[j][0]-target_range[1])/distance
                            elif data[j][0] < target_range[0]:
                                shoot = (data[j][0]-target_range[0])/distance
                            else:
                                shoot = 0
                            self.shootdict2[i] = shoot
                            break
            self.shootdict[key] = copy.deepcopy(self.shootdict2)

        for key in self.amvelodict:
            for i in self.amvelodict[key]:
                target_range = self.Data2[key][i]['Target Range']
                distance = self.Data2[key][i]['Distance']
                velocity = self.amvelodict[key][i]
                data = self.Data2[key][i]['data'] #start work here!
                for j in range(len(velocity)-1):
                    if velocity[j+1] < 0 and velocity[j] >= 0:
                        if type(data[j]) is not tuple:
                            if data[j] > target_range[1]:
                                shoot = (data[j]-target_range[1])/distance
                            elif data[j] < target_range[0]:
                                shoot = (data[j]-target_range[0])/distance
                            else:
                                shoot = 0
                            self.amshootdict2[i] = shoot
                            break
                        else:
                            if data[j][0] > target_range[1]:
                                shoot = (data[j][0]-target_range[1])/distance
                            elif data[j][0] < target_range[0]:
                                shoot = (data[j][0]-target_range[0])/distance
                            else:
                                shoot = 0
                            self.amshootdict2[i] = shoot
                            break
            self.amshootdict[key] = copy.deepcopy(self.amshootdict2)

    def _calc_totals(self):
        for key in self.velodict:
            for i in self.velodict[key]:
                shootcount = 0
                target_range = self.Data[key][i]['Target Range']
                velocity = self.velodict[key][i]
                data = self.Data[key][i]['data']
                for j in range(len(velocity)-1):
                    if (velocity[j+1] < 0 and velocity[j] >= 0) or (velocity[j+1] >= 0 and velocity[j] < 0):
                        if type(data[j]) is not tuple:
                            if data[j] > target_range[1] or data[j] < target_range[0]:
                                shootcount += 1
                        else:
                            if data[j][0] > target_range[1] or data[j][0] < target_range[0]:
                                shootcount += 1
                self.spshootcount2[i] = shootcount
            self.spshootcount[key] = copy.deepcopy(self.spshootcount2)

        for key in self.amvelodict:
            for i in self.amvelodict[key]:
                shootcount = 0
                target_range = self.Data2[key][i]['Target Range']
                velocity = self.amvelodict[key][i]
                data = self.Data2[key][i]['data']
                for j in range(len(velocity)-1):
                    if (velocity[j+1] < 0 and velocity[j] >= 0) or (velocity[j+1] >= 0 and velocity[j] < 0):
                        if type(data[j]) is not tuple:
                            if data[j] > target_range[1] or data[j] < target_range[0]:
                                shootcount += 1
                        else:
                            if data[j][0] > target_range[1] or data[j][0] < target_range[0]:
                                shootcount += 1
                self.amshootcount2[i] = shootcount
            self.amshootcount[key] = copy.deepcopy(self.amshootcount2)

    def _bar_plot(self):
        fig, ax = plt.subplots()
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(500,100,900,900)
        labels = ['Total Shoot Count', 'First Overshoot/Undershoot','Failures']
        spshoot = 0
        spfail = 0
        amshoot = 0
        amfail = 0
        sptotalcount = 0
        amtotalcount = 0
        for key in self.shootdict:
            for i in self.shootdict[key]:
                if self.shootdict[key][i] != 0:
                    spshoot += 1
                if self.Data[key][i]['Completed'] == False:
                    spfail += 1
        for key in self.amshootdict:
            for i in self.amshootdict[key]:
                if self.amshootdict[key][i] != 0:
                    amshoot += 1
                if self.Data2[key][i]['Completed'] == False:
                    amfail += 1
        for key in self.spshootcount:
            for i in self.spshootcount[key]:
                sptotalcount += self.spshootcount[key][i]
        for key in self.amshootcount:
            for i in self.amshootcount[key]:
                amtotalcount += self.amshootcount[key][i]
        spatial = [sptotalcount,spshoot,spfail]
        amplitude = [amtotalcount,amshoot,amfail]

        x = np.arange(len(labels))
        width = 0.35

        pair1 = ax.bar(x-width/2, spatial, width, label = 'Spatial')
        pair2 = ax.bar(x+width/2, amplitude, width, label = 'Amplitude')

        ax.set_title('SUBJECT %s' % self.subject)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(pair1)
        autolabel(pair2)

        fig.tight_layout()

        plt.savefig('.\Data\Subject_%s\SignalLevelData\comparison%s_FS%s_VS%s.png' % (self.subject,self.subject,FS,VS))

        plt.show()
            
    def _plot(self):
        fig, axs = plt.subplots(2,2)
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(500,100,900,900)
        axs[0,0].set_title('Spatial')
        axs[0,1].set_title('Amplitude')
        axs[1,1].set_axis_off()
        axs[1,0].set_axis_off()
        axs[0,0].scatter(self.indexlist,self.timelist,color='blue')
        axs[0,0].scatter(self.indexlist2,self.timelist2,color='red')
        coef = np.polyfit(self.totalindex1,self.totaltime1,1)
        poly1d_fn = np.poly1d(coef)
        axs[0,0].plot(self.totalindex1,poly1d_fn(self.totalindex1),color='orange')
        axs[0,1].scatter(self.indexlist3,self.timelist3,color='blue')
        axs[0,1].scatter(self.indexlist4,self.timelist4,color='red')
        coef2 = np.polyfit(self.totalindex2,self.totaltime2,1)
        poly1d_fn = np.poly1d(coef2)
        axs[0,1].plot(self.totalindex2,poly1d_fn(self.totalindex2),color='orange')
        fig.suptitle('SUBJECT %s' % self.subject)
        axs[0,0].set(xlabel='Index of Difficulty',ylabel='Completion Time')
        axs[0,1].set(xlabel='Index of Difficulty')
        axs[1,0].text(0.1,0.7,'LoBF Slope: %s' % coef[0])
        axs[1,1].text(0.1,0.7,'LoBF Slope: %s' % coef2[0])
        #plt.savefig('.\Data\Subject_%s\SignalLevelData\idct_FS%s_VS%s.png' % (self.subject,FS,VS))
        plt.show()
    
if __name__ == '__main__':
    spatial = []
    amplitude = []
    spatfail = []
    ampfail = []
    sptotal = []
    amtotal = []
    for i in range(6):
        DA = DataAnalysis(i+1)
        DA._calc_velocity()
        DA._calc_shoot()
        DA._calc_totals()
        spshoot = 0
        spfail = 0
        amshoot = 0
        amfail = 0
        spcount = 0
        amcount = 0
        for key in DA.shootdict:
            for i in DA.shootdict[key]:
                if DA.shootdict[key][i] != 0:
                    spshoot += 1
                if DA.Data[key][i]['Completed'] == False:
                    spfail += 1
        for key in DA.amshootdict:
            for i in DA.amshootdict[key]:
                if DA.amshootdict[key][i] != 0:
                    amshoot += 1
                if DA.Data2[key][i]['Completed'] == False:
                    amfail += 1
        for key in DA.spshootcount:
            for i in DA.spshootcount[key]:
                spcount += DA.spshootcount[key][i]
        for key in DA.amshootcount:
            for i in DA.amshootcount[key]:
                amcount += DA.amshootcount[key][i]
        spatial.append(spshoot)
        amplitude.append(amshoot)
        spatfail.append(spfail)
        ampfail.append(amfail)
        sptotal.append(spcount)
        amtotal.append(amcount)
    spatmean = np.mean(spatial)
    ampmean = round(np.mean(amplitude),1)
    spatfailmean = np.mean(spatfail)
    ampfailmean = np.mean(ampfail)
    sptotmean = round(np.mean(sptotal),1)
    amtotmean = round(np.mean(amtotal),1)
    spatstd = np.std(spatial)
    ampstd = np.std(amplitude)
    spatfailstd = np.std(spatfail)
    ampfailstd = np.std(ampfail)
    sptotstd = np.std(sptotal)
    amtotstd = np.std(amtotal)


    fig, ax = plt.subplots()
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(500,100,900,900)
    labels = ['Total Shoot Count','First Overshoot/Undershoot','Failures']
    x = np.arange(len(labels))
    width = 0.35
    spatial = [sptotmean,spatmean,spatfailmean]
    amplitude = [amtotmean,ampmean,ampfailmean]
    spaterror = [sptotstd,spatstd,spatfailstd]
    amperror = [amtotstd,ampstd,ampfailstd]

    pair1 = ax.bar(x-width/2,spatial,width,yerr=spaterror,capsize=10,label = 'Spatial')
    pair2 = ax.bar(x+width/2,amplitude,width,yerr=amperror,capsize=10,label = 'Amplitude')

    ax.set_title('Aggregate Data')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width()/1.3, height),
                        xytext=(0, 4),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(pair1)
    autolabel(pair2)
    plt.savefig('.\Data\Totalshootfail.png')
    plt.show()

    '''fig, axs = plt.subplots(2,7)
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(500,100,900,900)
    fig.suptitle('DATA')
    spaggindex1 = []
    spaggtime1 = []
    spaggindexfail1 = []
    spaggtimefail1 = []
    sptotaggindex1 = []
    sptotaggtime1 = []
    amaggindex1 = []
    amaggtime1 = []
    amaggindexfail1 = []
    amaggtimefail1 = []
    amtotaggindex1 = []
    amtotaggtime1 = []
    for i in range(6):
        DA = DataAnalysis(i+1)
        DA._calc_velocity()
        DA._calc_id()
        #axs[i,0].set_title('Spatial')
        #axs[i,1].set_title('Amplitude')
        spaggindex1.append(DA.indexlist)
        spaggindexfail1.append(DA.indexlist2)
        spaggtime1.append(DA.timelist)
        spaggtimefail1.append(DA.timelist2)
        sptotaggindex1.append(DA.indexlist)
        sptotaggindex1.append(DA.indexlist2)
        sptotaggtime1.append(DA.timelist)
        sptotaggtime1.append(DA.timelist2)
        amaggindex1.append(DA.indexlist3)
        amaggindexfail1.append(DA.indexlist4)
        amaggtime1.append(DA.timelist3)
        amaggtimefail1.append(DA.timelist4)
        amtotaggindex1.append(DA.indexlist3)
        amtotaggindex1.append(DA.indexlist4)
        amtotaggtime1.append(DA.timelist3)
        amtotaggtime1.append(DA.timelist4)
        axs[0,i].scatter(DA.indexlist,DA.timelist,color='blue')
        axs[0,i].scatter(DA.indexlist2,DA.timelist2,color='red')
        coef = np.polyfit(DA.totalindex1,DA.totaltime1,1)
        poly1d_fn = np.poly1d(coef)
        axs[0,i].plot(DA.totalindex1,poly1d_fn(DA.totalindex1),color='orange')
        axs[1,i].scatter(DA.indexlist3,DA.timelist3,color='blue')
        axs[1,i].scatter(DA.indexlist4,DA.timelist4,color='red')
        coef2 = np.polyfit(DA.totalindex2,DA.totaltime2,1)
        poly1d_fn = np.poly1d(coef2)
        axs[1,i].plot(DA.totalindex2,poly1d_fn(DA.totalindex2),color='orange')
        #axs[i,0].set(xlabel='Index of Difficulty',ylabel='Completion Time')
        #axs[i,1].set(xlabel='Index of Difficulty')

    spaggindex = []
    for sublist in spaggindex1:
        for item in sublist:
            spaggindex.append(item)
    spaggtime = []
    for sublist in spaggtime1:
        for item in sublist:
            spaggtime.append(item)
    spaggindexfail = []
    for sublist in spaggindexfail1:
        for item in sublist:
            spaggindexfail.append(item)
    spaggtimefail = []
    for sublist in spaggtimefail1:
        for item in sublist:
            spaggtimefail.append(item)
    sptotaggtime = []
    for sublist in sptotaggtime1:
        for item in sublist:
            sptotaggtime.append(item)
    sptotaggindex = []
    for sublist in sptotaggindex1:
        for item in sublist:
            sptotaggindex.append(item)
    amaggtime = []
    for sublist in amaggtime1:
        for item in sublist:
            amaggtime.append(item)
    amaggindex = []
    for sublist in amaggindex1:
        for item in sublist:
            amaggindex.append(item)
    amaggtimefail = []
    for sublist in amaggtimefail1:
        for item in sublist:
            amaggtimefail.append(item)
    amaggindexfail = []
    for sublist in amaggindexfail1:
        for item in sublist:
            amaggindexfail.append(item)
    amtotaggtime = []
    for sublist in amtotaggtime1:
        for item in sublist:
            amtotaggtime.append(item)
    amtotaggindex = []
    for sublist in amtotaggindex1:
        for item in sublist:
            amtotaggindex.append(item)

    axs[0,6].scatter(spaggindex,spaggtime,color='blue')
    axs[0,6].scatter(spaggindexfail,spaggtimefail,color='red')
    coef = np.polyfit(sptotaggindex,sptotaggtime,1)
    poly1d_fn = np.poly1d(coef)
    axs[0,6].plot(sptotaggindex,poly1d_fn(sptotaggindex),color='orange')
    axs[1,6].scatter(amaggindex,amaggtime,color='blue')
    axs[1,6].scatter(amaggindexfail,amaggtimefail,color='red')
    coef = np.polyfit(amtotaggindex,amtotaggtime,1)
    poly1d_fn = np.poly1d(coef)
    axs[1,6].plot(amtotaggindex,poly1d_fn(amtotaggindex),color='orange')
    #plt.savefig('.\Data\Total.png')
    plt.show()

    fig2, axs = plt.subplots(2,2)
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(500,100,900,900)
    axs[0,0].set_title('Spatial')
    axs[0,1].set_title('Amplitude')
    axs[1,1].set_axis_off()
    axs[1,0].set_axis_off()
    axs[0,0].scatter(spaggindex,spaggtime,color='blue')
    axs[0,0].scatter(spaggindexfail,spaggtimefail,color='red')
    coef = np.polyfit(sptotaggindex,sptotaggtime,1)
    poly1d_fn = np.poly1d(coef)
    axs[0,0].plot(sptotaggindex,poly1d_fn(sptotaggindex),color='orange')
    axs[0,1].scatter(amaggindex,amaggtime,color='blue')
    axs[0,1].scatter(amaggindexfail,amaggtimefail,color='red')
    coef2 = np.polyfit(amtotaggindex,amtotaggtime,1)
    poly1d_fn = np.poly1d(coef2)
    axs[0,1].plot(amtotaggindex,poly1d_fn(amtotaggindex),color='orange')
    fig.suptitle('ACCURATE')
    axs[0,0].set(xlabel='Index of Difficulty',ylabel='Completion Time')
    axs[0,1].set(xlabel='Index of Difficulty')
    axs[1,0].text(0.1,0.7,'LoBF Slope: %s' % coef[0])
    axs[1,1].text(0.1,0.7,'LoBF Slope: %s' % coef2[0])
    plt.savefig('.\Data\Aggregate.png')
    plt.show()'''
