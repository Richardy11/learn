import numpy as np
import copy
import multiprocessing as mp
import os
import pickle
import sys
import time
from scipy.io import savemat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.transform import Rotation as R
from scipy.optimize import curve_fit
from copy import deepcopy as dp

import matplotlib
matplotlib.use( 'QT5Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import pandas as pd

from DownSampler import DownSampler
from FeatureSmoothing import Smoothing
from FourierTransformFilter import FourierTransformFilter
from MyoArmband import MyoArmband
from RunClassifier import RunClassifier
from SenseController import SenseController
from TimeDomainFilter import TimeDomainFilter
from SpatialClassifier import SpatialClassifier

from TrainingDataGeneration import TrainingDataGenerator
from scipy.io import loadmat

from statsmodels.stats.weightstats import ztest as ztest
from scipy.stats import mannwhitneyu as mwu
from scipy.stats import wilcoxon

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from itertools import combinations
from sklearn.base import clone
from sklearn.metrics import accuracy_score

from multiprocessing import Pool

class FeatureAnalyzer:
    def __init__( self ):
        self.num_classes = 5
        self.num_channels = 8

        self.sampling_rate = 500
        self.num_FFT = 31
        self.num_unique_features = 36
        self.num_features = 40+self.num_FFT*self.num_channels
        self.bin_length = self.sampling_rate/(self.num_FFT+1)

        self.zscore_cutoff = 0.05

        self.palette = ['darkslateblue',
                        'firebrick',
                        'forestgreen',
                        'goldenrod',
                        'darkorchid',
                        'lightseagreen',
                        'plum',
                        'lightcoral',
                        'peru',
                        'lightpink',
                        'tan' ]
    
    def generateDataFromRaw():
        pass
    
    def generateDataFromFE( self, SUBJECT ):
        PARAM_FILE = os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ), 
                    'data', 'Subject_%s' % SUBJECT, 'RESCU_Protocol_Subject_%s.pkl' % SUBJECT )

        with open( PARAM_FILE, 'rb' ) as pkl:
            Parameters = pickle.load( pkl )

        EMG_SCALING_FACTOR = 10000.0

        tdg = TrainingDataGenerator()
        Xtrain, ytrain = tdg.emg_pre_extracted( CLASSES = Parameters['Calibration']['CUE_LIST'],
                                                PREVIOUS_FILE = '',
                                                practice_session = 0,
                                                CARRY_UPDATES = Parameters['Misc.']['CARRY_UPDATES'],
                                                total_training_samples = Parameters['Classification']['TOTAL_TRAINING_SAMPLES'],
                                                onset_threshold_enable = Parameters['Classification']['ONSET_THRESHOLD'],
                                                onset_scaler = Parameters['Calibration']['CALIB_ONSET_SCALAR'],
                                                gain = Parameters['General']['ELECTRODE_GAIN'],
                                                fft_length = Parameters['General']['FFT_LENGTH'],
                                                EMG_SCALING_FACTOR = EMG_SCALING_FACTOR,
                                                emg_window_size = Parameters['General']['WINDOW_SIZE'],
                                                emg_window_step = Parameters['General']['WINDOW_STEP'],
                                                CALIBRATION = Parameters['Calibration']['CALIB_METHOD'],
                                                FL = False,
                                                subject = SUBJECT)
        
        return Xtrain, ytrain

    def plotSingle( self, SUBJECT ):
        Xtrain, ytrain = self.generateDataFromFE( SUBJECT )
        fig = plt.figure(figsize = ( 6, 6 ), tight_layout = 3)
        matplotlib.rcParams['font.size'] = 7
        mngr = plt.get_current_fig_manager()
        geom = mngr.window.geometry()
        x, y, dx, dy = geom.getRect()
        mngr.window.setGeometry( 510, 470, 550, 550 )

        plt.plot( np.linspace( 0, 1, Xtrain.shape[0] ), np.mean( Xtrain, axis = 1 ) )
        plt.show()

    def calculateZtestClasswise( self, SUBJECTS ):

        pvals = np.zeros((len(SUBJECTS), self.num_classes, self.num_classes, self.num_features))
        pval_means = np.zeros(( len(SUBJECTS), self.num_classes-1, self.num_features))
        self.pval_means_of_means = np.zeros(( len(SUBJECTS), self.num_features))

        for key, sub in enumerate(SUBJECTS):
        
            Xtrain, ytrain = self.generateDataFromFE( sub )

            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    for k in range(self.num_features):
                        _, p = ztest(Xtrain[ytrain==i,k], Xtrain[ytrain==j,k], value = 0)
                        pvals[key, i, j, k] = 1 if np.isnan(p) else p

                try:
                    pval_means[key, i, :] = np.mean(pvals[key, i, i+1:self.num_classes, :], axis = 0)
                except: pass

            self.pval_means_of_means[key, :] = np.mean(pval_means[key, :, :], axis = 0)

    def calculateZtestFeaturewise( self, SUBJECTS ):

        self.pvals_FW = np.zeros((len(SUBJECTS), self.num_unique_features, self.num_unique_features, self.num_channels))
        pval_means_FW = np.zeros(( len(SUBJECTS), self.num_classes-1, self.num_features))
        #self.pval_means_of_means = np.zeros(( len(SUBJECTS), self.num_features))

        for key, sub in enumerate(SUBJECTS):
        
            Xtrain, ytrain = self.generateDataFromFE( sub )

            for i in range(self.num_channels):
                for j in range(self.num_unique_features):
                    for k in range(self.num_unique_features):
                        if k > j:
                            #_, p = ztest(Xtrain[:,i + (j*self.num_channels)], Xtrain[:,i + (k*self.num_channels)], value = 0)
                            #_, p = mwu(Xtrain[:,i + (j*self.num_channels)], Xtrain[:,i + (k*self.num_channels)])
                            _, p = wilcoxon(Xtrain[:,i + (j*self.num_channels)], Xtrain[:,i + (k*self.num_channels)])
                            self.pvals_FW[key, j, k, i] = 1 if np.isnan(p) else p
                            #self.pvals_FW[key, k, j, i] = 1 if np.isnan(p) else p

    def _optimalFeatures(self, SUBJECT):

        Classifier = SpatialClassifier()

        PARAM_FILE = os.path.join( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ), 
                'data', 'Subject_%s' % SUBJECT, 'RESCU_Protocol_Subject_%s.pkl' % SUBJECT  )

        with open( PARAM_FILE, 'rb' ) as pkl:
            Parameters = pickle.load( pkl )# compute training features and labels
        
        featuresToGo = list(range( 5, self.num_FFT+5, 1 ))

        Xtrain, ytrain = self.generateDataFromFE( SUBJECT )

        currentFeatureSet = []

        featureRank = np.zeros((self.num_FFT,self.num_FFT))

        featuresToGoIterator = 0

        selectedIndex = []

        while len(featuresToGo) > 0:
    
            for feat in featuresToGo:
                tempCurrentFeatureSet = dp(currentFeatureSet)
                tempCurrentFeatureSet.extend( list( range( feat*self.num_channels, feat*self.num_channels+self.num_channels, 1 ) ) )
                reducedXtrain = Xtrain[:, tempCurrentFeatureSet]

                conf = Classifier.SeparabilityScore(    Xtrain = reducedXtrain, 
                                                        ytrain = ytrain, 
                                                        train_percentage = 5, 
                                                        classes = Parameters['Calibration']['CUE_LIST'],
                                                        perc_lo = [0 for i in range(len(Parameters['Calibration']['CUE_LIST']))] , 
                                                        perc_hi = [1 for i in range(len(Parameters['Calibration']['CUE_LIST']))], 
                                                        threshold_ratio = Parameters['Classification']['THRESHOLD_RATIO'] )

                featureRank[feat-5, featuresToGoIterator] = np.mean(np.diag(conf))
            
            m = max(featureRank[:, featuresToGoIterator])
            mList = [i for i, j in enumerate(featureRank[:, featuresToGoIterator]) if j == m]

            if featuresToGoIterator == 0:
                firstRound = featureRank[:, featuresToGoIterator]

            selectedIndex.append(min(mList)+5)

            featuresToGoIterator += 1
            featuresToGo.remove(selectedIndex[-1])
            currentFeatureSet.extend( list( range( selectedIndex[-1]*self.num_channels, selectedIndex[-1]*self.num_channels+self.num_channels, 1 ) ) )
            
            print( 'SUBJECT: ', SUBJECT, 'Iteration: ', featuresToGoIterator )
        
        return [selectedIndex, firstRound]

    def plotFirstFeatures(self, SUBJECTS):
        try:
            with open( f"Data\FeatureSFS.pkl", 'rb' ) as f:
                results = pickle.load(f)
            scores = np.zeros((len(SUBJECTS),self.num_FFT))

            for i in range( len(SUBJECTS) ):
                idxs = np.argsort(results[i][1][:])
                scores[i,:] = abs(idxs-32)

            x_data = np.linspace( 0, self.num_FFT-1, self.num_FFT )

            fig, zax = plt.subplots(figsize=(16,10), dpi= 80, tight_layout = 6)

            xticklabels = []
            for feats in range(self.num_FFT+5):

                if feats == 0:
                    xticklabels.append('MAV')
                elif feats == 1:
                    xticklabels.append('VAR')
                elif feats == 2:
                    xticklabels.append('WFL')
                elif feats == 3:
                    xticklabels.append('ZC')
                elif feats == 4:
                    xticklabels.append('SSC')
                else:
                    xticklabels.append(str((feats-5)*self.bin_length) + '-' + str((feats-5)*self.bin_length+(self.bin_length)) + ' Hz')

            zax.axhline(y=16, color='r', linestyle=':', alpha = 0.6)

            for key in range(len(SUBJECTS)+1):
                
                if key == len(SUBJECTS):
                    dataToPlot = np.mean(scores, axis=0)

                    offset = 0.25 - ( key * 0.5/len(SUBJECTS) )
                    zax.vlines(x_data+offset, ymin=0, ymax=dataToPlot, color='k', alpha=0.75, linewidth=2)
                    zax.scatter(x_data+offset, dataToPlot, s=25, color='k', alpha=1, label = 'Mean values')       
                else:
                    dataToPlot = scores[key, :]

                    offset = 0.25 - ( key * 0.5/len(SUBJECTS) )
                    zax.vlines(x_data+offset, ymin=0, ymax=dataToPlot, color=self.palette[key], alpha=0.25, linewidth=2)
                    zax.scatter(x_data+offset, dataToPlot, s=25, color=self.palette[key], alpha=0.7, label = 'Subject '+ str(key+1))               
                
            zax.set_title( 'Scores of first iteration of SFS analysis', fontweight = 'bold')# y=1.0, pad=-14, fontdict = {'fontsize': 12}  )

            zax.set_ylim([0, 33])
            zax.set_xticks(x_data)
            zax.set_xticklabels(xticklabels[5:], rotation=60, fontdict={'horizontalalignment': 'right', 'size':10})

            zax.set(xlabel='Features', ylabel='Score')

            zax.legend()

            plt.pause(0.01)

        except:
            pass

    def plotFeaturesScore(self, SUBJECTS):
        try:
            with open( f"Data\FeatureSFS.pkl", 'rb' ) as f:
                results = pickle.load(f)
            scores = np.zeros((len(SUBJECTS),self.num_FFT))

            for i in range( len(SUBJECTS) ):
                weight = 0
                for j in range(self.num_FFT):
                    
                    scores[i,results[i][0][j]-5] = self.num_FFT - weight
                    weight += 1 

            x_data = np.linspace( 0, self.num_FFT-1, self.num_FFT )

            fig, zax = plt.subplots(figsize=(16,10), dpi= 80, tight_layout = 6)

            xticklabels = []
            for feats in range(self.num_FFT+5):

                if feats == 0:
                    xticklabels.append('MAV')
                elif feats == 1:
                    xticklabels.append('VAR')
                elif feats == 2:
                    xticklabels.append('WFL')
                elif feats == 3:
                    xticklabels.append('ZC')
                elif feats == 4:
                    xticklabels.append('SSC')
                else:
                    xticklabels.append(str((feats-5)*self.bin_length) + '-' + str((feats-5)*self.bin_length+(self.bin_length)) + ' Hz')

            zax.axhline(y=16, color='r', linestyle=':', alpha = 0.6)
            zax.vlines(x_data, ymin=0, ymax=np.repeat(self.num_FFT, self.num_FFT), color='k', alpha=0.25, linewidth=2, zorder=0)
            zax.hlines(x_data, xmin=0, xmax=np.repeat(self.num_FFT, self.num_FFT), color='k', alpha=0.25, linewidth=2, zorder=0)

            for key in range(len(SUBJECTS)):

                dataToPlot = []
                
                sortedScores = np.argsort(scores[key, :])

                for i in reversed(sortedScores):
                    dataToPlot.append(i)
                
                zax.scatter(x_data, dataToPlot, s=75, color=self.palette[key], alpha=1, label = 'Subject '+ str(key+1))
                #zax.plot(x_data, dataToPlot, color=self.palette[key], alpha=0.75)               
                
            zax.set_title( 'Scores of first iteration of SFS analysis', fontweight = 'bold')# y=1.0, pad=-14, fontdict = {'fontsize': 12}  )

            zax.set_ylim([-0.5, 31])
            zax.set_xticks(x_data)
            zax.set_yticks(x_data)
            zax.set_yticklabels(xticklabels[5:], rotation=60, fontdict={'horizontalalignment': 'right', 'size':10})

            zax.legend()

            zax.set(xlabel='Features', ylabel='Score')

            plt.pause(0.01)

        except:
            pass

    def plotOptimalFeatures(self, SUBJECTS):

        try:
            with open( f"Data\FeatureSFS.pkl", 'rb' ) as f:
                results = pickle.load(f)
        except:
            with Pool( processes=len(SUBJECTS) ) as pool:
                results = pool.map(self._optimalFeatures, SUBJECTS)

            with open( f"Data\FeatureSFS.pkl", 'wb' ) as f:
                pickle.dump(results, f)

        scores = np.zeros((len(SUBJECTS),self.num_FFT))

        for i in range( len(SUBJECTS) ):
            weight = 0
            for j in range(self.num_FFT):
                
                scores[i,results[i][0][j]-5] = weight
                weight += 1 

        try:
            x_data = np.linspace( 0, self.num_FFT-1, self.num_FFT )

            fig, zax = plt.subplots(figsize=(16,10), dpi= 80, tight_layout = 6)

            xticklabels = []
            for feats in range(self.num_FFT+5):

                if feats == 0:
                    xticklabels.append('MAV')
                elif feats == 1:
                    xticklabels.append('VAR')
                elif feats == 2:
                    xticklabels.append('WFL')
                elif feats == 3:
                    xticklabels.append('ZC')
                elif feats == 4:
                    xticklabels.append('SSC')
                else:
                    xticklabels.append(str((feats-5)*self.bin_length) + '-' + str((feats-5)*self.bin_length+(self.bin_length)) + ' Hz')
            zax.axhline(y=16, color='r', linestyle=':', alpha = 0.6)
            for key in range(len(SUBJECTS)+1):
                
                if key == len(SUBJECTS):
                    dataToPlot = np.mean(scores, axis=0)

                    offset = 0.25 - ( key * 0.5/len(SUBJECTS) )
                    zax.vlines(x_data+offset, ymin=0, ymax=dataToPlot, color='k', alpha=0.75, linewidth=2)
                    zax.scatter(x_data+offset, dataToPlot, s=25, color='k', alpha=1, label = 'Mean')
                    m, b = np.polyfit(x_data, dataToPlot, 1)
                    zax.plot(x_data+offset, m*(x_data+offset) + b, color='k', alpha = 0.8)       
                else:
                    dataToPlot = scores[key, :]

                    offset = 0.25 - ( key * 0.5/len(SUBJECTS) )
                    zax.vlines(x_data+offset, ymin=0, ymax=dataToPlot, color=self.palette[key], alpha=0.25, linewidth=2)
                    zax.scatter(x_data+offset, dataToPlot, s=25, color=self.palette[key], alpha=0.7, label = 'Subject '+ str(key+1))  

                    m, b = np.polyfit(x_data, dataToPlot, 1)
                    zax.plot(x_data+offset, m*(x_data+offset) + b, color=self.palette[key], alpha = 0.8)             
                
            zax.set_title( 'Mean scores of SFS analysis', fontweight = 'bold')# y=1.0, pad=-14, fontdict = {'fontsize': 12}  )

            zax.set_ylim([0, 33])
            zax.set_xticks(x_data)
            zax.set_xticklabels(xticklabels[5:], rotation=60, fontdict={'horizontalalignment': 'right', 'size':10})

            zax.set(xlabel='Features', ylabel='Score')

            zax.legend()

            plt.pause(0.01)
        except Exception as e:
            print(e)

    def plotSingleGraph(self, SUBJECTS):

        def func(x, a, b, c):
            return a * np.exp(-b * x) + c

        try:
            mean_p_vals = np.zeros((len(SUBJECTS), self.num_FFT+5))

            x_data = np.linspace( 0, self.num_FFT+4, self.num_FFT+5 )

            fig, zax = plt.subplots(figsize=(16,10), dpi= 80, tight_layout = 6)

            xticklabels = []
            for feats in range(self.num_FFT+5):

                if feats == 0:
                    xticklabels.append('MAV')
                elif feats == 1:
                    xticklabels.append('VAR')
                elif feats == 2:
                    xticklabels.append('WFL')
                elif feats == 3:
                    xticklabels.append('ZC')
                elif feats == 4:
                    xticklabels.append('SSC')
                else:
                    xticklabels.append(str((feats-5)*self.bin_length) + '-' + str((feats-5)*self.bin_length+(self.bin_length)) + ' Hz')

            for key, sub in enumerate(SUBJECTS):
                for feats in range(self.num_FFT+5):
                    start_idx = feats*self.num_channels
                    end_idx = 8+feats*self.num_channels
                    
                    mean_p_vals[key, feats] = np.mean(self.pval_means_of_means[key, start_idx:end_idx])

                offset = 0.25 - ( key * 0.5/len(SUBJECTS) )
                zax.axhline(y=self.zscore_cutoff, color='r', linestyle=':', alpha = 0.6)
                zax.vlines(x_data+offset, ymin=0, ymax=mean_p_vals[key], color=self.palette[key], alpha=0.25, linewidth=2)
                zax.scatter(x_data+offset, mean_p_vals[key], s=25, color=self.palette[key], alpha=0.7, label = 'Subject '+ str(key+1))

                m, b = np.polyfit(x_data, mean_p_vals[key], 1)

                zax.plot(x_data+offset, m*(x_data+offset) + b, color=self.palette[key], alpha = 0.8)
                
                
            zax.set_title( 'Mean p values of features accross all channels', fontweight = 'bold')# y=1.0, pad=-14, fontdict = {'fontsize': 12}  )

            zax.set_ylim([0, 0.5])
            zax.set_xticks(x_data)
            zax.set_xticklabels(xticklabels, rotation=60, fontdict={'horizontalalignment': 'right', 'size':10})

            zax.set(xlabel='Features', ylabel='p-values')

            zax.legend()

            plt.pause(0.01)
        except Exception as e:
            print(e)
            print("p values not available")

    def plotChannelGraph(self, SUBJECTS):

        #try:
        #fig = plt.figure(figsize = ( 6, 6 ), tight_layout = 3)
        
        rows = int(np.ceil(np.sqrt(self.num_unique_features)))
        columns = int(np.ceil(np.sqrt(self.num_unique_features)))

        fig, axs = plt.subplots( figsize=(16,10), dpi= 80, nrows= rows, ncols= columns, tight_layout = 6)
        
        fig.suptitle('Channel-wise p-values for all features', fontweight='bold')

        x_data = np.linspace( 0, self.num_channels-1, self.num_channels )

        for key, sub in enumerate(SUBJECTS):
            for feats in range(self.num_unique_features):
                start_idx = feats*self.num_channels
                end_idx = 8+feats*self.num_channels
                
                y_data = self.pval_means_of_means[key, start_idx:end_idx]
                
                offset = 0.5 - ( key * 0.5/len(SUBJECTS) )
                axs[int( np.floor(feats/columns) ), int(feats%columns)].axhline(y=self.zscore_cutoff, color='r', linestyle=':', alpha = 0.6)
                axs[int( np.floor(feats/columns) ), int(feats%columns)].vlines(x_data+offset, ymin=0, ymax=y_data, color=self.palette[key], alpha=0.7, linewidth=2)
                axs[int( np.floor(feats/columns) ), int(feats%columns)].scatter(x_data+offset, y_data, s=10, color=self.palette[key], alpha=0.8, label = 'Subject '+ str(key+1))

                if feats == 0:
                    tit_str = 'MAV'
                elif feats == 1:
                    tit_str = 'VAR'
                elif feats == 2:
                    tit_str = 'WFL'
                elif feats == 3:
                    tit_str = 'ZC'
                elif feats == 4:
                    tit_str = 'SSC'
                else:
                    tit_str = 'FFT Band: ' + str((feats-5)*self.bin_length) + ' Hz - ' + str((feats-5)*self.bin_length+(self.bin_length)) + ' Hz'
                axs[int( np.floor(feats/columns) ), int(feats%columns)].set_title( tit_str, y=1.0, pad=-14, fontdict = {'fontsize': 8}  )

                axs[int( np.floor(feats/columns) ), int(feats%columns)].set_ylim([0,1.1])

                axs[int( np.floor(feats/columns) ), int(feats%columns)].set_xticks(x_data)

        for ax in axs.flat:
            ax.set(xlabel='Channels', ylabel='p-values')
            
        plt.pause(0.01)

        #except:
        #    print("p values not available")

    def plotFeatureGraph(self, SUBJECTS):

        def newline(p1, p2, ax, color='darkslateblue', alpha = 0.1, label = ''):
            l = mlines.Line2D([p1[0],p2[0]], [p1[1],p2[1]], color=color, marker='o', markersize=4, alpha = alpha, label = label)
            ax.add_line(l)
            return l

        fig, fax = plt.subplots(figsize=(16,10), dpi= 80, tight_layout = 6)

        p_val_feats_dict = {}

        for key, sub in enumerate(SUBJECTS):
            p_val_feats_dict[key] = {}
            for feats in range(self.num_unique_features):
                p_val_feats_dict[key][feats] = {}
                for inner_feats in range(self.num_unique_features):

                    '''bool_arr = self.pvals_FW[key, feats, inner_feats, : ] > self.zscore_cutoff
                    p_val_feats_dict[key][feats][inner_feats] = True if list(bool_arr).count(True) > 6 else False'''

                    bool_arr = np.mean(self.pvals_FW[key, feats, inner_feats, : ]) > self.zscore_cutoff
                    p_val_feats_dict[key][feats][inner_feats] = np.all(bool_arr)

            tempfax = fax

            tempfax.vlines(x=1, ymin=0, ymax=self.num_unique_features, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
            tempfax.vlines(x=3, ymin=0, ymax=self.num_unique_features, color='black', alpha=0.7, linewidth=1, linestyles='dotted')

            # Points
            ydata = list(range(0, self.num_unique_features, 1))
            tempfax.scatter(y= ydata, x=np.repeat(1, self.num_unique_features), s=10, color=self.palette[key], alpha=0.7)
            tempfax.scatter(y= ydata, x=np.repeat(3, self.num_unique_features), s=10, color=self.palette[key], alpha=0.7)

            for feats in range(self.num_unique_features):
                if feats == 0:
                    tit_str = 'MAV'
                elif feats == 1:
                    tit_str = 'VAR'
                elif feats == 2:
                    tit_str = 'WFL'
                elif feats == 3:
                    tit_str = 'ZC'
                elif feats == 4:
                    tit_str = 'SSC'
                else:
                    tit_str = 'FFT Band: ' + str((feats-5)*self.bin_length) + ' Hz - ' + str((feats-5)*self.bin_length+(self.bin_length)) + ' Hz'

                tempfax.text(1-0.05, feats, tit_str, horizontalalignment='right', verticalalignment='center', fontdict={'size':10})
                tempfax.text(3+0.05, feats, tit_str, horizontalalignment='left', verticalalignment='center', fontdict={'size':10})
                
                p1 = feats
                for inner_feats in range(self.num_unique_features):
                    if p_val_feats_dict[key][feats][inner_feats]:
                        newline([1,p1], [3,inner_feats], ax = tempfax, color=self.palette[key], label = 'Subject '+ str(key+1) )

                if key == len(SUBJECTS)-1:
                    for inner_feats in range(self.num_unique_features):
                        for key2, sub in enumerate(SUBJECTS):
                            if not p_val_feats_dict[key2][feats][inner_feats]:
                                break
                            elif key2 == key:
                                newline([1,p1], [3,inner_feats], ax = tempfax, color='black', alpha = 0.8, label = 'Overlapping redundancy')
                        

            # Decoration
            tempfax.set_title("Feature overlaps where means of p-values across channels exceed " + str(self.zscore_cutoff), fontweight = 'bold' )
            tempfax.set(xlim=(0,4), ylim=(-1,self.num_unique_features+1), ylabel='Features')
            tempfax.set_xticks([1,3])
            tempfax.set_xticklabels(["", ""])
            tempfax.set_yticks([0])
            tempfax.set_yticklabels([""])
            # Lighten borders
            tempfax.spines["top"].set_alpha(.0)
            tempfax.spines["bottom"].set_alpha(.0)
            tempfax.spines["right"].set_alpha(.0)
            tempfax.spines["left"].set_alpha(.0)

        plt.pause(0.01)

if __name__ == '__main__':

    SUBJECTS = [2, 3, 4, 5, 6, 7]

    FA = FeatureAnalyzer()

    FA.calculateZtestClasswise( SUBJECTS = SUBJECTS )

    FA.calculateZtestFeaturewise( SUBJECTS = SUBJECTS )

    FA.plotSingleGraph( SUBJECTS = SUBJECTS )

    FA.plotChannelGraph( SUBJECTS = SUBJECTS )

    FA.plotFeatureGraph( SUBJECTS = SUBJECTS )

    FA.plotOptimalFeatures( SUBJECTS = SUBJECTS )

    FA.plotFirstFeatures( SUBJECTS = SUBJECTS )

    FA.plotFeaturesScore( SUBJECTS = SUBJECTS )

    plt.show()

    #FA.plotSingle( SUBJECT = SUBJECT )

    #all(j > self.zscore_cutoff for j in pval_means_of_means[:, i])

# def plotFeatureGraph(self, SUBJECTS):

#         '''rows = int(np.ceil(np.sqrt(self.num_features)))
#         columns = int(np.ceil(np.sqrt(self.num_features)))

#         fig, fax = plt.subplots(figsize=(16,10), dpi= 80, nrows= rows, ncols= columns)
        
#         x_data = np.linspace( 0, self.num_features-1, self.num_features )'''

#         def newline(p1, p2, ax, color='black'):
#             l = mlines.Line2D([p1[0],p2[0]], [p1[1],p2[1]], color='darkslateblue', marker='o', markersize=4, alpha = 0.05)
#             ax.add_line(l)
#             return l

#         rows = len(SUBJECTS)
#         columns = 8

#         fig, fax = plt.subplots(figsize=(16,10), dpi= 80, nrows= rows, ncols= columns, tight_layout = 3)

#         #fig.set_title("Significant redundacy calculated features-wise", fontdict={'size':22})

#         p_val_feats_dict = {}

#         for key, sub in enumerate(SUBJECTS):
#             p_val_feats_dict[key] = {}
#             for feats in range(self.num_features):
#                 #fax.vlines(x_data, ymin=0, ymax=self.pvals_FW[key, feats,:], color='darkslateblue', alpha=0.7, linewidth=2)
#                 '''fax[int( np.floor(feats/columns) ), int(feats%columns)].scatter(x_data, self.pvals_FW[key, feats,:], s=25, color='darkslateblue', alpha=0.95)
#                 fax[int( np.floor(feats/columns) ), int(feats%columns)].axhline(y=self.zscore_cutoff, color='r', linestyle='-')
#                 fax[int( np.floor(feats/columns) ), int(feats%columns)].set_ylim([0,1.1])'''

#                 bool_arr = self.pvals_FW[key, feats, :] > self.zscore_cutoff
#                 p_val_feats_dict[key][feats] = np.where(bool_arr)[0]
            
#             for figs in range(columns):

#                 #idxs = list(range(figs, self.num_features, columns))
#                 vals_per_ax = int(self.num_features/columns)
#                 idxs = list( range( figs * vals_per_ax, figs * vals_per_ax + vals_per_ax - 1, 1) )

#                 try:
#                     tempfax = fax[key, figs]
#                 except:
#                     tempfax = fax[figs]

#                 tempfax.vlines(x=1, ymin=0, ymax=self.num_features+5, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
#                 tempfax.vlines(x=3, ymin=0, ymax=self.num_features+5, color='black', alpha=0.7, linewidth=1, linestyles='dotted')

#                 # Points
#                 for idx in idxs:
#                     tempfax.scatter(y= idx, x=np.repeat(1, 1), s=10, color='darkslateblue', alpha=0.7)
#                     xdata = np.repeat(3, len( p_val_feats_dict[key][idx] ) )
#                     tempfax.scatter(y= p_val_feats_dict[key][idx], x=xdata, s=10, color='darkslateblue', alpha=0.7)

#                     p1 = idx

#                     for p2 in p_val_feats_dict[key][idx]:
#                         newline([1,p1], [3,p2], tempfax)
#                         '''tempfax.text(1-0.05, p1, c + ', ' + str(round(p1)), horizontalalignment='right', verticalalignment='center', fontdict={'size':14})
#                         tempfax.text(3+0.05, p2, c + ', ' + str(round(p2)), horizontalalignment='left', verticalalignment='center', fontdict={'size':14})'''

#                 # Decoration
                
#                 tempfax.set(xlim=(0,4), ylim=(0,self.num_features), ylabel='Features')
#                 tempfax.set_xticks([1,3])
#                 tempfax.set_xticklabels(["", ""])
#                 tempfax.set_yticks([0])
#                 tempfax.set_yticklabels([""])
#                 # Lighten borders
#                 tempfax.spines["top"].set_alpha(.0)
#                 tempfax.spines["bottom"].set_alpha(.0)
#                 tempfax.spines["right"].set_alpha(.0)
#                 tempfax.spines["left"].set_alpha(.0)
#                 '''if feats == 0:
#                     tit_str = 'MAV'
#                 elif feats == 1:
#                     tit_str = 'VAR'
#                 elif feats == 2:
#                     tit_str = 'WFL'
#                 elif feats == 3:
#                     tit_str = 'ZC'
#                 elif feats == 4:
#                     tit_str = 'SSC'
#                 else:
#                     tit_str = 'FFT Band: ' + str((feats-5)*self.bin_length) + ' Hz - ' + str((feats-5)*self.bin_length+(self.bin_length)) + ' Hz'
#                 fax.set_title( tit_str, y=1.0, pad=-14, fontdict = {'fontsize': 8}  )

#                 fax.set_ylim([0,1.1])'''

#         '''for ax in axs.flat:
#             ax.set(xlabel='Channels', ylabel='p-values')'''

#         '''left_label = [str(c) + ', '+ str(round(y)) for c, y in zip(df.continent, df['1952'])]
#         right_label = [str(c) + ', '+ str(round(y)) for c, y in zip(df.continent, df['1957'])]
#         klass = ['red' if (y1-y2) < 0 else 'green' for y1, y2 in zip(df['1952'], df['1957'])]'''

#         '''# draw line
#         # https://stackoverflow.com/questions/36470343/how-to-draw-a-line-with-matplotlib/36479941
#         def newline(p1, p2, color='black'):
#             ax = plt.gca()
#             l = mlines.Line2D([p1[0],p2[0]], [p1[1],p2[1]], color='red' if p1[1]-p2[1] > 0 else 'green', marker='o', markersize=6)
#             ax.add_line(l)
#             return l

#         fig, ax = plt.subplots(1,1,figsize=(14,14), dpi= 80)

#         # Vertical Lines
#         ax.vlines(x=1, ymin=0, ymax=self.num_features+5, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
#         ax.vlines(x=3, ymin=0, ymax=self.num_features+5, color='black', alpha=0.7, linewidth=1, linestyles='dotted')

#         # Points
#         ax.scatter(y=df['1952'], x=np.repeat(1, self.num_features/4), s=10, color='black', alpha=0.7)
#         ax.scatter(y=df['1957'], x=np.repeat(3, self.num_features/4), s=10, color='black', alpha=0.7)

#         # Line Segmentsand Annotation
#         for p1, p2, c in zip(df['1952'], df['1957'], df['continent']):
#             newline([1,p1], [3,p2])
#             ax.text(1-0.05, p1, c + ', ' + str(round(p1)), horizontalalignment='right', verticalalignment='center', fontdict={'size':14})
#             ax.text(3+0.05, p2, c + ', ' + str(round(p2)), horizontalalignment='left', verticalalignment='center', fontdict={'size':14})

#         # 'Before' and 'After' Annotations
#         ax.text(1-0.05, 13000, 'BEFORE', horizontalalignment='right', verticalalignment='center', fontdict={'size':18, 'weight':700})
#         ax.text(3+0.05, 13000, 'AFTER', horizontalalignment='left', verticalalignment='center', fontdict={'size':18, 'weight':700})

#         # Decoration
#         ax.set_title("Slopechart: Comparing GDP Per Capita between 1952 vs 1957", fontdict={'size':22})
#         ax.set(xlim=(0,4), ylim=(0,14000), ylabel='Mean GDP Per Capita')
#         ax.set_xticks([1,3])
#         ax.set_xticklabels(["1952", "1957"])
#         plt.yticks(np.arange(500, 13000, 2000), fontsize=12)'''
#         plt.show()
            
#         plt.show()