import numpy as np
import queue
import copy
import scipy.io as sio
import math
from collections import deque
import multiprocessing as mp
import matplotlib
matplotlib.use( "QT5Agg" )
import matplotlib.pyplot as plt

from DownSampler import DownSampler
from SegmentAnalysisFilter import SegmentAnalysisFilter

class Segmentation:
    def __init__(self, onset_threshold,  SENS_LIST = [], overwrite=5, onset_data_caching = False, counter = 1, plot = True, plot_max = 0.05):

        self.segmentation_delay     = 30
        self.onset_threshold        = onset_threshold

        self.MAV_buffer             = deque()
        self.data_buffer            = deque()
        self.onset_data_caching      = onset_data_caching
        self.activity               = { 'markers': [0],
                                        'index': [0] }
        self.overwrite              = overwrite
        self.new_activity           = False
        self.new_onset              = False
        self.counter                = counter
        self.ISC_minimum            = -np.inf
        if counter == 1:
            self.ISC_coeff          = []
        else:
            self.ISC_coeff          = [0] * (counter-1)
        self.CR                     = []
        self.filtered_out           = []
        # data caching
        self.cache_downsampling_step= 1
        self.cache_track            = 1
        self.sparse_cache           = []
        self.cache                  = []
        self.rest_cache             = []
        self.previous_downsampling_step = 1

        self.cache_list             = []
        self.CR_list                = []
        self.rest_cache_list        = []
        self.rest_CR_list           = []
        self.cache_dictionary       = []

        self.cache_iter             = 0
        self.onset_occured          = False

        self.downsample = DownSampler()

        self.plot = plot
        self.plot_max = plot_max

        self.add_cache_onset = False

        if self.plot:
            self.run_plot()

    def run_plot(self):
        try:
            self._queue = mp.Queue()
            self._exit_event = mp.Event()

            self._plotter = mp.Process( target = self._plot )
            self._plotter.start()
        except:
            pass

    def _plot(self):
        buffer_length = int( 500 * 5 )

        fig = plt.figure( figsize = ( 9.0 , 3.0 ))
        mngr = plt.get_current_fig_manager()
        geom = mngr.window.geometry()
        x, y, dx, dy = geom.getRect()
        mngr.window.setGeometry( 1065, 905, dx, dy )

        ax = fig.add_subplot(111)
        ax.set_xlim([0,buffer_length])
        ax.set_ylim([0,self.plot_max])

        avg = ax.plot( np.linspace( 0, buffer_length, buffer_length ), np.zeros( shape = ( buffer_length, ) ), zorder = -1 )
        ISC = ax.plot( np.linspace( 0, buffer_length, buffer_length ), np.zeros( shape = ( buffer_length, ) ), zorder = -1 )
        plt.axhline(y=self.onset_threshold, color = 'y')
        plt.axhline(y=self.onset_threshold*2, color = 'c')
        segments = []
        cache_markers = []

        fig.tight_layout()
        while not self._exit_event.is_set():
            signal = []
            coeff = []
            activity = []

            # pull samples from queue
            while self._queue.qsize() > 0:
                signal = []
                coeff = []
                activity = []
                '''if self._queue.qsize() > 1:
                    print(self._queue.qsize())
                    queue_data = self._queue.get() 
                    print(self._queue.qsize())'''               
                queue_data = self._queue.get()

                signal = queue_data[0]
                
                coeff = queue_data[1]

                if len(queue_data[2]) > 0:
                    activity = queue_data[2]
                    cache = queue_data[3]

                cache_marker = queue_data[4]
                
                # if we go valid samples
                if signal:
                    
                    # concatenate data
                    ydata = avg[ 0 ].get_ydata()
                    ydata = np.append( ydata, signal )
                    ydata = ydata[-buffer_length:]
                    avg[0].set_ydata( ydata )

                    # concatenate data
                    isc_ydata = ISC[ 0 ].get_ydata()
                    isc_ydata = np.append( isc_ydata, coeff )
                    isc_ydata = isc_ydata[-buffer_length:]
                    ISC[0].set_ydata( isc_ydata )

                    if segments is not None:
                        for line in segments:
                            line.set_xdata([line.get_xdata()[0]-1, line.get_xdata()[0]-1])

                    if cache_markers is not None:
                        for line in cache_markers:
                            line.set_xdata([line.get_xdata()[0]-1, line.get_xdata()[0]-1])

                    if activity is not None and len(activity):
                        for l in activity:
                            segments.append(ax.axvline(x=abs(l - buffer_length), color = 'r'))
                    
                    if cache_marker > 0:
                        cache_markers.append(ax.axvline(x=abs(buffer_length), color = 'y'))
                        '''try:
                            cache_tmp = np.vstack(cache[0])
                            cache_tmp2 = np.mean(cache_tmp[:,:8], axis=1)
                            ax.plot( np.linspace( 0, cache_tmp.shape[0], cache_tmp.shape[0] ), cache_tmp2)
                        except:
                            pass'''
            
            plt.pause( 0.001 )

    def _data_cache(self, current_sample):
        
        # check if data should be saved
        if self.activity['markers'][-1] >= 1 or ( self.counter - self.activity['index'][-1] ) > 50:

            if self.cache_track == self.cache_downsampling_step:
                # check if it is first fill
                self.sparse_cache.append( copy.deepcopy(current_sample) )
                self.cache_track = 1

                # check if cache is full
                if len(self.sparse_cache) == (self.overwrite*2):
                    self.cache_downsampling_step = self.cache_downsampling_step *2
                    self.sparse_cache = self.downsample.uniform_cache(self.sparse_cache, self.overwrite)
                    self.sparse_cache = self.sparse_cache[:self.overwrite]

            else:
                self.cache_track += 1
        else:
            print('not actually caching')

    def _finalize_cache(self, merge = False, drop = False):

        if drop is False:
            if merge is False:
                if len(self.sparse_cache) < self.overwrite:
                    try:
                        self.cache.append(np.vstack(self.sparse_cache))
                    except: pass
                else:
                    temp_sparse_cache = self.downsample.uniform_cache(self.sparse_cache, self.overwrite)
                    #temp_sparse_cache = self.downsample.uniform_cache(self.sparse_cache[int(np.ceil(self.overwrite/2)):-int(np.floor(self.overwrite/2))], self.overwrite)
                    self.cache.append(np.vstack(temp_sparse_cache[:self.overwrite]))
                
                self.previous_downsampling_step = copy.deepcopy(self.cache_downsampling_step)
            else:
                # if short cache just adjust last value
                if len(self.sparse_cache) <= self.overwrite:
                    try:
                        self.cache[-1][-1] = self.sparse_cache[round(len(self.sparse_cache)/2)]
                    except:
                        pass
                # if longer cache compare cache downsampling rate to figure out how much of the cache to overwrite
                else:
                    if self.previous_downsampling_step > self.cache_downsampling_step:
                        self.cache[-1][-1] = self.sparse_cache[round(len(self.sparse_cache)/2)]
                    elif self.previous_downsampling_step == self.cache_downsampling_step:
                        self.cache[-1][-2] = self.sparse_cache[round(len(self.sparse_cache)/3)]
                        self.cache[-1][-1] = self.sparse_cache[round(len(self.sparse_cache)/3)*2]
                    else:
                        replace = math.log2(self.cache_downsampling_step) - math.log2(self.previous_downsampling_step) + 2
                        if replace >= self.overwrite:
                            replace = self.overwrite - 1
                        temp_sparse_cache = self.downsample.uniform_cache(self.sparse_cache, replace)
                        self.cache[-1][-replace:] = np.vstack(temp_sparse_cache[:replace])

        self.cache_downsampling_step= 1
        self.cache_track            = 1
        self.sparse_cache           = []

        while ( len(self.activity['markers'])-1 )/2 > len(self.cache):
            del self.activity['markers'][-2:]
            del self.activity['index'][-2:]
            return False
        
        return True

    def _segmentation_post_processing(self):
        self._activity_adjustment()
        # solve confidence rejection
        # self._confidence_rejection()
        self.new_activity = True

    def _activity_adjustment(self):
        find_indecies_c2c = []
        for key, i in enumerate( reversed( self.activity['markers'] ) ):
            if i == 1:
                index_of_onset = len(self.activity['markers']) - key - 1
                find_indecies_c2c.reverse()
                break
            elif i == 0:
                index_of_offset = len(self.activity['markers']) - key - 1
            elif i > 1:
                find_indecies_c2c.append(len(self.activity['markers']) - key - 1)

        self.current_segment_markers = [1]
        self.current_segment_index = [self.activity['index'][index_of_onset]]

        try:
            for key, idx in enumerate(find_indecies_c2c):
                if self.activity['index'][idx] - self.activity['index'][idx-1] > self.segmentation_delay*2:
                    self.current_segment_markers.extend([0,1])
                    self.current_segment_index.extend([self.activity['index'][idx],self.activity['index'][idx]])
        except:
            pass

        self.current_segment_markers.append(0)
        self.current_segment_index.append(self.activity['index'][index_of_offset])
        self.current_cache = self.cache[-int(len(self.current_segment_markers)//2):]

        del self.activity['markers'][index_of_onset:]
        del self.activity['index'][index_of_onset:]

        self.activity['markers'].extend(self.current_segment_markers)
        self.activity['index'].extend(self.current_segment_index)

    def _segmentation_check(self):

        self.new_onset = False

        cache_onset = 0
        # check activity status
        if self.activity['markers'][-1] == 1:
            active = True
            ISC = False
        elif self.activity['markers'][-1] >= 1:
            active = True
            ISC = True
        else:
            active = False
            ISC = False

        last_change_point = int(self.counter - self.activity['index'][-1])
        # look at the mean of MAVs across all channels for latest sample
        avg_signal_means = np.mean(self.MAV_buffer[-1])
        # check ISC offset
        if active:
            ISC_offset = self.counter - self.activity['index'][-1]
        else:
            ISC_offset = 0

        # Onset detection
        if active == False and avg_signal_means > self.onset_threshold:
            
            '''if len(self.sparse_cache) >= self.overwrite:
                self._finalize_cache()
                self.rest_cache = self.cache'''

            #self.reinit()
            self.activity['markers'].append(1)
            self.activity['index'].append(self.counter)
            self.new_onset = True
            self.add_cache_onset =  True

        # offset detection
        elif active and avg_signal_means < self.onset_threshold and last_change_point >= self.segmentation_delay:

            # if there is class to class too close to offset
            if ISC == True and last_change_point < self.segmentation_delay*1.5:
                if self.onset_data_caching == False:
                    self._finalize_cache(merge = True)
                else:
                    self._finalize_cache(drop = True)

                # delete class to classes that are within the range
                if self.activity['markers'][-1] != 1 and self.counter - self.activity['index'][-1] < self.segmentation_delay*1.5:
                    self.activity['markers'][-1] = 0

                self._segmentation_post_processing()

            # if onset and offset are too close delete onset
            elif last_change_point == self.segmentation_delay:
                del self.activity['markers'][-1]
                del self.activity['index'][-1]
                self._finalize_cache(drop = True)
            # otherwise add offset
            else:
                self.activity['markers'].append(0)
                self.activity['index'].append(self.counter)
                if self._finalize_cache():
                    self._segmentation_post_processing()
            
            self.__ISC_minimum = -np.inf

        # ISC detection
        elif ISC_offset > self.segmentation_delay*2:
            channel_diff = self.MAV_buffer[-1] - self.MAV_buffer[0]
            aggregate = np.sum(np.absolute(channel_diff), axis=0)/8

            if aggregate > self.onset_threshold*2:

                means_of_MAVs = np.mean(self.MAV_buffer,1)
                min_index = np.argmin(means_of_MAVs)
                self.ISC_minimum = means_of_MAVs[min_index]

                for key, i in enumerate( reversed( self.activity['markers'] ) ):
                    if i == 1:
                        index_of_onset = len(self.activity['markers']) - key - 1
                        break

                ISc_displacement = self.segmentation_delay - min_index

                if self.counter - self.activity['index'][index_of_onset] - ISc_displacement < self.segmentation_delay*2:
                    self.ISC_minimum = avg_signal_means
                    ISc_displacement = 0

                self.activity['markers'].append(self.activity['markers'][-1]+1)
                self.activity['index'].append(self.counter - ISc_displacement)
                self._finalize_cache()

                self.add_cache_onset =  True

        if active == True:
            if  self.counter - self.activity['index'][-1] > self.segmentation_delay*1.5 and (self.onset_data_caching == False or self.cache_downsampling_step < 8):
                self._data_cache(self.data_buffer[-1])
                if self.add_cache_onset:
                    cache_onset = self.counter
                    self.add_cache_onset = False
        else:
            pass
            #self._data_cache(self.data_buffer[-1])

        if 'aggregate' in locals():
            self.ISC_coeff.append(aggregate)
        else:
            aggregate = 0
            self.ISC_coeff.append(aggregate)
        
        if self.plot:
            idx_to_plot = copy.deepcopy(self.current_segment_index)
            idx_to_plot = [abs(x - self.counter) for x in idx_to_plot]
            self.add([avg_signal_means, aggregate, idx_to_plot, self.cache[-int(len(self.current_segment_markers)//2):], cache_onset])
            cache_onset = 0
        
        #self.counter = len(output_predictions)#+= 1

    def add( self, data ):
        """
        Add data to be plotted

        Parameters
        ----------
        data : numpy.ndarray (n_channels,)
            The data sample to be added to the plotting queue
        """
        try:
            self._queue.put( data, timeout = 1e-3 )
        except queue.Full:
            pass

    def close( self ):
        """
        Stop the plotting subprocess while releasing subprocess resources
        """
        self._exit_event.set()
        while self._queue.qsize() > 0:
            try:
                self._queue.get( timeout = 1e-3 )
            except queue.Empty:
                pass
        self._plotter.join()


#-----------------------------------------------------------------------------------------------------------------------
# PUBLIC METHODS
#-----------------------------------------------------------------------------------------------------------------------
    def run_segmentation(self, current_feature_vector, filtered_output, counter):

        self.counter = counter

        self.current_segment_markers = []
        self.current_segment_index = []   
        # current sample to run segmentation on
        self.MAV_buffer.append(current_feature_vector[0:8,])
        if len(self.MAV_buffer) > self.segmentation_delay:
            self.MAV_buffer.popleft()

        self.data_buffer.append(current_feature_vector)
        if len(self.data_buffer) > self.segmentation_delay:
            self.data_buffer.popleft()

        self.filtered_out.append(filtered_output)
        self._segmentation_check()

    def reinit(self, counter = -1):
        self.new_activity             = False
        # for segmentation_check
        if counter > 0:
            self.counter            = 0
        self.ISC_minimum            = -np.inf
        self.CR                       = []

        # data caching
        self.cache_downsampling_step= 1
        self.cache_track            = 1
        self.sparse_cache           = []
        self.cache                  = []
        self.rest_cache             = []
        self.previous_downsampling_step = 1

    def segmentation_process(self, feat, pred, gui, CONFIDENCE_REJECTION, raw_output_predictions, output_predictions):

        modified_gui = False
        
        # add datapoint to segmeentation
        self.run_segmentation( feat, pred, len(output_predictions) )
        # is there new movement onset?            
        # if new activity completed save data and add new segment cache to GUI
        if self.new_activity is True:
            
            '''self.CR_list.append( self.CR )
            CR_temp = np.hstack( self.CR_list )'''

            # update GUI
            packets = []
            for key, idx in enumerate(self.current_segment_index):
                
                if key % 2 == 0:
                    idx1 = idx
                    current_cache = np.array(self.current_cache[int(key//2)])
                    continue
                # only first 
                if key > 1 and modified_gui:
                    break
                
                idx2 = idx

                is_select = False#self.analyzer.filter(np.array(raw_output_predictions[idx1:idx2]), np.array(output_predictions[idx1:idx2]))

                select_labels = output_predictions[idx1:idx2]

                update_labels = np.unique( np.hstack( select_labels ) )

                start_point = 0
                for i in output_predictions[idx1:idx2]:
                    if i == 0:
                        start_point += 1
                    else:
                        break

                length = len(output_predictions[idx1+start_point:idx2]) * 10
                is_CR = False #CR_temp[-(num_new_segments-i)] < 90 and CR_temp[-(num_new_segments-i)] != 0
                
                status = 0
                if is_CR and CONFIDENCE_REJECTION:
                    status = -1
                elif is_select:
                    status = 1

                update_labels = np.delete(update_labels, np.where(update_labels == 0))

                if key == 1:
                    rest_length = (self.current_segment_index[0] - self.activity['index'][-len(self.current_segment_index)-1])*10

                order = [ int(np.ceil(key/2)), int(len(self.current_segment_index)/2) ]

                if len(update_labels) > 0 and current_cache.shape[0] >= self.overwrite/2:
                    packets.append( [current_cache, update_labels, status, length, rest_length] )
                else:
                    pass #print('update_labels', update_labels, "\n", 'current_cache.shape[0]', current_cache.shape[0], '\n', 'select_labels', select_labels)

                #self.reinit()
            
            if len(packets) > 0:
                for i, p in enumerate(packets):
                    packets[i].append([i+1,len(packets)])
                    
            gui.add( packets )



if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import time

    onset_threshold = 0.005

    segmentation = Segmentation(onset_threshold=onset_threshold)

    data_ranges= np.array([ [0.0025, 0.02, 0.033, 0.015], [0.0035, 0.025, 0.04, 0.018], [100, 200, 200, 35] ])
    order = [0, 1, 0, 2, 0, 1, 2, 1, 3, 0]

    sim_data = []
    sim_pred = []

    for key,i in enumerate(order):
        sim_pred.append([i]*int(data_ranges[2,i]))
        sim_data.append(np.random.uniform( low=data_ranges[0,i], high=data_ranges[1,i], size=(8,int(data_ranges[2,i]) ) ) )
        if key != len(order)-1:
            sim_pred.append([i]*10)
            sim_data.append(np.linspace(    start=[data_ranges[0,order[key]] for k in range(8)], 
                                            stop=[data_ranges[0,order[key+1]] for k in range(8)], 
                                            num=10,
                                            axis = 1 ) )

    sim_data = np.hstack(sim_data)
    sim_pred = np.hstack(sim_pred)

    for i, feat in enumerate(sim_data.T):
        segmentation.run_segmentation(feat, sim_pred[i])
        time.sleep( 0.001 )

        '''if i%200 == 0:
            segmentation.run_plot()
        elif i%100 == 0:
            segmentation.close()'''

    ISC_coeff = segmentation.ISC_coeff
    sim_data = np.vstack(sim_data)

    print(segmentation.activity)
    print(segmentation.cache)

