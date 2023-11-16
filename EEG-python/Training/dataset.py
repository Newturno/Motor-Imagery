import mne
from mne.datasets import eegbci
from mne.decoding import CSP
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class EEG:
    def __init__(self, path, subjects, runs):
        self.subpath = ''
        self.path = path
        print(path)
        self.subjects = subjects
        self.runs = runs
        
        # download data if does not exist in path.
        # self.load_data()
        self.data_to_raw()
    def filters(self, freq):
        raw = self.raw
        order = 5
        low, high = freq
        print(f">>> Apply filter.")
        #self.raw.filter(low, high, method='iir', iir_params=dict(order=order))
        self.raw.filter(low, high, fir_design='firwin', verbose='error')
        #self.raw.notch_filter(50,filter_length='auto', phase='zero')
        return  raw
    def raw_ica(self):
        raw = self.raw
        ica = mne.preprocessing.ICA(n_components=1, max_iter=100)
        ica.fit(raw)
        ica.exclude = [1, 2]  # details on how we picked these are omitted here
        ica.plot_properties(raw, picks=ica.exclude)
        ica.apply(raw)
        print('ICA DONE ????')
        return  raw
        
    def data_to_raw(self):
        fullpath = os.path.join(self.path, *self.subpath.split(sep='/'))
        #print(f">>> Extract all subjects from: {fullpath}.")
        extension = "fif"
        raws = []
        count = 1
        for i, subject in enumerate(self.subjects):
            sname = f"S{str(subject).zfill(3)}".upper()
            
            for j, run in enumerate(self.runs):
                rname = f"{sname}R{str(run).zfill(2)}".upper()
                path_file = os.path.join(fullpath, sname, f'{rname}.{extension}')
                #print(path_file)
                #print(f"Loading file #{count}/{len(self.subjects)*len(self.runs)}: {f'{rname}.{extension}'}")
                raw = mne.io.read_raw_fif( path_file , preload=True, verbose="error")
                raws.append(raw)
                count += 1

        raw = mne.io.concatenate_raws(raws)
        #montage = mne.channels.make_standard_montage('standard_1020')
        #raw.set_montage(montage)
        self.raw = raw
        return self.raw
    
    def set_reference(self,raw,channel):
        self.raw = raw.set_eeg_reference(ref_channels=channel)
        return self.raw
    def pickChannel(self,raw,channel):
        self.raw = raw.pick_channels(channel)
        return self.raw
    
    def raw_preprocess(self,raw,event_id,rest_stage=False):
        #return as epoch data
        dura_sam = 250 * 5
        trialofintereted = []
        padding = 3*250
        offset = 3 * 250
        offset2 = 1 * 250
        iir_param = dict(order=6, ftype='butter', output='sos')
        raw_pad = raw.get_data()
        channel_length = len(raw.info['ch_names'])
        padded = np.empty([2, dura_sam+(2*padding)+(2*offset)])
        tg_onset_sth = []
        
        self.eeg_raw = raw.copy().get_data()
        timestamps = raw.times
        eeg_df = pd.DataFrame(self.eeg_raw.T,columns=raw.ch_names)
        eeg_df['Timestamps'] = timestamps * 250
        
        # target filter
        for e in event_id:
            tg_onset_sth.extend(np.where(eeg_df['STIM MARKERS'] == int(e))[0])
        
        print(tg_onset_sth)
        
        #tg_onset_sth.extend(np.where(eeg_df['STIM MARKERS'] == int(2.0))[0])
        
        selected_onset = np.array(tg_onset_sth)
        onset = raw.copy().get_data()
        
        for trials in range(selected_onset.shape[0]):
            onset_idx = selected_onset[trials]
            #fist is channle second is selected time samples(1250)
            if rest_stage == True:
                temp = raw_pad[:,onset_idx - 250 :onset_idx + 250 + offset2]
            else:                                
                temp = raw_pad[:,onset_idx - offset :onset_idx+dura_sam + offset2]

            #print(temp -= mean)
            #print(temp.shape)
            #print(padded.shape)
            # for ch in range(temp.shape[0]):
            #     #print("channel" + str(ch))
            #     if ch == 2:
            #         break
            #     #print(temp[ch, :].shape)
            #     padded[ch, :] = np.pad(temp[ch, :], (padding, padding), mode='mean')
            #     #print("After padded")
            #     #print(padded.shape)
            #filtered_data = mne.filter.filter_data(temp[0:3,:], sfreq=250, l_freq=8, h_freq=13,method='fir',verbose='error')
            filtered_data = mne.filter.filter_data(temp[0:channel_length-1,:], sfreq=250, l_freq=8, h_freq=13,method='iir',iir_params=iir_param,verbose='error') # bandpass
            filtered_temp = mne.filter.notch_filter(filtered_data, Fs=250, freqs=50, verbose='error') # notch
            
            #print(filtered_temp.shape)
            #plt.plot(filtered_temp[0,:])
            if rest_stage == True:
                filtered = filtered_temp[:,offset2:offset2+250]
            else:
                filtered = filtered_temp[:,offset2:offset+dura_sam]

            trialofintereted.append(filtered)
        temp = [t[np.newaxis, ...] for t in trialofintereted]
        data = np.concatenate(temp)
        y = eeg_df[eeg_df['STIM MARKERS'].isin(event_id)]
        y = y['STIM MARKERS'].to_numpy().astype(int) - 1
        self.X = data
        self.y = y
        return self.X,self.y

    def get_X_y(self,epochs,tmin=0,tmax=4):
        
        #epochs=epochs.resample(160)
            #events , event_id=self.create_epochs()
        self.X = epochs.get_data()
        self.y = epochs.events[:, -1]-1
        return self.X, self.y 
    def apply_baseline(self,epochs):
        baseline = []
        for e in range(epochs.shape[0]):
            avg_epoch = np.mean(epochs[e,:,:(250*2)], axis=1)
            baseline.append(avg_epoch)
        baseline = np.asarray(baseline)
        
        for e in range(epochs.shape[0]):
            for c in range(int(epochs.shape[1])):
                epochs[e,c,:] = epochs[e,c,:] - baseline[e,c]
        
        return epochs
    def epochs(self,raw,tmin,tmax,baseline):
        events = mne.find_events(raw, stim_channel='STIM MARKERS',verbose='error')
        epochs = mne.Epochs(
        raw,
        events,
        event_id=[1,2],
        tmin=tmin,
        tmax=tmax,
        picks="data",
        on_missing='warn',
        proj=True,
        baseline=baseline,
        preload=True,
        verbose='error'
            )
        return epochs
    def apply_CSP(self,X,y,n_component=4,reg=None, log=True, norm_trace=False):
        csp = CSP(n_components=n_component, reg=None, log=True, norm_trace=False)
        X_csp = csp.fit_transform(X,y)
        
        return X_csp


class Physionet:
    def __init__(self, path, base_url, subjects, runs):
        self.subpath = ''
        self.path = path
        print(path)
        self.base_url = base_url
        self.subjects = subjects
        self.runs = runs
        
        # download data if does not exist in path.
        # self.load_data()
        self.data_to_raw()
    
    def load_data(self):
        print(f">>> Start download from: {self.base_url}.")
        print(f"Downloading files to: {self.path}.")
        for subject in self.subjects:
            eegbci.load_data(subject,self.runs,path=self.path,base_url=self.base_url)
        print("Done.")
        return self.raw
    def filter(self, freq):
        raw = self.raw
        low, high = freq
        print(f">>> Apply filter.")
        self.raw.filter(low, high, fir_design='firwin', verbose=20)
        #self.raw.notch_filter(50,filter_length='auto', phase='zero')
        return  raw
    def raw_ica(self):
        raw = self.raw
        ica = mne.preprocessing.ICA(n_components=1, max_iter=100)
        ica.fit(raw)
        ica.exclude = [1, 2]  # details on how we picked these are omitted here
        ica.plot_properties(raw, picks=ica.exclude)
        ica.apply(raw)
        print('ICA DONE ????')
        return  raw
        
    def get_events(self):
        event_id = dict(T1=0, T2=1) # the events we want to extract
        events, event_id = mne.events_from_annotations(self.raw, event_id=event_id)
        return events, event_id
    
    def get_epochs(self, events, event_id):
        picks = mne.pick_types(self.raw.info, eeg=True, exclude='bads')
        tmin = 0
        tmax = 4
        epochs = mne.Epochs(self.raw, events, event_id, tmin, tmax, proj=True, 
                            picks=picks, baseline=None, preload=True)
        return epochs
    
    def data_to_raw(self):
        fullpath = os.path.join(self.path, *self.subpath.split(sep='/'))
        #print(f">>> Extract all subjects from: {fullpath}.")
        extension = "fif"
        raws = []
        count = 1
        for i, subject in enumerate(self.subjects):
            sname = f"S{str(subject).zfill(3)}".upper()
            
            for j, run in enumerate(self.runs):
                rname = f"{sname}R{str(run).zfill(2)}".upper()
                path_file = os.path.join(fullpath, sname, f'{rname}.{extension}')
                #print(path_file)
                #print(f"Loading file #{count}/{len(self.subjects)*len(self.runs)}: {f'{rname}.{extension}'}")
                raw = mne.io.read_raw_fif( path_file , preload=True, verbose='WARNING' )
                raws.append(raw)
                count += 1

        raw = mne.io.concatenate_raws(raws)
        eegbci.standardize(raw)
        montage = mne.channels.make_standard_montage('standard_1005')
        raw.set_montage(montage)
        print(raw.info['ch_names'])
        print(raw.info['sfreq'])
        self.raw = raw
    
    def create_epochs(self):
        print(">>> Create Epochs.")
        
        events, event_id = self.get_events()
        self.epochs = self.get_epochs(events, event_id)
        print("Done.")
        return events , event_id
    
    def get_X_y(self):
        if self.epochs is None:
            events , event_id ,epochs=self.create_epochs()
        self.X = self.epochs.get_data()
        self.y = self.epochs.events[:, -1]
        return self.X, self.y