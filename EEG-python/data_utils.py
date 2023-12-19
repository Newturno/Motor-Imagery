import os
import datetime
from pathlib import Path
import mne
from mne.datasets import eegbci
from mne.time_frequency import tfr_multitaper
from config import *
from brainflow.board_shim import BoardShim
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import random
from matplotlib.colors import TwoSlopeNorm
import pandas as pd
import numpy as np
from pymongo import MongoClient
import logging
logging.basicConfig(filename=NAME,filemode='a')
logging.getLogger().setLevel(logging.DEBUG)
def now_datestring():
    return datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

def save_raw_to_dataframe(raw,name):
    epoch_dataframe = raw.copy().to_data_frame()
    csv_folder = create_session_folder(PARTICIPANT_ID,CSV_DIR)
    csv_name = f'{name}.csv'
    epoch_dataframe.to_csv(os.path.join(csv_folder,csv_name),encoding='utf-8')

def save_raw(raw, name,dir,info):
    headname = f'{int(info[1]):03d}'
    save_path = f'{"S"+str(headname)}'
    folder_path = create_session_folder(save_path,dir)
    raw.save(os.path.join(folder_path, f'{name}{TYPE_OF_FILE}'))
    return os.path.basename(folder_path)

def save_fig(data,name,trial):
    folder_path = create_session_folder(PARTICIPANT_ID,ERD_FOLDER)
    data.savefig(os.path.join(folder_path, f'{name}_{trial:02d}{FIG_FILE}'))
    return os.path.basename(folder_path)

def create_session_folder(subj,dir):
    folder_name = f'{subj}'
    if os.path.isdir(os.path.join(dir, folder_name)):
        folder_path = os.path.join(dir, folder_name)
    else:
        folder_path = os.path.join(dir, folder_name)
        Path(folder_path).mkdir(exist_ok=True)
    return folder_path

def load_recording(rec_folder):
    return mne.io.read_raw_fif(os.path.join(RECORDING_DIR, rec_folder, 'raw.fif'))

def getdata(data,board,clear_buffer=False,n_samples=None,dropEnable = False):
    """
        Get data that has been recorded to the board. (and clear the buffer by default)
        if n_samples is not passed all of the data is returned.
    """
    # Creating MNE objects from brainflow data arrays
    # the only relevant channels are eeg channels + marker channel
    # get row index which holds markers
    marker_channel = BoardShim.get_marker_channel(board)
    
    #row which hold eeg data
    eeg_channels = BoardShim.get_eeg_channels(board)[:16]
    print(len(eeg_channels))
    print(f'Before {data[eeg_channels]}')
    #data[eeg_channels] = data[eeg_channels] / 1e6
    
    #for active electrode 
    #data[eeg_channels] = (data[eeg_channels,:] * 24) / 8
    print(f'After {data[eeg_channels]}')
    #eeg row + marker row (8 + 1)
    data = data[eeg_channels + [marker_channel]]
    print(len(data))
    
     #string of all channel name ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']
    ch_names = ['FP1','FP2','F7','F3','FZ','F4','F8','T3','C3','CZ','C4','T4','P3','P4','O1','O2']+ ['STIM MARKERS']
    print(len(ch_names))
    #ch_names = ['C3', 'C4', 'CZ'] + ['STIM MARKERS']
    ch_types = ['eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','stim']
    
    #ch_names = ['PZ','T3','C3','CZ','C4','T4'] + ['STIM MARKERS']
    #ch_types = ['eeg','eeg','eeg','eeg','eeg','eeg','stim']
    #sample rate
    sfreq = BoardShim.get_sampling_rate(board)
    #Create Raw data from MNE
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    raw_data = raw.copy()
    
    if dropEnable == True:
        raw_data.pick_channels(['C3','C4','STIM MARKERS']) 

    
    return raw_data

def getepoch(raw,tmin,tmax,reject_bad=False,on_missing='warn'):
    if reject_bad == False:
        reject_criteria = dict(eeg=100e-6)  # 100 µV
    else: 
        reject_criteria = None
    flat_criteria = dict(eeg=1e-6)  # 1 µV
    markers= [0,1,2]
    epochs_list = []
    baseline = (-3,0)
    #raws = [raws]
    events = mne.find_events(raw,stim_channel='STIM MARKERS')
    event_dict = {'Left': 1, 'Right': 2}
    epochs = mne.Epochs(raw,events,event_id=event_dict,tmin=tmin, tmax=tmax,reject=None, picks="data",
    on_missing=on_missing, preload=True, baseline=baseline,reject_by_annotation=False,proj=True,verbose=True)
    #epochs = epochs.copy().resample(160)
    #epochs_list.append(epochs)
    #epochs = mne.concatenate_epochs(epochs_list)
    labels = epochs.events[:,-1]-1
    #print(f'Found {len(labels)} epochs')
    #print(epochs) 
    
    return epochs.get_data(),epochs,labels


def get_database():
    # Provide the mongodb atlas url to connect python to mongodb using pymongo
    CONNECTION_STRING = 'mongodb+srv://newturno:newturno123@cluster0.nwci3nc.mongodb.net/'

    # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
    client = MongoClient(CONNECTION_STRING)

    # Create the database for our example (we will use the same database throughout the tutorial
    return client['MI']

def upload(data):
    dbname = get_database()
    collection_name = dbname["user_data"]
    rec = collection_name.insert_one(data)
    return rec 
    
def send_raw(database):
    data = {
            "name": database.str,
            "X": database.train.tolist(),
            "y": database.lable.tolist()
    }
    #rec = upload(data)
    #if rec != None:
    requests.post("http://localhost:8000/items",json=data)
        
def send_train(database):
    data = {
            "name": database.str,
            "X": database.train.tolist(),
            "y": database.lable.tolist()
    }
    requests.post("http://localhost:8000/train",json=data)

def randomStimuli(numTrials):
    '''Input
        1. Stimuli set
        2. randomnumber
        3. marker
        Output
        
        Dict of stimuli and marker    
    '''
    #ubuntu, delete folder
    image_stimuli = []
    video_stimuli = []
    for cat in CATEGORIES:
        l = get_filenames_in_path(f"{IMAGE_FOLDER}{cat}")
        image_stimuli.append(f'{IMAGE_FOLDER}{cat}{"/"}{l[0]}')
    
    for cat in CATEGORIES:
        v = get_filenames_in_path(f"{VIDEO_FOLDER}{cat}")
        video_stimuli.append(f'{VIDEO_FOLDER}{cat}{"/"}{v[0]}')
    image_list=[]
    video_list=[]
    image_list,numIm_list = randomlist(image_stimuli,numTrials,0,1)
    video_list,numVi_list = randomlist(video_stimuli,numTrials,0,1)
    return image_list,numIm_list, video_list,numVi_list

def randomlist(stimuli,num_range,nmin,nmax):
    return_list = []
    num_list = []
    left_i = 0
    right_i = 0
    for i in range(num_range):
        # random check number of list before append in list
        num = random.randint(nmin,nmax) #0,1
        if num == nmin:
            if left_i   != (num_range/2):
                return_list.append(stimuli[num])
                num_list.append(num)
                left_i += 1
            elif left_i == (num_range/2) and right_i != (num_range/2):
                return_list.append(stimuli[nmax])
                num_list.append(nmax)
                right_i += 1

                
        elif num == nmax:
            if right_i  != (num_range/2):
                return_list.append(stimuli[num])
                num_list.append(num)
                right_i += 1
            elif left_i != (num_range/2) and right_i == (num_range/2):
                return_list.append(stimuli[nmin])
                num_list.append(nmin)
                left_i += 1
                 
    return return_list,num_list
    
def Erd_Plot(epochs,trial):
    
    print("ERE PROCESS")
    freqs = np.arange(8, 14)  # frequencies from 2-35Hz
    vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
    baseline = (0, 2)  # baseline interval (in s)
    cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center & max ERDS
    kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1,
                buffer_size=None, out_type='mask')  # for cluster test
    tmin, tmax = 0, 7
    
    tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles=freqs, use_fft=True,
                     return_itc=False, average=False, decim=1)
    tfr.crop(tmin, tmax).apply_baseline(baseline, mode="percent")
    
    
    df = tfr.to_data_frame(time_format=None, long_format=True)
    print(df)
    # Map to frequency bands:
    freq_bounds = {'_': 0,
               'delta': 3,
               'theta': 7,
               'alpha': 13,
               'beta': 35,
               'gamma': 140}
    df['band'] = pd.cut(df['freq'], list(freq_bounds.values()),
                    labels=list(freq_bounds)[1:])
    
    # Filter to retain only relevant frequency bands:
    freq_bands_of_interest = ['delta', 'theta', 'alpha', 'beta']

    df = df[df.band.isin(freq_bands_of_interest)]

    df['band'] = df['band'].cat.remove_unused_categories()

    # Order channels for plotting:
    df['channel'] = df['channel'].cat.reorder_categories(('C3', 'C4'),
                                                        ordered=True)
    
    g = sns.FacetGrid(df, row='band', col='condition')
    g.map(sns.lineplot, 'time', 'value', 'channel', n_boot=10)
    axline_kw = dict(color='black', linestyle='dashed', linewidth=0.5, alpha=0.5)
    g.map(plt.axhline, y=0, **axline_kw)
    g.map(plt.axvline, x=0, **axline_kw)
    g.set(ylim=(-1, None))
    g.set_axis_labels("Time (s)", "ERDS (%)")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.add_legend(ncol=2, loc='lower center')
    g.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.08)
    save_fig(g,NAME,trial)

def apply_baseline(epochs):
        baseline = []
        for e in range(epochs.shape[0]):
            avg_epoch = np.mean(epochs[e,:,:(250*2)], axis=1)
            baseline.append(avg_epoch)
        baseline = np.asarray(baseline)
        
        for e in range(epochs.shape[0]):
            for c in range(int(epochs.shape[1])):
                epochs[e,c,:] = epochs[e,c,:] - baseline[e,c]
        
        return epochs
def raw_preprocess(raw):
    #return as epoch data
        dura_sam = 250 * 5
        trialofintereted = []
        padding = 3*250
        offset = 3 * 250
        offset2 = 1 * 250
        iir_param = dict(order=6, ftype='butter', output='sos')
        raw_pad = raw.get_data()
        padded = np.empty([2, dura_sam+(2*padding)+(2*offset)])
        tg_onset_sth = []
        
        eeg_raw = raw.copy().get_data()
        timestamps = raw.times
        eeg_df = pd.DataFrame(eeg_raw.T,columns=raw.ch_names)
        eeg_df['Timestamps'] = timestamps * 250
        
        tg_onset_sth.extend(np.where(eeg_df['STIM MARKERS'] == int(1.0))[0])
        tg_onset_sth.extend(np.where(eeg_df['STIM MARKERS'] == int(2.0))[0])
        
        selected_onset = np.array(tg_onset_sth)
        onset = raw.copy().get_data()
        
        for trials in range(selected_onset.shape[0]):
            onset_idx = selected_onset[trials]
            #fist is channle second is selected time samples(1250)
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
            filtered_data = mne.filter.filter_data(temp[0:3,:], sfreq=250, l_freq=8, h_freq=13,method='iir',iir_params=iir_param,verbose=False) # bandpass
            filtered_temp = mne.filter.notch_filter(filtered_data, Fs=250, freqs=50, verbose=False) # notch
            #print(filtered_temp.shape)
            #plt.plot(filtered_temp[0,:])
            filtered = filtered_temp[:,offset2:offset+dura_sam]
            
            trialofintereted.append(filtered)
        temp = [t[np.newaxis, ...] for t in trialofintereted]
        data = np.concatenate(temp)
        y = eeg_df[eeg_df['STIM MARKERS'].isin([1.0,2.0])]
        y = y['STIM MARKERS'].to_numpy().astype(int) - 1
        X = data
        y = y
        return X,y