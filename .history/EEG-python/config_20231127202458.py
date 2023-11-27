#Image config
from array import array
from brainflow import BoardIds
import random

#Application config
#==============================================
# experiment parameters
#==============================================
#[1366,768]
#[1536,864]
#SCREEN_SIZE:array = [1920,1080]
SCREEN_SIZE:array = [1536, 864]
TOTAL_IMAGE:int = 2
NUM_TRIAL:int = 20#20
STIM_CHECK = 0
# baseline run
BASELINE_EYEOPEN:int = 60 #60second
BASELINE_EYECLOSE:int = 60 #60second
ALERT_TIME:int = 800 #8second 
INSTRUCTION_TIME:int = 2 
#stimuli time (left arrow and right arrow)
STIM_TIME:int = 5 #4second
STIM_BLINK_TIME:int = 0 #0second
FIXATION_TIME:int = 2 #10 "+" inter trial interval
"""EXE_COUNT:int = 0
IMAGINE_COUNT:int = 0
EXECUTE_NO:array=[3,5,7,9,11,13]
IMAGINE_NO:array=[4,6,8,10,12,14]"""

#EEG config
TIME_OUT = 0
IP_PORT = 0
IP_PROTOCOL = 0
IP_ADDRESS =''
BOARD_ID = 0 # 0 = openbci 8 = unicorn
SERIAL_PORT = "/dev/cu.usbserial-DM03GRPK"
MAC_ADDRESS = ''
OTHER = ''
STREAMER_PARAMS = ''
SERIAL_NUMBER = ''
FILE = ''
MASTER_BOARD = 0
PRESET = 0

#Marker config
#BLOCK_DICT:dict[int,str] = {1:'execute_left',2:'executed_right',3:'imagine_left',4:'imagine_right' }
BLOCK_DICT:dict[int,str] = {1:'execute',2:'imagine',3:'executed',4:'imagine' }
BLOCK_MARKER:dict = {0:1.0 , 1:2.0}


SOUND_DICT:dict[int,str] = {0:'./sound/Left.wav',1:'./sound/Right.wav',4:'./sound/Rest.wav',3:'./sound/Start.wav',2:'./sound/Stop.wav'}
FOOT = True

IMAGE_DICT:dict[int,str] = {0:'./images/left/left2.png',1:'./images/right/right2.png'}
VIDEO_DICT = [
 { 0:'./video/left/left3.mp4', 1: './video/right/right3.mp4' },
 { 0: './video/left/left2.avi', 1: './video/right/right2.avi' },
 ]
STIMULIT_SEQUENCE = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
#STIMULIT_SEQUENCE = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#STIMULIT_SEQUENCE = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
IS_VIDEO:bool = True
PLAY_VIDEO:bool = False
PLAY_SOUND:bool = False
CALIBRATION:bool = False
DROPENABLE:bool = False
IS_BASELINE:bool = False
IS_FINISH:bool = False

#Session 1, 2 no random 3,4 random
# odd is excuted even is imagined
#experiemet setting
NAME:str = 'New'
PARTICIPANT_ID:str = 'S003'
ORDER_NUM:int = 1
VIDEO_ORDER:int = 0 # 0:realhand, 1:animation
NO_RANDOM:bool = True
SCREEN_NUM:int = 2
# Online task
ONLINE_ID:str = 'S01_EX'
IS_ONLINE:dict = {'Off':False , 'On':True}

#co adaptive
CHOICE:dict = {'0':False , '1':True}
#unicorn 8 neuro crown 23
IS_DUMMY:dict = {'0': 0 , '1': 8, '2': -1}
BOARD:dict = {"OpenBCI":0 , "Unicorn":8,"Dummy":-1,"Goldcup":0}
#BOARD:dict = {0:0 , 1:0,2:-1}
IS_FINETUNE:bool = False

MARKER:dict = {'left':1.0,'right':2.0,'fixation':4.0}

#All directory
RECORDING_DIR:str = 'dataset/recorded_EEG'
CSV_DIR:str = 'csv/'
TYPE_OF_FILE ='.fif'
FIG_FILE = '.png'
FOLDER:str = 'EEG-python/'
CATEGORIES:list[str] = ['left','right']
IMAGE_FOLDER:str = 'images/'
VIDEO_FOLDER:str = 'video/'
ERD_FOLDER:str = 'erd/'
ONLINE_FOLDER:str = 'dataset/online_EEG'
RECORED_PATH = "/Users/pongkornsettasompop/Desktop/work/Motor-Imagery/EEG-python/dataset/recorded_EEG"
ONLINE_PATH = "/Users/pongkornsettasompop/Desktop/work/Motor-Imagery/EEG-python/Training/save_weight/S17_Conv2_iir_-2-5_8-13_3ch/0.6726_S17_Conv2_iir_-2-5_8-13_3ch_0.6726_58.3333.pth"

DUMMY_PATH = "/Users/pongkornsettasompop/Desktop/work/Motor-Imagery/EEG-python/Training/save_weight/Test/0.7243_Test_0.7243_50.0000.pth"

ONLINE_PATH_DICT = {
    'Fabby':"/Users/pongkornsettasompop/Desktop/work/Motor-Imagery/EEG-python/Training/save_weight/S18_Conv2_iir_-2-5_8-13_3ch/0.6620_S18_Conv2_iir_-2-5_8-13_3ch_0.6620_65.0000.pth",
    'Pop':"/Users/pongkornsettasompop/Desktop/work/Motor-Imagery/EEG-python/Training/save_weight/S19_Conv2_irr_-2-5_8-13_3ch/0.6654_S19_Conv2_irr_-2-5_8-13_3ch_0.6654_60.0000.pth",
    'New':"/Users/pongkornsettasompop/Desktop/work/Motor-Imagery/EEG-python/Training/save_weight/S17_Conv2_iir_-2-5_8-13_3ch/0.6726_S17_Conv2_iir_-2-5_8-13_3ch_0.6726_58.3333.pth"
}
BG_DICT ={
    'Fabby':"img/fabby_bg.png",
    'Pop':"img/pop_bg.png",
    'New':"img/new_bg.png"
}



#MQTT SETUP
BROKER = '10.42.0.1'
PORT = 1883
TOPIC = "hci"
# Generate a Client ID with the publish prefix.
CLIENT_ID = f'publish-{random.randint(0, 1000)}'
USERNAME = 'dobot'
PASSWORD = '12345678#'
