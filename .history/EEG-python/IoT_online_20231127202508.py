from psychopy import core,gui
from data_utils import *
import logging
from config import *
from beeply.notes import *
import sounddevice as sd
import soundfile as sf
import concurrent.futures
import time
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams
import sys
sys.path.append(os.path.abspath(os.path.join('Training')))
from model import ConvNet,ConvNet2
from paho.mqtt import client as mqtt_client
import torch
  


def startExperiment(board,info):
    logging.basicConfig(filename=info[0],filemode='a')
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger().setLevel(logging.DEBUG)
    global IS_FINISH
    
    #MQTT CONNECT
    if info[4] == True:
        #client = None
        path = ONLINE_PATH
        model = ConvNet2()
        model.load_state_dict(torch.load(path))
        model.eval()
        print("Model loaded")
        client = connect_mqtt()
        print("Mqtt connect")
    
    data = board.get_board_data()
    if FOOT == True:
        sequence = STIMULIT_SEQUENCE
    else:
        sequence = STIMULIT_SEQUENCE
    sound_array, fs_array = getSound()
    #Start
    logging.info(f"Experiment order: {sequence}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        for trials in range(NUM_TRIAL):
            print(f'{"Trials :" + str(trials+1)}')
            #voice left and right and rest
            #rest 2 sec
            playSound(sound_array[4],fs_array[4],1,board,3.0)
            #cue and Imagine 6 sec
            playSound(sound_array[sequence[trials]],fs_array[sequence[trials]],6,board,BLOCK_MARKER[sequence[trials]])
            
            if info[4] == True:
                online_predict(board,info,client,model)
    
    
    if info[4] == True:
        IS_FINISH = True
    else:
        #saving
        block_name = f'{PARTICIPANT_ID}R{ORDER_NUM:02d}' 
        headname = f'{int(info[1]):03d}'
        block_name = f'{"S" + str(headname)}R{int(info[2]):02d}'
        data = board.get_board_data()
        data_copy = data.copy()
        raw = getdata(data_copy,info[3],n_samples = 250,dropEnable = DROPENABLE)
        save_raw(raw,block_name,RECORDING_DIR,info)
        IS_FINISH = True

    
    return IS_FINISH

def playSound(sound,fs,second,board,marker):
    start = time.time()
    sd.play(sound, fs)
    sd.wait()
    print(f"Marker: {marker}")
    board.insert_marker(marker)
    core.wait(second)
    stop  = time.time()
    print(f"Trigger time = {(stop-start)}  Second ")

def publish(client,output):
    msg = f"{output}"
    # main function to send
    result = client.publish(TOPIC, output)
    # result: [0, 1]
    status = result[0]
    if status == 0:
        print(f"Send `{msg}` to topic `{TOPIC}`")
    else:
        print(f"Failed to send message to topic {TOPIC}")

def online_predict(board_shim,info,client,model):
    #Online inference                
    #Get data from board
    data = board_shim.get_board_data()
    #data = board_shim.get_current_board_data(250*9)
    data_copy = data.copy()
    #sending and predict
    raw = getdata(data_copy,info[3],n_samples = 250,dropEnable = True)
    raw=raw.notch_filter([50])
    raw.filter(8,14, method='fir', verbose=20)
                
    #save file
    # headname = f'{int(info[1]):03d}'
    # block_name = f'{"S" + str(headname)}R{int(info[2]):02d}'
    # save_raw(raw,block_name,ONLINE_FOLDER,info)
    # info[2] = int(info[2]) + 1
    #change to epoch data before send
    logging.info('--------------------------------------------------')
    train_epochs,epochs_raw_data,labels = getepoch(raw,-3,5)
    logging.info('-------------------------------------------------')
    logging.info('Sending')
    X_t,y_train = train_epochs.copy(),labels
                

    X_tensor = torch.from_numpy(X_t).float()
    y_tensor = torch.from_numpy(y_train).long()

    output = model(X_tensor)
    logging.info("Prob:{}".format(str(output)))
    logging.info("Actual:{}".format(str(y_tensor)))

    _, predicted = torch.max(output, 1)
    logging.info("Predicted:{}".format(str(predicted)))
    final_output = int(predicted[0].item())
    
    #print(IoT_output)
    if final_output == 0:
        output = 'left'
    elif final_output == 1:
        output = 'right'
    print(output)
    publish(client,output)
    core.wait(33)
    throw = board_shim.get_board_data() 
    #Back to rest stage?
    #wait until finish rest stage       

def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(CLIENT_ID)
    client.username_pw_set(USERNAME, PASSWORD)
    client.on_connect = on_connect
    client.connect(BROKER, PORT)
    return client

def main():
    global STIM_CHECK
    global PLAY_VIDEO
    global IS_FINISH
    # GUI input
    logging.info("Begin experiment")
    myDlg = gui.Dlg(title="Motor Imagery experiment",screen = 2)
    myDlg.addField('Name:',initial='fabby')
    myDlg.addField('Participant ID',initial='17')
    myDlg.addField('Order Number:',initial='26')
    myDlg.addField('Board:', choices=["OpenBCI", "Unicorn","Dummy"],initial='Off' )
    myDlg.addField('Online:', choices=["On", "Off"],initial='On' )
    ok_data = myDlg.show()  # show dialog and wait for OK or Cancel
    if myDlg.OK:  # or if ok_data is not None
        ok_data[3] = BOARD[ok_data[3]]
        ok_data[4] = IS_ONLINE[ok_data[4]]
        print(ok_data)
    else:
        print('user cancelled')
    
    BoardShim.enable_dev_board_logger()
    #brainflow initialization 
    params = BrainFlowInputParams()
    #params.serial_port = SERIAL_PORT
    if ok_data[3] == 0:
        #think pulse
        print("Think pulse activate")
        params.serial_port = SERIAL_PORT
        THINKPULSE_CONFIG = 'x1040111Xx2040111Xx3040111Xx4040111Xx5140010Xx6140010Xx7140010Xx8140010X'
        board_shim = BoardShim(ok_data[3], params)
    else:
        board_shim = BoardShim(ok_data[3], params)

    #board prepare
    try:
        board_shim.prepare_session()
        if ok_data[3] == 0:
            board_shim.config_board(THINKPULSE_CONFIG)
            board_shim.config_board('<')

    except brainflow.board_shim.BrainFlowError as e:
        print(f"Error: {e}")
        print("The end")
        time.sleep(1)
        sys.exit()
    #board start streaming
    board_shim.start_stream()

    ##############################################
    # Experiment session
    ##############################################
    while True:
        start = time.time()
        IS_FINISH = startExperiment(board_shim,ok_data)
        if IS_FINISH:
            stop  = time.time()
            print(f"Total experiment time = {(stop-start)/60} ")
            core.wait(10)
            break
    logging.info('End')
    
    if board_shim.is_prepared():
            logging.info('Releasing session')
            # stop board to stream
            board_shim.stop_stream()
            board_shim.release_session()

def getSound():
    sound_array = []
    fs_array = []
    for i in range(5):
        data, fs = sf.read(SOUND_DICT[i])
        sound_array.append(data)
        fs_array.append(fs)
    return sound_array,fs_array


if __name__ == "__main__":
    #getSound()
    main()


                