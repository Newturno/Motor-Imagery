import sys
from PyQt5.QtWidgets import (
    QMainWindow, QApplication,QFileDialog,
    QLabel, QToolBar, QAction, QStatusBar,QCheckBox,QDialog,
    QPushButton,QHBoxLayout,QVBoxLayout,QStackedLayout,QWidget,QLineEdit,QComboBox
)
from PyQt5.QtGui import QIcon,QPixmap
from PyQt5.QtCore import Qt,QSize
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
from model import ConvNet2
from paho.mqtt import client as mqtt_client
import torch


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        # setting title
        self.setWindowTitle("Mode selection")
        screen = QApplication.primaryScreen()
        size = screen.size()
        self.ui_width = size.width()
        self.ui_height = size.height()
        print('Size: %d x %d' % (size.width(), size.height()))
        # setting geometry# 
        #self.resize(720, 430)
        self.resize( self.ui_width,self.ui_height)
        self.UiComponents()
        self.fileURL = ONLINE_PATH_DICT["New"]
        self.boardID = BOARD["OpenBCI"]
        
        
        # showing all the widgets
        self.show()
    
    # method for widgets
    def UiComponents(self):
        self.ax = int((self.ui_width / 2) - 120)
        self.ay = int(self.ui_height)
        # Load the background image using QPixmap
        background_image = QPixmap("img/main_bg.png")
        # Create a QLabel to display the background image
        self.background_label = QLabel(self)
        self.background_label.setPixmap(background_image) 
        self.background_label.setGeometry(1,1,self.ui_width,self.ui_height)
        
        self.qlabel = QLabel(self)
        self.qlabel.setGeometry(self.ax-50, 150, 500, 50)
        
        
        online_btn = QPushButton("Online Mode",self)
        online_btn.setGeometry(self.ax, 200, 230, 50)
        online_btn.pressed.connect(self.online)
        
        file_btn = QPushButton('Open File', self)
        file_btn.clicked.connect(self.showDialog)
        file_btn.setGeometry(self.ax, 270, 230, 50)
        
        btn = QPushButton("Exit",self)
        btn.setGeometry(self.ax, 340, 230, 50)
        btn.pressed.connect(self.closeEx)
        
        board_input = QComboBox(self)
        board_input.addItems(["OpenBCI", "Unicorn", "Dummy"])
        board_input.setCurrentIndex(0)
        board_input.setGeometry(self.ax , 410, 230, 30)
        board_input.activated[str].connect(self.handleInput)
        
        weight_input = QComboBox(self)
        weight_input.addItems(["Fabby", "Pop", "New"])
        weight_input.setCurrentIndex(2)
        weight_input.setGeometry(self.ax , 480, 230, 30)
        weight_input.activated[str].connect(self.handleWeight)
        
        self.showMaximized() 
        
        
        
    def handleInput(self, text):
        self.boardID = BOARD[text]
    def handleWeight(self,text):
        self.fileURL = ONLINE_PATH_DICT[text]
        self.background_label.setPixmap(QPixmap(BG_DICT[text])) 
        filename = os.path.basename(self.fileURL)
        self.qlabel.setText(filename)
    
    
    def online(self):
        self.info = [self.boardID,True]
        print(self.info[0])
        self.w = RecordWindow(self.info,self.fileURL)
        self.w.show()
        self.close()
        print("Online")
        
    def closeEx(self):
        self.close()
        
    def showDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly  # You can specify additional options here if needed

        self.fileURL , _ = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*);;Text Files (*.txt)", options=options)

        if self.fileURL :
            # Do something with the selected file, for example, print its path
            filename = os.path.basename(self.fileURL)
            self.qlabel.setText(filename)
            print("Selected file:", self.fileURL )
            
            
class RecordWindow(QWidget):
    def getSound(self):
        sound_array = []
        fs_array = []
        for i in range(5):
            data, fs = sf.read(SOUND_DICT[i])
            sound_array.append(data)
            fs_array.append(fs)
        return sound_array,fs_array    
    
    def startExperiment(self,board,info):
        logging.basicConfig(filename="Online-Log",filemode='a')
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger().setLevel(logging.DEBUG)
        global IS_FINISH
        
        #MQTT CONNECT
        if info[1] == True:
            #client = None
            path = self.fileURL
            model = ConvNet2()
            model.load_state_dict(torch.load(path))
            model.eval()
            print("Model loaded")
            #client = None
            client = self.connect_mqtt()
            print("Mqtt connect")
        
        data = board.get_board_data()
        sequence = STIMULIT_SEQUENCE
        sound_array, fs_array = self.getSound()
        #Start
        logging.info(f"Experiment order: {sequence}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            for trials in range(NUM_TRIAL):
                print(f'{"Trials :" + str(trials+1)}')
                #voice left and right and rest
                #rest 2 sec
                self.playSound(sound_array[4],fs_array[4],1,board,3.0)
                #cue and Imagine 6 sec
                self.playSound(sound_array[sequence[trials]],fs_array[sequence[trials]],6,board,BLOCK_MARKER[sequence[trials]])
                
                if info[1] == True:
                    self.online_predict(board,info,client,model)
        
        
        if info[1] == True:
            IS_FINISH = True
        return IS_FINISH
    
    def closeEx(self):
        self.w = MainWindow()
        self.w.show()
        self.close()
    
    def playSound(self,sound,fs,second,board,marker):
        start = time.time()
        sd.play(sound, fs)
        sd.wait()
        print(f"Marker: {marker}")
        board.insert_marker(marker)
        core.wait(second)
        stop  = time.time()
        print(f"Trigger time = {(stop-start)}  Second ")

    def publish(self,client,output):
        msg = f"{output}"
        # main function to send
        result = client.publish(TOPIC, output)
        # result: [0, 1]
        status = result[0]
        if status == 0:
            print(f"Send `{msg}` to topic `{TOPIC}`")
        else:
            print(f"Failed to send message to topic {TOPIC}")

    def online_predict(self,board_shim,info,client,model):           
        #Get data from board
        data = board_shim.get_board_data()
        data_copy = data.copy()
        #sending and predict
        raw = getdata(data_copy,info[0],n_samples = 250)
        X_t,y_train = raw_preprocess(raw)
        X_t = apply_baseline(X_t) 
        #raw=raw.notch_filter([50])
        #raw.filter(8,14, method='fir', verbose=20)

        logging.info('--------------------------------------------------')
        #train_epochs,epochs_raw_data,labels = getepoch(raw,-3,5)
        logging.info('-------------------------------------------------')
        logging.info('Sending')
        #X_t,y_train = train_epochs.copy(),labels
                    

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
        self.publish(client,output)
        core.wait(33)
        #core.wait(5)
        throw = board_shim.get_board_data() 
        #Back to rest stage?
        #wait until finish rest stage       

    def connect_mqtt(self):
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

    # method for widgets
    def UiComponents(self):
        # Load the background image using QPixmap
        background_image = QPixmap("img/Ui.png")
        # Create a QLabel to display the background image
        self.background_label = QLabel(self)
        self.background_label.setPixmap(background_image)
        
        label = QLabel("Finish",self)
        font = label.font()
        font.setPointSize(40)
        label.setFont(font)
        #label.setFixedSize(300, 150)
        label.setGeometry(280, 100, 170, 130)
        
        
        btn = QPushButton("Exit",self)
        btn.setGeometry(290, 210, 150, 50)
        btn.pressed.connect(self.closeEx)

        
    def start(self):
        #Brainflow setting
        BoardShim.enable_dev_board_logger()
        #brainflow initialization 
        params = BrainFlowInputParams()
        #params.serial_port = SERIAL_PORT
        if self.info[0] == 0:
            #think pulse
            print("Think pulse activate")
            params.serial_port = SERIAL_PORT
            THINKPULSE_CONFIG = 'x1040111Xx2040111Xx3040111Xx4040111Xx5140010Xx6140010Xx7140010Xx8140010X'
            board_shim = BoardShim(self.info[0], params)
        else:
            board_shim = BoardShim(self.info[0], params)

        #board prepare
        try:
            board_shim.prepare_session()
            if self.info[0] == 0:
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
            IS_FINISH = self.startExperiment(board_shim,self.info)
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
        
    def __init__(self,info,fileURL=""):
        super().__init__()
               # setting title
        self.setWindowTitle("Experiment start")
  
        # setting geometry# 
        self.setFixedSize(QSize(720, 430))
        self.fileURL = fileURL
        self.info = info
        print(self.fileURL)
        self.UiComponents()

        self.show()
        
        self.start()
        

# stylesheet = """
#     MainWindow {
#         background-image: url("img/new_bg2.png"); 
#         background-repeat: no-repeat; 
#         background-position: center;
#     }
# """

app = QApplication(sys.argv)
#app.setStyleSheet(stylesheet)
app.setWindowIcon(QIcon("img/icon.png"))


window = MainWindow()
window.show()

app.exec()