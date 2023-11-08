# Convolutional neural network (two convolutional layers)
from torch import nn
import torch
import math
in_channels = 2  # C3, C4
out_channels = (
    64  # five is logical because we have freq= 8, 9, 10, 11, 12 that we want to capture
)
out_size = 2  # left or right
kernel_size = 5
linear_shape = 41024


# Convolutional neural network (two convolutional layers)
# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # using sequential helps bind multiple operations together
        self.layer1 = nn.Sequential(
            # padding = (kernel_size - 1) / 2 = 2
            nn.Conv1d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        # 144064
        # 96064
        # 64064
        # 128064
        # 80000
        #112064
        #112000
        self.fc = nn.Linear(112000 , 2)
        self.drop_out = nn.Dropout(0.5)

    def forward(self, x):
        out = self.layer1(x)
        out = self.drop_out(out)
        out = self.layer2(out)
        out = self.drop_out(out)
        out = self.layer3(out)
        # out = self.drop_out(out)
        # out = self.layer4(out)
        # out = self.drop_out(out)

        out = out.reshape(out.size(0), -1)  # can also use .view()
        out = self.fc(out)
        return out

class ConvNet2(nn.Module):
    def __init__(self):
        super(ConvNet2, self).__init__()
        
        #using sequential helps bind multiple operations together
        self.layer1 = nn.Sequential(
            #padding = (kernel_size - 1) / 2 = 2
            nn.Conv1d(3, 32, kernel_size=10, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, stride=3, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        
        self.fc = nn.Linear(7488 , 2)

        self.drop_out = nn.Dropout(0.5)

        
    def forward(self, x):
        out = self.layer1(x)
        out = self.drop_out(out)

        out = self.layer2(out)
        out = self.drop_out(out)
        
        out = self.layer3(out)
        out = self.drop_out(out)
        # out = self.layer4(out)
        # out = self.drop_out(out)

        out = out.reshape(out.size(0), -1)   #can also use .view()
        out = self.fc(out)

        return out
class ConvNet_physionet(nn.Module):
    def __init__(self):
        super(ConvNet_physionet, self).__init__()

        # using sequential helps bind multiple operations together
        self.layer1 = nn.Sequential(
            # padding = (kernel_size - 1) / 2 = 2
            nn.Conv1d(2, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.fc = nn.Linear(41024, 2)
        self.drop_out = nn.Dropout(0.5)

    def forward(self, x):
        out = self.layer1(x)
        out = self.drop_out(out)
        out = self.layer2(out)
        out = self.drop_out(out)
        out = self.layer3(out)
        # out = self.drop_out(out)
        # out = self.layer4(out)
        # out = self.drop_out(out)

        out = out.reshape(out.size(0), -1)  # can also use .view()
        out = self.fc(out)
        return out
# Convolutional neural network (two convolutional layers)
class CNN2D(nn.Module):
    def __init__(self):
        super(CNN2D, self).__init__()

        # using sequential helps bind multiple operations together
        self.layer1 = nn.Sequential(
            # padding = (kernel_size - 1) / 2 = 2
            nn.Conv1d(2, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(82048, 2)
        self.drop_out = nn.Dropout(0.5)

    def forward(self, x):
        out = self.layer1(x)
        out = self.drop_out(out)
        out = self.layer2(out)
        out = self.drop_out(out)
        out = out.reshape(out.size(0), -1)  # can also use .view()
        out = self.fc(out)
        return out

class gamenet(nn.Module):
    def __init__(self):
        super(gamenet, self).__init__()
        self.l1 = nn.Sequential(
            # in_channel = 16
            # out_channel or Filter size = 100
            # kernel size = (1,25)
            # stride = 1
            # padding = Same
            # Relu
            nn.Conv1d(2, 100, kernel_size=2, stride=1, padding='same'),
            nn.BatchNorm1d(100),
            nn.ReLU(),
        )
        self.l2 = nn.Sequential(
            # in_channel = 100
            # out_channel or Filter size = 100
            # kernel size = (16,1)
            # stride = 1
            # padding = Valid
            # Relu
            nn.Conv1d(100, 100, kernel_size=2, stride=1, padding="valid"),
            # nn.Conv2d(100,100,kernel_size=(64,1),stride=1,padding='valid'),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            
        )
        self.l3 = nn.Sequential(
            # in_channel = 100
            # out_channel = 50
            # kernel size = (1,30)
            # stride = 1
            # padding = Same
            # Relu
            nn.Conv1d(100, 50, kernel_size=30, stride=1, padding="same"),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            
        )
        self.maxpooling1 = nn.MaxPool1d(kernel_size=7, stride=5)
        self.l4 = nn.Sequential(
            # in_channel = 50
            # out_channel = 50
            # kernel size = (1,30)
            # stride = 1
            # padding = Same
            # Relu
            nn.Conv1d(50, 50, kernel_size=30, stride=1, padding="same"),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            
        )
        self.maxpooling2 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.flatten = nn.Sequential(
            nn.Flatten(), nn.BatchNorm1d(9950), nn.Dropout(0.15)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(9950, 1024), nn.ReLU(), nn.BatchNorm1d(1024), nn.Dropout(0.15)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.15)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.15)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.15)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.15)
        )
        self.fc5 = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.15)
        )
        self.fc6 = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(0.15)
        )
        self.softmax = nn.Sequential(nn.Linear(32, 2), nn.Softmax())

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.maxpooling1(out)
        out = self.l4(out)
        out = self.maxpooling2(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        out = self.fc6(out)
        out = self.softmax(out)
        return out


