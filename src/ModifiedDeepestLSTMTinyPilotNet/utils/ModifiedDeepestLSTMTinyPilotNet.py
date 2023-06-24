import torch
import torch.nn as nn
import sys 
import os
sys.path.append(os.path.abspath("/home/meiqi/carla-examples/ModifiedDeepestLSTMTinyPilotNet/utils/"))

from convlstm import ConvLSTM
# from torchsummary import summary # only for debugging

class DeepestLSTMTinyPilotNet(nn.Module):
    def __init__(self, image_shape, num_labels):
        super(DeepestLSTMTinyPilotNet, self).__init__()
        self.num_channels = image_shape[2]
        self.cn_1 = nn.Conv2d(self.num_channels, 8, kernel_size=3, stride=2)
        self.relu_1 = nn.ReLU()
        self.cn_2 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
        self.relu_2 = nn.ReLU()
        self.cn_3 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
        self.relu_3 = nn.ReLU() 
        self.dropout_1 = nn.Dropout(0.2)

        self.clstm_n = ConvLSTM(8, 8, (5, 5), 3, batch_first=True, bias=True, return_all_layers=False)
        
        self.fc_1 = nn.Linear(8*35*24+1, 50)
        self.relu_fc_1 = nn.ReLU()
        self.fc_2 = nn.Linear(50, 10)
        self.relu_fc_2 = nn.ReLU()
        self.fc_3 = nn.Linear(10, num_labels)

    def forward(self, img, speed):        
        out = self.cn_1(img)
        out = self.relu_1(out)
        out = self.cn_2(out)
        out = self.relu_2(out)
        out = self.cn_3(out)
        out = self.relu_3(out)
        out = self.dropout_1(out)
        # add additional dimension at 1
        out = out.unsqueeze(1) 

        _, last_states = self.clstm_n(out)
        out =  last_states[0][0]  # 0 for layer index, 0 for h index

        # flatten & concatenate with speed
        out = out.reshape(out.size(0), -1)
        speed = speed.view(speed.size(0), -1)
        out = torch.cat((out, speed), dim=1)
        
        out = self.fc_1(out)
        out = self.relu_fc_1(out)
        out = self.fc_2(out)
        out = self.relu_fc_2(out)
        out = self.fc_3(out)

        out = torch.sigmoid(out)

        return out

if __name__ == '__main__':
    print("Modified Deepest LSTM Tiny PilotNet")
    model = DeepestLSTMTinyPilotNet((288, 200, 6), 3)
    # summary(model, (3, 100, 50))
    input = torch.rand((1, 6, 288, 200))
    speed = torch.rand((1, ))
    out = model(input, speed)
    #print(model)
    print(out)