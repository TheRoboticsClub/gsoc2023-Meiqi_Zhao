import torch
import torch.nn as nn
import sys 
import os
sys.path.append(os.path.dirname(__file__))

from convlstm import ConvLSTM


class PilotNetOneHot(nn.Module):
    def __init__(self, image_shape, num_labels, num_hlc, num_light):
        super(PilotNetOneHot, self).__init__()
        self.num_channels = image_shape[2]
        self.cn_1 = nn.Conv2d(self.num_channels, 8, kernel_size=3, stride=2)
        self.relu_1 = nn.ReLU()
        self.cn_2 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
        self.relu_2 = nn.ReLU()
        self.cn_3 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
        self.relu_3 = nn.ReLU() 
        self.dropout_1 = nn.Dropout(0.2)

        self.clstm_n = ConvLSTM(8, 8, (5, 5), 3, batch_first=True, bias=True, return_all_layers=False)
        
        self.fc_1 = nn.Linear(8*35*24+1+num_hlc+num_light, 50)  # add embedding layer output size 
        self.relu_fc_1 = nn.ReLU()
        self.fc_2 = nn.Linear(50, 10)
        self.relu_fc_2 = nn.ReLU()
        self.fc_3 = nn.Linear(10, num_labels)

    def forward(self, img, speed, hlc, light):
        out = self.cn_1(img)
        out = self.relu_1(out)
        out = self.cn_2(out)
        out = self.relu_2(out)
        out = self.cn_3(out)
        out = self.relu_3(out)
        out = self.dropout_1(out)
        out = out.unsqueeze(1)  # add additional dimension at 1

        _, last_states = self.clstm_n(out)
        out =  last_states[0][0]  # 0 for layer index, 0 for h index

        # flatten & concatenate with speed
        out = out.reshape(out.size(0), -1)
        speed = speed.view(speed.size(0), -1)
        hlc = hlc.view(hlc.size(0), -1) 
        light = light.view(light.size(0), -1)  
        out = torch.cat((out, speed, hlc, light), dim=1)  # Concatenate the high-level commands to the rest of the inputs
        
        out = self.fc_1(out)
        out = self.relu_fc_1(out)
        out = self.fc_2(out)
        out = self.relu_fc_2(out)
        out = self.fc_3(out)

        out = torch.sigmoid(out)

        return out
    
class PilotNetOneHotEnhanced(nn.Module):
    def __init__(self, image_shape, num_labels, num_hlc, num_light):
        super(PilotNetOneHotEnhanced, self).__init__()
        self.num_channels = image_shape[2]
        self.cn_1 = nn.Conv2d(self.num_channels, 16, kernel_size=3, stride=2)
        self.relu_1 = nn.ReLU()
        self.cn_2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.relu_2 = nn.ReLU()
        self.cn_3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.relu_3 = nn.ReLU() 
        self.cn_4 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.relu_4 = nn.ReLU()
        self.dropout_1 = nn.Dropout(0.25)

        self.clstm_n = ConvLSTM(64, 32, (5, 5), 3, batch_first=True, bias=True, return_all_layers=False)
        
        self.fc_1 = nn.Linear(5993, 100)
        self.relu_fc_1 = nn.ReLU()
        self.dropout_2 = nn.Dropout(0.5)
        self.fc_2 = nn.Linear(100, 50)
        self.relu_fc_2 = nn.ReLU()
        self.fc_3 = nn.Linear(50, num_labels)

    def forward(self, img, speed, hlc, light):
        out = self.cn_1(img)
        out = self.relu_1(out)
        out = self.cn_2(out)
        out = self.relu_2(out)
        out = self.cn_3(out)
        out = self.relu_3(out)
        out = self.cn_4(out)
        out = self.relu_4(out)
        out = self.dropout_1(out)
        out = out.unsqueeze(1)  # add additional dimension at 1

        _, last_states = self.clstm_n(out)
        out =  last_states[0][0]  # 0 for layer index, 0 for h index

        # flatten & concatenate with speed
        out = out.reshape(out.size(0), -1)
        speed = speed.view(speed.size(0), -1)
        hlc = hlc.view(hlc.size(0), -1) 
        light = light.view(light.size(0), -1)  
        out = torch.cat((out, speed, hlc, light), dim=1)
        
        out = self.fc_1(out)
        out = self.relu_fc_1(out)
        out = self.dropout_2(out)
        out = self.fc_2(out)
        out = self.relu_fc_2(out)
        out = self.fc_3(out)

        out = torch.sigmoid(out)

        return out

class PilotNetOneHotNoLight(nn.Module):
    def __init__(self, image_shape, num_labels, num_hlc):
        super(PilotNetOneHotNoLight, self).__init__()
        self.num_channels = image_shape[2]
        self.cn_1 = nn.Conv2d(self.num_channels, 8, kernel_size=3, stride=2)
        self.relu_1 = nn.ReLU()
        self.cn_2 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
        self.relu_2 = nn.ReLU()
        self.cn_3 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
        self.relu_3 = nn.ReLU() 
        self.dropout_1 = nn.Dropout(0.2)

        self.clstm_n = ConvLSTM(8, 8, (5, 5), 3, batch_first=True, bias=True, return_all_layers=False)
        
        self.fc_1 = nn.Linear(8*35*24+1+num_hlc, 50)  # add embedding layer output size 
        self.relu_fc_1 = nn.ReLU()
        self.fc_2 = nn.Linear(50, 10)
        self.relu_fc_2 = nn.ReLU()
        self.fc_3 = nn.Linear(10, num_labels)

    def forward(self, img, speed, hlc):
        out = self.cn_1(img)
        out = self.relu_1(out)
        out = self.cn_2(out)
        out = self.relu_2(out)
        out = self.cn_3(out)
        out = self.relu_3(out)
        out = self.dropout_1(out)
        out = out.unsqueeze(1)  # add additional dimension at 1

        _, last_states = self.clstm_n(out)
        out =  last_states[0][0]  # 0 for layer index, 0 for h index

        # flatten & concatenate with speed
        out = out.reshape(out.size(0), -1)
        speed = speed.view(speed.size(0), -1)
        hlc = hlc.view(hlc.size(0), -1) 
        out = torch.cat((out, speed, hlc), dim=1)  # Concatenate the high-level commands to the rest of the inputs
        out = self.fc_1(out)
        out = self.relu_fc_1(out)
        out = self.fc_2(out)
        out = self.relu_fc_2(out)
        out = self.fc_3(out)

        out = torch.sigmoid(out)

        return out
    

class PilotNetEmbeddingNoLight(nn.Module):
    def __init__(self, image_shape, num_labels, num_hlc):
        super(PilotNetEmbeddingNoLight, self).__init__()
        self.num_channels = image_shape[2]
        self.cn_1 = nn.Conv2d(self.num_channels, 8, kernel_size=3, stride=2)
        self.relu_1 = nn.ReLU()
        self.cn_2 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
        self.relu_2 = nn.ReLU()
        self.cn_3 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
        self.relu_3 = nn.ReLU() 
        self.dropout_1 = nn.Dropout(0.2)

        self.clstm_n = ConvLSTM(8, 8, (5, 5), 3, batch_first=True, bias=True, return_all_layers=False)
        
        # Add embedding layer for high-level commands
        self.embedding = nn.Embedding(num_hlc, 5)
        
        self.fc_1 = nn.Linear(8*35*24+1+5, 50)  # add embedding layer output size 
        self.relu_fc_1 = nn.ReLU()
        self.fc_2 = nn.Linear(50, 10)
        self.relu_fc_2 = nn.ReLU()
        self.fc_3 = nn.Linear(10, num_labels)

    def forward(self, img, speed, hlc):
        out = self.cn_1(img)
        out = self.relu_1(out)
        out = self.cn_2(out)
        out = self.relu_2(out)
        out = self.cn_3(out)
        out = self.relu_3(out)
        out = self.dropout_1(out)
        out = out.unsqueeze(1)  # add additional dimension at 1

        _, last_states = self.clstm_n(out)
        out =  last_states[0][0]  # 0 for layer index, 0 for h index

        # flatten & concatenate with speed
        out = out.reshape(out.size(0), -1)
        speed = speed.view(speed.size(0), -1)
        hlc = self.embedding(hlc).view(hlc.size(0), -1)  # convert hlc to dense vectors using the embedding layer
        out = torch.cat((out, speed, hlc), dim=1)  # Concatenate the high-level commands to the rest of the inputs
        
        out = self.fc_1(out)
        out = self.relu_fc_1(out)
        out = self.fc_2(out)
        out = self.relu_fc_2(out)
        out = self.fc_3(out)

        out = torch.sigmoid(out)

        return out

if __name__ == '__main__':
    print("Modified Deepest LSTM Tiny PilotNet")
    model = PilotNetOneHot((288, 200, 6), 3, 4, 4)
    # summary(model, (3, 100, 50))
    input = torch.rand((1, 6, 288, 200))
    speed = torch.rand((1, ))
    hlc = torch.zeros(1, 4)
    hlc[0] = 1.0
    light = torch.zeros(1, 4)
    light[0] = 1.0
    out = model(input, speed, hlc, light)
    #print(model)
    print(out)