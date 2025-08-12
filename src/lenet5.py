import torch
from torch import nn

def create_LeNet5_model(device, channels, state=None):
    net = Lenet5(channels)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)   # to use the multiple GPUs 
    net = net.to(device)
    if state:
        net.load_state_dict(state)
    return net

    
# LeNet5 Model: Once defined in the server, it is sent to all clients for their local training 
class Lenet5(nn.Module):
    def __init__(self, channels):
        super(Lenet5, self).__init__()
        self.features = nn.Sequential  (
                nn.Conv2d(channels, 6, kernel_size = 5),
                nn.ReLU (inplace = True),
                nn.MaxPool2d(kernel_size = 2, stride = 1),
                #----------------------------------------------------- 1
                nn.Conv2d(6, 16, kernel_size = 5),
                nn.ReLU (inplace = True),
                nn.MaxPool2d(kernel_size = 2, stride = 1),
              )
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.classifier = nn.Sequential  (
                nn.Linear (16*5*5, 120),
                nn.ReLU (inplace =  True),
                nn.Linear(120, 84),
                nn.Linear(84,10),
                nn.LogSoftmax(dim = 1), #10 classes
            )
        
    def forward(self, x):
        cx = self.features(x)
        conv_x = self.avgpool(cx)
        x_flat = conv_x.view(conv_x.size(0), -1)
        y_hat = self.classifier(x_flat)
        return y_hat