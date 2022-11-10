### 两个agent共享参数
### 神经网络输入是agent的可观测的地图部分

import  torch

class PolicyNet(torch.nn.Module):
    def __init__(self,action_dim):
        super(PolicyNet, self).__init__()
        self.conv=torch.nn.Sequential(
            #输入40*40
            torch.nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,padding=1,stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=0, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AvgPool2d(kernel_size=2),
            #输入变成4*4
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(inplace=True),
        )
        self.fc=torch.nn.Sequential(
                torch.nn.Linear(512*4*4,1024),
                torch.nn.Linear(1024,512),
                torch.nn.Linear(512,action_dim)
        )


    def forward(self, x):
        x=self.conv(x)
        x=torch.flatten(x,start_dim=1)
        x=self.fc(x)
        return x


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.conv = torch.nn.Sequential(
            # 输入40*40
            torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=0, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AvgPool2d(kernel_size=2),
            # 输入变成4*4
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(inplace=True),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(512 * 4 * 4, 1024),
            torch.nn.Linear(1024, 512),
            torch.nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x