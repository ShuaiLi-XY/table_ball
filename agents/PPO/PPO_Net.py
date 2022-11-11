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
            torch.nn.AvgPool2d(kernel_size=2),
        )
        self.fc=torch.nn.Sequential(
                torch.nn.Linear(512*2*2,1024),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(1024,512),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(512,action_dim)

        )

    # 第一个动作范围是-100,200
    # 第二个动作范围是-30，30
    def forward(self, x):
        x=self.conv(x)
        x=torch.flatten(x,start_dim=1)
        x=self.fc(x)
        x=torch.tanh(x)
        m = torch.tensor([50, 0]).to(device)
        t = torch.tensor([150, 50]).to(device)
        return x*t+m


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

if __name__ == '__main__':
    device=torch.device("cuda")
    s=PolicyNet(2)
    s.to(device)
    # 第一个动作范围是-100,200
    # 第二个动作范围是-30，30
    x=torch.randint(high=100,low=0,size=(1280,1,40,40)).to(device)
    x=x.float()

    print(x)

    print(s(x))
