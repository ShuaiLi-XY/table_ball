### 两个agent共享参数
### 神经网络输入是agent的可观测的地图部分

import torch
import torch.nn.functional as F
from torch.distributions import Beta, Normal



#使用beta分布取代高斯分布
class PolicyNet(torch.nn.Module):
    def __init__(self,action_dim):
        super(Actor_Beta, self).__init__()
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
            torch.nn.AvgPool2d(kernel_size=2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(512 * 2 * 2, 1024),
            torch.nn.Tanh(),
            torch.nn.Linear(1024, 512),
            torch.nn.Tanh(),
        )
        self.device = torch.device("cuda")
        self.alpha_layer = torch.nn.Linear(512, action_dim)
        self.beta_layer = torch.nn.Linear(512, action_dim)

    def forward(self, s):
        s=self.conv(s)
        s=torch.flatten(s,start_dim=1)
        s=self.fc(s)
        # alpha and beta need to be larger than 1,so we use 'softplus' as the activation function and then plus 1
        alpha = F.softplus(self.alpha_layer(s)) + 1.0
        beta = F.softplus(self.beta_layer(s)) + 1.0
        dist = Beta(alpha, beta)
        a = dist.sample()
        m = torch.tensor([50, 0]).to(self.device)
        t = torch.tensor([150, 30]).to(self.device)
        return a*t+m

    def mean(self, s):
        alpha, beta = self.forward(s)
        mean = alpha / (alpha + beta)  # The mean of the beta distribution
        return mean



class ValueNet(torch.nn.Module):
    def __init__(self):
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
    device = torch.device("cuda")
    s = Actor_Beta(2)
    s.to(device)
    # 第一个动作范围是-100,200
    # 第二个动作范围是-30，30
    x = torch.randint(high=100, low=0, size=(128, 1, 40, 40)).to(device)
    x = x.float()
    print(x)
    print(s(x))
