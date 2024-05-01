import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt


if torch.cuda.is_available():
    device=torch.device("cuda")
else:
    device=torch.device("cpu")
 
mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
 
transforms_fn=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std)
])
 
train_data=torchvision.datasets.CIFAR100('cifar',train=True,transform=transforms_fn,download=False)

test_data=torchvision.datasets.CIFAR100('cifar',train=False,transform=transforms_fn,download=False)

#class_indices = list(range(0, 100)) #定义要抽取的类别序号
#train_indices = [i for i in range(len(train_data)) if train_data.targets[i] in class_indices]
#train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices) #创建自定义的采样器，仅选择包含所选类别的样本
 
#test_indices = [i for i in range(len(test_data)) if test_data.targets[i] in class_indices]
#test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)
 

#train= DataLoader(train_data, batch_size=64, sampler=train_sampler) #在下面的源码解释中可以看出，如果sampler不为默认的None的时候，不用设置shuffle属性了
#test = DataLoader(test_data, batch_size=64, sampler=test_sampler)
batchSize=1024
learning_rate=0.0005

train_loader = torch.utils.data.DataLoader(train_data, 
                                           batch_size=batchSize, 
                                           shuffle=True,
                                           pin_memory=True,
                                           )

test_loader = torch.utils.data.DataLoader(test_data,
                                            batch_size=batchSize,
                                            shuffle=False,
                                            pin_memory=True,
                                            )

class BasicBlock(nn.Module):
    def __init__(self, in_channel, s):
        """
        基础模块, 共有两种形态, 1.s=1输入输出维度相同时 2.s=2特征图大小缩小一倍, 维度扩充一倍
        :param in_channel: 输入通道数维度
        :param s: s=1 不缩小 s=2 缩小尺度
        """
        super(BasicBlock, self).__init__()
        self.s = s
        self.conv1 = nn.Conv2d(in_channel, in_channel * s, kernel_size=3, stride=s, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channel * s)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channel * s, in_channel * s, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channel * s)
        if self.s == 2:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, in_channel * s, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(in_channel * s)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.s == 2:  # 缩小
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out
#构建RESNET18
class ResNet18(nn.Module):
    def __init__(self, n_class, zero_init_residual=True):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            BasicBlock(in_channel=64, s=1),
            BasicBlock(in_channel=64, s=1),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(in_channel=64, s=2),
            BasicBlock(in_channel=128, s=1),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(in_channel=128, s=2),
            BasicBlock(in_channel=256, s=1),
        )
        self.layer4 = nn.Sequential(
            BasicBlock(in_channel=256, s=2),
            BasicBlock(in_channel=512, s=1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, n_class)

        # 初始化参数 -> 影响准确率 7%
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # 初始化BasicBlock -> 影响准确率 1-2%
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
 
#网络模型
model=ResNet18(n_class=100)
model.to(device)
#损失函数
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(params=model.parameters(),lr=learning_rate, momentum=0.9)
#loss_fn.to(device)
 
 
 
 
train_acc_list = []
train_loss_list = []
test_acc_list = []
test_loss_list=[]
epochs=50
 
for epoch in range(epochs):
    print("-----第{}轮训练开始------".format(epoch + 1))
    train_loss=0.0
    test_loss=0.0
    train_sum,train_cor,test_sum,test_cor=0,0,0,0
 
    model.train()
    for batch_idx,(data,target) in enumerate(train_loader):
        data,target=data.to(device),target.to(device)
 
        optimizer.zero_grad() 
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step() 
 
        train_loss += loss.item()
 
        _, predicted = torch.max(output.data, 1)
        train_cor += (predicted == target).sum().item() 
        train_sum += target.size(0)
 
    model.eval()
    for batch_idx1,(data,target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
 
        output = model(data)
        loss = loss_fn(output, target)
        test_loss+=loss.item()
        _, predicted = torch.max(output.data, 1)
        test_cor += (predicted == target).sum().item()
        test_sum += target.size(0)
 
    print("Train loss:{}   Train accuracy:{}%   Test loss:{}   Test accuracy:{}%".format(train_loss/batch_idx,100*train_cor/train_sum,
                                                                                       test_loss/batch_idx1,100*test_cor/test_sum))
    train_loss_list.append(train_loss / batch_idx)
    train_acc_list.append(100 * train_cor / train_sum)
    test_acc_list.append(100 * test_cor/ test_sum)
    test_loss_list.append(test_loss / batch_idx1)

torch.save(model,"CIFAR100_epoch{}.pth".format(epochs))
 
 
 
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig=plt.figure()
plt.plot(range(len(train_loss_list)),train_loss_list,'blue')
plt.plot(range(len(test_loss_list)),test_loss_list,'red')
plt.legend(['训练损失','测试损失'],fontsize=14,loc='best')
plt.xlabel('训练轮数',fontsize=14)
plt.ylabel('损失值',fontsize=14)
plt.grid()
plt.show()
 
fig=plt.figure()
plt.plot(range(len(train_acc_list)),train_acc_list,'blue')
plt.plot(range(len(test_acc_list)),test_acc_list,'red')
plt.legend(['训练准确率','测试准确率'],fontsize=14,loc='best')
plt.xlabel('训练轮数',fontsize=14)
plt.ylabel('准确率(%)',fontsize=14)
plt.grid()
plt.show()

