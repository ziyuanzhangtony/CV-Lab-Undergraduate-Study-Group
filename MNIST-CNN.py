import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

T = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])



train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True, transform=T),
    batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=T),
    batch_size=64, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output



model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()
    losses = []
    acces = []
    for itr, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        # print(loss.item())
        losses.append(loss.item())

        test_data, test_target = next(iter(test_loader))
        model.eval()
        output = model(test_data)
        _, pred = torch.max(output, dim=1)
        a = pred.eq(test_target)
        acc = a.sum().item()/len(a)
        acces.append(acc)

        if(itr>200):
            break
    plt.plot(acces)
    plt.show()


train()

# for data, target in train_loader:
#     plt.imshow(data[0,0,:,:])
#     plt.show()
#     print(target[0])
#     break


