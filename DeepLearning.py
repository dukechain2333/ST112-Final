import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import math
import matplotlib.pyplot as plt

df = pd.read_csv("Data/Phishing_BestFirst.csv", delimiter= ',').drop(labels=0)
print(df)
y = df.pop('class')
y = y.replace('phishing', 1).replace('benign', 0)
y = np.array(y)
y = y.tolist()
y = torch.unsqueeze(torch.FloatTensor(y), dim = 1)
print(y)
print(y.shape)

x = df
x = np.array(x)
x = x.tolist()
x = torch.squeeze(torch.FloatTensor(x), dim =1)
print(x)
print(x.shape)

class Model(nn.Module):
    #定义多层神经网络
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(13, 10)
        self.fc2 = torch.nn.Linear(10,7)
        self.fc3 = torch.nn.Linear(7,4)
        self.fc4 = torch.nn.Linear(4,1)
        
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p = 0.15) #dropout 1
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p = 0.15) #dropout2
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p = 0.15)
        y_pred = torch.sigmoid(self.fc4(x)) #sigmoid
        return y_pred

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        print('hi')
        m.weight.data = torch.randn(m.weight.data.size()[0], m.weight.data.size()[1])
        m.bias.data = torch.randn(m.bias.data.size()[0])

model = Model()
model.apply(weight_init)
criterion  = torch.nn.BCELoss() #定义损失函数 binary corsstropy
optimizer = torch.optim.SGD(model.parameters(),lr = 0.03) #学习率设置为0.01,学习率为超参数 ，可以自己设置
Loss = []
print(x.shape)

for epoch in range(20000):
    y_pred = model(x)
    #计算误差
    loss = criterion(y_pred,y)
    #
    #prin(loss.item())
    Loss.append(loss.item())
    #每迭代1000次打印Lost并记录
    '''if epoch%100 == 0:
        print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, 2000, loss.item()))'''
    #梯度清零
    optimizer.zero_grad()
    #反向传播
    loss.backward()
    #更新梯度
    optimizer.step()
for i in range(len(y_pred)):
    if(y_pred[i]>0.5):
        y_pred[i] = 1.0
    else:
        y_pred[i] = 0.0
#print(y_pred)
type(y_pred)

print((y_pred == y).sum().item()/len(y) )# torch.Tensor.sum()函数)
