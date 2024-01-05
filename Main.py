import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

df=pd.read_csv('C:/Users/HITESH SONI/OneDrive/Desktop/Python/Projects/Bharat Intern/Stock Prediction LSTM/NFLX.csv')

open_price=df['Open']
# print(open_price)

seq_len=15
mm=MinMaxScaler()
scaled_price=mm.fit_transform(np.array(open_price)[...,None]).squeeze()
# print(scaled_price)

X=[]
Y=[]

for i in range(len(scaled_price) - seq_len):
    X.append(scaled_price[i : i + seq_len])
    Y.append(scaled_price[i + seq_len])
    
# print(X[0])

X=np.array(X)[...,None]
Y=np.array(Y)[...,None]

train_X=torch.from_numpy(X[ : int(0.8 * X.shape[0])]).float()
train_Y=torch.from_numpy(Y[ : int(0.8 * Y.shape[0])]).float()
test_X=torch.from_numpy(X[int(0.8 * X.shape[0]) : ]).float()
test_Y=torch.from_numpy(Y[int(0.8 * X.shape[0]) : ]).float()

# print(train_X.shape,train_Y.shape)

class Model(nn.Module):
    def __init__(self , input_size , hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size , hidden_size , batch_first = True)
        self.fc = nn.Linear(hidden_size,1)
    def forward(self,x):
        output , (hidden , cell)= self.lstm(x)
        return self.fc(output[: , -1 , :])
        
model = Model(1 , 96)

optimizer = torch.optim.Adam(model.parameters() , lr = 0.001)
loss_fn = nn.MSELoss()

num_epochs = 100

for epoch in range(num_epochs):
    output = model(train_X)
    loss = loss_fn(output , train_Y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0 and epoch != 0:
        print(epoch , "epoch loss" , loss.detach().numpy())
        
model.eval()
with torch.no_grad():
    output = model(test_X)

pred = mm.inverse_transform(output.numpy())
real = mm.inverse_transform(test_Y.numpy())

plt.plot(pred.squeeze() , color = "red" , label = "predicted")
plt.plot(real.squeeze() , color = "green" , label = "real")
plt.show()
