import torch
import torch.nn as nn

# f = 2*x

# training set
X = torch.tensor([[1],[2],[3],[4]], dtype = torch.float32)
y = torch.tensor([[2],[4],[6],[8]], dtype = torch.float32)

# test set
x_test = torch.tensor([5], dtype = torch.float32)

# model 
n_samples, n_features = X.shape
input_size = n_features
output_size = n_features 
model = nn.Linear(input_size, output_size)

# Training
lr = 0.1
n_iter = 300
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = lr)

print(f'Prediction before training: {model(x_test).item():.3f}')

for epoch in range(n_iter):
    
    y_hat = model(X)

    l = loss(y,y_hat)

    l.backward()

    optimizer.step()
    
    optimizer.zero_grad()

    if epoch % 30 == 0:
        [w,b] = model.parameters()
        print(f'Epoch: {epoch+1}, weight: {w[0][0].item():.3f}, bias: {b.item():.3f}, loss: {l:.8f}')

print(f'Prediction after training: {model(x_test).item():.3f}')
