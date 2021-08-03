import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import ImageDataset, transf, noise_transf
from model import CNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)

learning_rate = 0.001
training_epochs = 15

celeba_dir = '../../dataset/celeba/img_align_celeba'
dataset = ImageDataset(celeba_dir,
                       transform=transf,
                       noise_transform=noise_transf)

data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
total_batch = len(data_loader)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# CNN 모델 정의
model = CNN().to(device)
model.apply(weights_init)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:
        X = X.to(device)
        Y = Y.to(device)
        optimizer.zero_grad()
        output = model(X)
        cost = criterion(output, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

torch.save(model.state_dict(), 'checkpoint/model.pt')
