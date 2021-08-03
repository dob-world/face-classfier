import torch
from torch.utils.data import DataLoader

from dataset import ImageDataset, transf, noise_transf
from model import CNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_size = 64

celeba_dir = '../../dataset/celeba/img_align_celeba'
dataset = ImageDataset(celeba_dir,
                       transform=transf,
                       noise_transform=noise_transf,
                       length=10,
                       )
data_loader = DataLoader(dataset, batch_size=10)


model = CNN().to(device)
model.load_state_dict(torch.load('checkpoint/model.pt'))
model.eval()

# 학습을 진행하지 않을 것이므로 torch.no_grad()
with torch.no_grad():
    for x, y in data_loader:
        y = y.to(device)
        x = x.to(device)
        prediction = model(x)
        a = torch.sub(prediction, y).abs()
        print(a)
        
