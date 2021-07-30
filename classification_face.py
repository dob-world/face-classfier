import os

import torch
from torch.utils.data import DataLoader

from dataset import ImageDataset, transf, noise_transf
from model import CNN

# 결과 저장 폴더 지정
def init(result_path, face_path, no_face_path ):
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    if not os.path.isdir(face_path):
        os.mkdir(face_path)
    if not os.path.isdir(no_face_path):
        os.mkdir(no_face_path)

def cp_files(filenames, path):
    for f in filenames:
        save_path = os.path.join(path, f.split('\\')[-1])
        os.system(f"cp {f} {save_path}")


    

# 변수 선언
device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_size = 64
limit = 0.9
iter_cnt = 5

# 데이터 불러오기
data_dir = '../data'
dataset = ImageDataset(data_dir,
                       transform=transf,
                       noise_transform=noise_transf,
                       is_predict=True,
                       length=10000
                       )
data_loader = DataLoader(dataset, batch_size=100)

# 모델 불러오기
model = CNN().to(device)
model.load_state_dict(torch.load('checkpoint/model.pt'))
model.eval()


if __name__ == "__main__":
    result_path = "results"
    face_path = "results/face"
    no_face_path = "results/no-face"

    init(result_path, face_path, no_face_path)

    # 학습을 진행하지 않을 것이므로 torch.no_grad()

    faces = []
    no_faces= []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            prediction = model(x)
            for j in range(iter_cnt):
                prediction = torch.add(prediction, model(x))
            prediction = prediction / (iter_cnt+1)
            for idx, i in enumerate(prediction):
                if i > limit:
                    faces.append(y[idx])
                else:
                    no_faces.append(y[idx])
        print(f"faces count: {len(faces)}")
        print(f"no-faces count: {len(no_faces)}")
        cp_files(faces, face_path)
        cp_files(no_faces, no_face_path)
    