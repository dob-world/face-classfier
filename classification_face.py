import os
import argparse

import torch
from torch.utils.data import DataLoader

from dataset import ImageDataset, transf, noise_transf
from model import CNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 결과 저장 폴더 지정
def init( face_path, no_face_path ):
    if not os.path.isdir(face_path):
        os.mkdir(face_path)
    if not os.path.isdir(no_face_path):
        os.mkdir(no_face_path)

def cp_files(filenames, path):
    for f in filenames:
        save_path = os.path.join(path, f.split('\\')[-1])
        os.system(f"cp {f} {save_path}")


    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--data_dir', type=str, default="data", help='Output of images classified as faces')
    parser.add_argument('--model_dir', type=str, default="checkpoint", help='Output of images classified as faces')
    parser.add_argument('--face_dir', type=str, default="face", help='Output of images classified as faces')
    parser.add_argument('--no_face_dir', type=str, default="no-face", help='Output of images classified as no-faces')
    parser.add_argument('--limit', type=float, default=0.9, help='Output of images classified as faces')

    args = parser.parse_args()
    init(args.face_dir, args.no_face_dir)

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


    # 학습을 진행하지 않을 것이므로 torch.no_grad()

    with torch.no_grad():
        for x, y in data_loader:
            faces = []
            no_faces= []
            x = x.to(device)
            prediction = model(x)
            for idx, i in enumerate(prediction):
                if i > args.limit:
                    faces.append(y[idx])
                else:
                    no_faces.append(y[idx])
            cp_files(faces, args.face_dir)
            cp_files(no_faces, args.no_face_dir)
    