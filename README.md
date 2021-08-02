# face classfier

여러 장의 이미지를 읽어들여, 얼굴과 아닌 것을 분류하여 저장합니다.

## Environment

제 실행환경은 아래와 같습니다.

- python 3.9

- pytorch 1.9
- cuda 11.1

## How to Run

```
git clone https://github.com/wo7864/face-classfier
```

```
pip install -r requirements.txt
```

아래 링크를 클릭하여 모델을 다운로드하고, **checkpoint** 폴더를 생성하여 저장합니다.

[모델 다운로드](https://drive.google.com/file/d/1kX6034aTPtBUpirtAEfQ-7FX_J0ztjD6/view?usp=sharing)

```
python classfication_face.py --options...
```

- data_dir: 불러올 데이터의 디렉토리 경로를 지정합니다.
  `Default: data`
- model_dir: 불러올 모델의 경로를 지정합니다.
  `Default: checkpoint`
- face_dir:  [face]로 분류된 이미지의 저장 경로를 지정합니다.
  `Default: face`
- no_face_dir: [no-face]로 분류된 이미지의 저장 경로를 지정합니다.
  `Default: no-face`
- limit: [face]로 분류할 정확도의 하한선을 지정합니다.
  `Default: 0.9`



