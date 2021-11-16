### Environment
- 참고
  - https://teddylee777.github.io/colab/tensorflow-gpu-install-windows
- Packages
  - python
    - 3.7.6
  - tensorflow-gpu
    - 2.6.0
  - CUDA
    - cuDNN과 버전을 맞춰야함
    - 11.0
    - cuda_11.0.2_win10_network.exe
    - https://developer.nvidia.com/cuda-toolkit-archive
    - 설치후 cusolver64_10.dll을 복사해서 cusolver64_11.dll로 이름 변경
  - cuDNN
    - 8.1.0 for CUDA 11.0,11.1 and 11.2
    - cudnn-11.2-windows-x64-v8.1.0.77.zip
    - https://developer.nvidia.com/rdp/cudnn-archive
- VSCODE extension
  - Python
  - Python for VSCode
  - Python Extension Pack
  - ipykernel
- VENV
  - python 버전 지정 설치
    - virtualenv venv --python=python2.7
- pip install gin-config lvis 
### object detection API
- https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html
- 과정
  - COCO API installation
    - pip install pycocotools-windows
    - python -m pip install --use-feature=2020-resolver .
      - no such option: --use-feature가 나올경우
        - python -m pip install --upgrade pip
      - pycocotools-windows를 이전에 설치했으면 해당 에러는 무시
    - PYTHON_PATH에 아래 경로들 추가
      - E:\Project\DeepLearning_FastCampus\models
      - E:\Project\DeepLearning_FastCampus\models\research
      - E:\Project\DeepLearning_FastCampus\models\research\slim
- 학습
  - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_training_and_evaluation.md
  - 
### WSL2
- windows insider로 업데이트
- WSL2 Linux 커널 업데이트 패키지 설치
  - https://docs.microsoft.com/ko-kr/windows/wsl/install-manual#step-4---download-the-linux-kernel-update-package
- 기존 Ubuntu 업데이트
```
  wsl -l -v   # 기존 vm 확인    
  wsl --set-version Ubuntu-18-04 2   # 기존 vm 업데이트    
```
- 드라이브문자로 할당
- https://www.lesstif.com/software-architect/wsl-2-windows-subsystem-for-linux-2-89555812.html
  
### Datasets
 - 한국어 글자체 이미지 : https://aihub.or.kr/aidata/133
 - http://101.101.175.217:8080/static/aiocr/learning
 - CarPlate : https://www.kaggle.com/andrewmvd/car-plate-detection

### jupyter
- ipynb를 py로 변환
 - jupyter nbconvert --to script [filename].ipynb 
