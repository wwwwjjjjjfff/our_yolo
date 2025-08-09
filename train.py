import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(model=r"../ultralytics/cfg/models/11/DFF_MSCA.yaml")
    model.train(data="../mydata/danwan8_4.yaml", epochs=100, imgsz=[640, 640], batch=4, lr0=0.01,classes=0,)










