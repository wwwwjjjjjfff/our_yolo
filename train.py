import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

# # Load a model
# model = YOLO("yolo11n.yaml")  # build a new model from YAML
# model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights


if __name__ == '__main__':
    from ultralytics import YOLO
    from ultralytics import YOLO

    # model = YOLO(model=r"C:\wjf\yolov11\ultralytics\mydata\cfg\yolo11.yaml").load("./yolo11n.pt")
    # model = YOLO(model=r"../ultralytics/cfg/models/11/yolo11-Small-LDConv-MSCA.yaml").load("./yolo11n.pt")
    model = YOLO(model=r"../ultralytics/cfg/models/v5/0.yaml")
    model.train(data="../mydata/danwan8_4.yaml", epochs=100, imgsz=[640, 640], batch=2, lr0=0.01,classes=0,)

    # model = YOLO(r"D:\RJH\YOLO\ultralytics-8.3.131\ultralytics-8.3.131\ultralytics\cfg\models\11\yolo11small.yaml").load("yolo11n.pt")
    # model.train(data="D:/RJH/YOLO/ultralytics-8.3.131/ultralytics-8.3.131/data/popian4.yaml",
    #             cache=False,
    #             imgsz=640,
    #             epochs=300,
    #             single_cls=True,  # 是否是单类别检测
    #             batch=8,
    #             close_mosaic=10,
    #             workers=0,
    #             device='cuda',
    #             optimizer='SGD',
    #             amp=True,
    #             project='D:/RJH/YOLO/ultralytics-8.3.131/ultralytics-8.3.131/runs/train',
    #             name='exp',
    #             lr0=0.001,
    #             )


# from ultralytics import YOLO
#
# # Load a model
# model = YOLO("C:/wjf/yolov11/ultralytics/runs/train/exp/weights/last.pt")  # load a partially trained model
#
# # Resume training
# results = model.train(resume=True)







