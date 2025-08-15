# code/train.py

import os
import sys
import subprocess
from ultralytics import YOLO

# 安装 ultralytics（建议放在 requirements.txt 里，此处备用）
# subprocess.call(['pip', 'install', 'ultralytics'])

print("wuming test train job start")

# 预训练模型路径
pretrained_model_path = "/opt/ml/input/data/pretrained/yolo11n.pt"
data_yaml_path = "/opt/ml/input/data/train/data.yaml"

# 加载模型
print(f"正在加载模型：{pretrained_model_path}")
model = YOLO(pretrained_model_path)

# 训练
model.train(
    data=data_yaml_path,
    batch=8,
    workers=8,
    epochs=100,
    imgsz=640,
    device='cuda:0',
    #task='segment',
    close_mosaic=600,
    project="/opt/ml/model",
    overlap_mask=False,
    single_cls=True,
    mask_ratio=1,
    box=7.5,
    cls=0.1,
    dfl=4.5
)
