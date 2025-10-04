import torch
from PIL import Image

model = torch.hub.load("ultralytics/yolov5", "custom", path="runs/train/exp5/weights/best.pt")  # local model
img=Image.open(r"D:\datasets\judge_head\images\train2017\000000000016.jpg")

results = model(img)
print(results.render())
print(results.ims[0])
results.show()