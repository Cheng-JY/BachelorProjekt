import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import os
import cv2
import json
import glob
from tqdm.notebook import tqdm
import requests

if __name__ == '__main__':
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    img = Image.open(requests.get(url, stream=True).raw)
    transformed_img = transform_image(img)[:3].unsqueeze(0)
    print(transformed_img.shape)

    dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

    embedding = dinov2_vits14(transformed_img)



