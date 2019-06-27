import torch
from PIL import Image
from torchvision.transforms import ToTensor
import visdom
import os

file='tsne_true_image.png'
path='/home/xiaoxi.xjl/my_code/UDA/distribution/images'
img_file=os.path.join(path,file)
img=Image.open(img_file).convert('RGB')
img=ToTensor()(img)
vis=visdom.Visdom()
vis.image(img)
