from torchvision.datasets import CIFAR10
import numpy as np
import cv2
import torch
from glob import glob
from torch.utils.data import Dataset
from PIL import Image

'''
Transforming cifar datasets into JPG format
'''
def generate_train_jpg():
    mymap={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
    data=CIFAR10(root="./",download=True)
    for i,label in data:
        cv2.imwrite("./dataset/train/"+str(label)+"_"+str(mymap[label])+".jpg",np.array(i))
        mymap[label]+=1

def generate_test_jpg():
    my_test_map={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
    data=CIFAR10(root="./",download=True,train=False)
    for i,label in data:
        cv2.imwrite("./dataset/test/"+str(label)+"_"+str(my_test_map[label])+".jpg",np.array(i))
        my_test_map[label]+=1

if __name__ == "__main__":
    generate_test_jpg()
    generate_train_jpg()



'''
pytorch dataloader
q dataloader is Quantitative Dataset selected in the test dataset
'''


class TrainDataset(Dataset):
    def __init__(self,transform):
        self.imagelist=glob("./dataset/train/*.jpg")
        self.len=len(self.imagelist)
        self.transform=transform
    def __getitem__(self, index):
        image_path = self.imagelist[index]
        label=int(image_path.split("_")[0][-1])
        image = Image.open(image_path)
        return self.transform(image),label

    def __len__(self):
        return self.len


class TestDataset(Dataset):
    def __init__(self,transform):
        self.imagelist=glob("./dataset/test/*.jpg")
        self.len=len(self.imagelist)
        self.transform=transform
    def __getitem__(self, index):
        image_path = self.imagelist[index]
        label=int(image_path.split("_")[0][-1])
        image = Image.open(image_path)
        return self.transform(image),label

    def __len__(self):
        return self.len


class QDataset(Dataset):
    def __init__(self,transform):
        self.imagelist=glob("./dataset/q/*.jpg")
        self.len=len(self.imagelist)
        self.transform=transform
    def __getitem__(self, index):
        image_path = self.imagelist[index]
        label=int(image_path.split("_")[0][-1])
        image = Image.open(image_path)
        return self.transform(image),label

    def __len__(self):
        return self.len