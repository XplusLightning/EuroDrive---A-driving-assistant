from torch.utils.data import Dataset
from PIL import Image
import torch
import config
import torchvision.transforms as transforms
import numpy as np
# from sklearn import preprocessing
# from cv2 import imread 
from mss import mss
from time import sleep


def readTxt(file_path):
    img_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            item = lines.strip().split()
            img_list.append(item)
    file_to_read.close()
    return img_list

def readNames(dictionary):
    names = []
    for i in dictionary:
        names.append(i)
    return names

class RoadSequenceDataset(Dataset):

    def __init__(self, file_path, transforms):

        self.img_list = readTxt(file_path)
        self.dataset_size = len(self.img_list)
        self.transforms = transforms
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        img_path_list = self.img_list[idx]
        data = Image.open(img_path_list[4])
        label = Image.open(img_path_list[5])
        data = self.transforms(data)
        label = torch.squeeze(self.transforms(label))
        sample = {'data': data, 'label': label}
        return sample

class RoadSequenceDatasetList2(Dataset):

    def __init__(self, file_path, transforms):

        self.img_list = readTxt(file_path)
        self.dataset_size = len(self.img_list)
        self.transforms = transforms
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        img_path_list = self.img_list[idx]
        data = []
        for i in range(5):
            data.append(torch.unsqueeze(self.transforms(Image.open(img_path_list[i])), dim=0))
        data = torch.cat(data, 0)
        label = Image.open(img_path_list[5])
        label = torch.squeeze(self.transforms(label))
        sample = {'data': data, 'label': label}
        return sample

class RoadSequenceDatasetList(Dataset):

    def __init__(self, image_sequence, transforms):
        self.images = image_sequence
        self.dataset_size = 1
        self.transforms = transforms
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        data = []
        # for i in range(5):
        data.append(torch.unsqueeze(self.transforms(self.images), dim=0))
        data = torch.cat(data, 0)
        # label = torch.squeeze(self.transforms(Image.fromarray(np.zeros_like(self.images[0]))))
        # sample = {'data': data, 'label': label}
        sample = {'data': data}
        return sample

def screenshot():
        bounding_box = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
        sct = mss()
        sct_img = sct.grab(bounding_box)
        frame = Image.frombytes(
                'RGB', 
                (sct_img.width, sct_img.height), 
                sct_img.rgb, 
            )
        return frame
    
if __name__ == '__main__':
    op_tranforms = transforms.Compose([transforms.ToTensor()])
    
    image_sequence = []
    for i in range(5):
        image_sequence.append(screenshot())
        sleep(0.01)
    
    a = RoadSequenceDatasetList(image_sequence, transforms=op_tranforms)
    b = RoadSequenceDatasetList2(file_path=config.test_path, transforms=op_tranforms)
    
    # print(b[0] == a[0])
    print(b[2])