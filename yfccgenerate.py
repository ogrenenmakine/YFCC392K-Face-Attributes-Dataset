from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from PIL import Image
import os

import time
from facenet_pytorch import MTCNN, InceptionResnetV1_fe, fixed_image_standardization, training
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm

class Augmentation(data.Dataset):
    def __init__(self, data_path, image_size):
        super(Augmentation, self).__init__()
        self.data_path = data_path

        self.images = np.genfromtxt(self.data_path, skip_header=0, usecols=[0], delimiter="\n", dtype=np.str)
        
        self.tf = transforms.Compose([
            transforms.RandomAffine(degrees=(3, 3),translate=(0.1, 0.1),scale=(0.9, 1.1)),
            transforms.ColorJitter(hue=.05, saturation=.05),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.length = len(self.images)
    def __getitem__(self, index):
        return torch.stack([self.tf(Image.open(self.images[index])) for i in range(128)])
    def __len__(self):
        return self.length
    

    
# Root directory for dataset
data_path  = "list_of_face_images.txt"

# Number of workers for dataloader
workers = os.cpu_count()

# Batch size during training
batch_size = 1

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = (128,128)

dataset = Augmentation(data_path, image_size)
test_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=workers)

device = torch.device("cuda:0")

# Changed the code, when it returns, now it returns the features and labels. Look at the line 637 in the inception_resnet_v1.py
resnet = InceptionResnetV1_fe(
    classify=False,
    num_classes=40
).to(device)

#resnet = nn.DataParallel(resnet)

pretrained_dict = torch.load("resnet_fe*.pth")
pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
resnet.load_state_dict(pretrained_dict)

print("model is ready")

resnet.eval()
file_to_write = open("results_without_sharpen_labels.txt","a")
file_to_write_sharpen = open("results_with_sharpen_labels.txt","a")
features_to_write = open("results_without_sharpen_features.txt", "a")
print("beginning to get the labels and features")
file_name = np.genfromtxt(data_path, usecols=[0], delimiter="\n", dtype=np.str)
for (i, batch) in tqdm(enumerate(test_loader)):
    batch = torch.squeeze(batch,0).to(device)
    feature, label = resnet(batch) #model returns features(2048) and labels(40)
    label_normal = torch.mean(torch.sigmoid(label),0)*2-1  #average the labels (256, 40) -> (1, 40)
    label_sharpened = torch.mean(torch.sigmoid(label * 3), 0)*2 - 1
    feature = torch.mean(feature, 0)                #average the features shape (256, 2048) -> (1, 2048)
    label_text = '\t'.join(list(label_normal.cpu().detach().numpy().astype(str)))
    feature_text = '\t'.join(list(feature.cpu().detach().numpy().astype(str)))
    
    label_sharpened_text = '\t'.join(list(label_sharpened.cpu().detach().numpy().astype(str)))
    file_to_write.write(file_name[i]+'\t'+label_text+"\n")
    features_to_write.write(file_name[i]+'\t'+feature_text+"\n")
    file_to_write_sharpen.write(file_name[i]+'\t'+label_sharpened_text+"\n")
    

    
file_to_write.close() 
features_to_write.close()
file_to_write_sharpen.close()


# This below code sample is from the linear_labeling.ipynb file. 
# I think this loop is unncessary, because when we look at the code below, it gets the prediction and
# multiplies with 3, we already got the prediction in line 83. Therefore, we can just mutliply by 
# three in the upper for loop. 


