from torch.utils.data import Dataset
import re
import glob
import os
from skimage.transform import rescale
from skimage import io
import torch

class ComphyDataset(Dataset):

    def __init__(
            self,
            split='train',
            data_root='',
            down_sz=64,
            ):
        self.down_sz = down_sz
        self.data_root = data_root
        self.files = self.get_files()
        split_index = int(0.95 * len(self.files))
        if split=='train':
            self.files = self.files[:split_index]
        elif split=='val':
            self.files = self.files[split_index:]
        print('{} items loaded for the {}_set'.format(len(self.files),split))

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        video_path = self.files[idx]
        frames = []
        for img_file in sorted(os.listdir(video_path),key=self.natural_sort_key):
            img_path = os.path.join(video_path, img_file)
            scaled_img = self.rescale_img(io.imread(img_path))
            img = torch.tensor(scaled_img,dtype=torch.float32).permute((2,0,1))
            frames.append(img)
        batch = torch.stack(frames,dim=0)
        return batch

    def get_files(self):
        paths=[]
        subdirs = os.listdir(self.data_root)
        for subdir in subdirs:
            videos = os.listdir(os.path.join(self.data_root,subdir))
            for video in videos:
                paths.append(os.path.join(self.data_root, subdir, video))
        return paths

    def rescale_img(self,img):
        H,W,C = img.shape
        down = rescale(img, [self.down_sz/H], order=3, mode='reflect', multichannel=True)
        return down

    def natural_sort_key(self,s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def build_comphyVid_dataset(args):
    train_dataset = ComphyDataset(split='train',**args)
    val_dataset = ComphyDataset(split='val',**args)
    return train_dataset, val_dataset
