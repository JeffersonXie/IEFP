
import os
import torch
import torch.utils.data as data
from PIL import Image
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

def default_loader(path):
    return Image.open(path).convert('RGB')

class Dataset_floder(data.Dataset):
    def __init__(self, data_root, data_list, transform = None, loader=default_loader):
        with open (data_list) as f:
            lines=f.readlines()
        imgs=[]
        for line in  lines:
            cls = line.split() 
            img_a_name = cls.pop(0)
            img_b_name = cls.pop(0)
            pair_label = cls.pop(0)
            if os.path.isfile(os.path.join(data_root, img_a_name)) and os.path.isfile(os.path.join(data_root, img_b_name)):
               # imgs.append((img_a_name, img_b_name, tuple([int(v) for v in cls])))
                imgs.append((img_a_name, img_b_name, int(pair_label)))
        self.data_root = data_root
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index): 
  #  def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img_a_name, img_b_name, label = self.imgs[index]
        img_a = self.loader(os.path.join(self.data_root, img_a_name))
        img_b = self.loader(os.path.join(self.data_root, img_b_name))
        if self.transform is not None:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
        return (img_a, img_b, label)
        

    def __len__(self):
        return len(self.imgs)
    


