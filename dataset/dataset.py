from PIL import Image
from torch.utils.data import Dataset
import torch
from .constant import LABEL2ID

class HENetdataset(Dataset):
    def __init__(self, img_paths, transform = None):
        self.img_paths = img_paths
        self.transform = transform
    def __len__(self):
        return len(self.img_paths)


    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        #img_np = np.array(img, dtype=np.uint8)

        res = self.transform(img) if self.transform else img

        label = LABEL2ID[img_path.split(",")[1][:-4]]
        return res, torch.as_tensor(label, dtype= torch.long)
    