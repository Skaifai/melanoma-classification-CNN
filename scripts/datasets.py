from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torch


class BenignAndMalignantDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None, device=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotation_file)
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations.iloc[index, 0]
        y_label = torch.tensor(float(self.annotations.iloc[index, 1]))
        if y_label == 1:
            img = Image.open(os.path.join(self.root_dir, "1", img_id)).convert("RGB")
        else:
            img = Image.open(os.path.join(self.root_dir, "0", img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, y_label

    # def __getitem__(self, index):
    #     img_id = self.annotations.iloc[index, 0]
    #     y_label = torch.tensor(float(self.annotations.iloc[index, 1]))
    #     if y_label == 1:
    #         path = os.path.join(self.root_dir, "1", img_id)
    #         # pil_img = Image.open(os.path.join(self.root_dir, "1", img_id)).convert("RGB")
    #     else:
    #         path = os.path.join(self.root_dir, "0", img_id)
    #         # pil_img = Image.open(os.path.join(self.root_dir, "0", img_id)).convert("RGB")
    #     image = PIL.Image.open(path)
    #     arr = np.array(image)
    #     tensor = torch.from_numpy(arr)
    #     tensor.to(self.device)
    #     if self.transform is not None:
    #         tensor = self.transform(tensor)
    #
    #     return tensor, y_label