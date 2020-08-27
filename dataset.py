import sys
import os
import torch.utils.data as data

from PIL import Image
from ParseVOC import voc_parsing

sys.path.insert(0, os.path.dirname(__file__))


class VOC(data.Dataset):
    IMAGE_FOLDER = "JPEGImages"
    LABEL_FOLDER = "Annotations"
    IMG_EXTENSIONS = '.jpg'

    def __init__(self, root, train=True, transform=None, target_transform=None, resize=448):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize_factor = resize

        if not self._check_exists():
            raise RuntimeError("Dataset not found.")

        self.data = voc_parsing()

    def _check_exists(self):
        print("Image Folder : {}".format(os.path.join(self.root, self.IMAGE_FOLDER)))
        print("Label Folder : {}".format(os.path.join(self.root, self.LABEL_FOLDER)))

        return os.path.exists(os.path.join(self.root, self.IMAGE_FOLDER)) and \
               os.path.exists(os.path.join(self.root, self.LABEL_FOLDER))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        key = list(self.data[index].keys())[0]

        img = Image.open(key).convert('RGB')
        current_shape = img.size
        img = img.resize((self.resize_factor, self.resize_factor))

        target = self.data[index][key]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            pass

        return img, target