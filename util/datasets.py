import os
import PIL

import torch
from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)


    # root = os.path.join(args.data_path, 'train' if is_train else 'val')
    # dataset = datasets.ImageFolder(root, transform=transform)
    
    if is_train:
        dataset = ImagenetDataset('train',
                                     args.data_path,
                                     transform=transform,
                                     )
    else:
        dataset = ImagenetDataset('val',
                                     args.data_path,
                                     transform=transform,
                                     )

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())

    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


from PIL import Image
from torch.utils.data import Dataset
import pyarrow as pa


def _get_images(annotations):
    images = []
    classes = []
    for line in annotations:
        if isinstance(line, bytes):
            line = line.decode()
        image_name, cls = line.strip('\n').split()
        images.append(image_name)
        classes.append(cls)
    return images, classes


class ImagenetDataset(Dataset):
    def __init__(self, image_set, data_path, transform=None):
        ann_file = os.path.join(data_path, f'meta/{image_set}.txt')
        data_path = os.path.join(data_path, image_set)
        self.image_set = image_set
        self.transform = transform
        self.data_path = data_path
        self.images, self.classes, self.class_to_idx = self._load_database(ann_file)

    def _load_database(self, annotation_file):
        annotation_file = os.path.abspath(annotation_file)
        print(f'loading annotations from {annotation_file} ...')
        with open(annotation_file, 'rt') as annotations:
            images, classes = _get_images(annotations)

        # convert possible classes to indices
        class_names = sorted(set(classes))
        # class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        class_to_idx = {class_name: int(class_name) for class_name in class_names}
        return pa.array(images), pa.array([class_to_idx[class_name] for class_name in classes]), class_to_idx

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index].as_py()
        target = self.classes[index].as_py()
        sample = self._load_image(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, path

    def _load_image(self, path):
        full_path = os.path.join(self.data_path, path)
        with open(full_path, 'rb') as f:
            return Image.open(f).convert('RGB')

