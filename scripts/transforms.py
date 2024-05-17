import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2

# Transforms
transform_v1 = transforms.Compose(
    [
        # v2.RandomResizedCrop(size=(28, 28), scale=(0.8, 1), ratio=(1, 1)),
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize((28, 28), antialias=True),
        v2.RandomRotation(180),
        v2.RandomHorizontalFlip(0.5),
        v2.RandomVerticalFlip(0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

transform_v2 = transforms.Compose(
    [
        # v2.RandomResizedCrop(size=(28, 28), scale=(0.8, 1), ratio=(1, 1)),
        v2.ToImage(),
        v2.Resize((32, 32), antialias=True),
        v2.CenterCrop((28, 28)),
        v2.ToDtype(torch.uint8, scale=True),
        v2.RandomRotation(180),
        v2.RandomHorizontalFlip(0.5),
        v2.RandomVerticalFlip(0.5),
        v2.ToDtype(torch.float64, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

transform_vgg16 = transforms.Compose(
    [
        # v2.RandomResizedCrop(size=(28, 28), scale=(0.8, 1), ratio=(1, 1)),
        v2.ToImage(),
        v2.Resize((227, 227), antialias=True),
        v2.CenterCrop((224, 224)),
        v2.ToDtype(torch.uint8, scale=True),
        v2.RandomRotation(180),
        v2.RandomHorizontalFlip(0.5),
        v2.RandomVerticalFlip(0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

valid_transform = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)



# TODO
def show_image_after_transforms(loader):
    image, _ = iter(loader).__next__()
    image = image.permute(1, 2, 0)
    plt.imshow(image)
    plt.show()
