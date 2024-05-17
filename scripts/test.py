import os.path
from local_variables import *
from Code.scripts.datasets import BenignAndMalignantDataset
from Code.scripts.transforms import valid_transform, show_image_after_transforms
from Code.scripts.utils import check_accuracy
import torch
from torch.utils.data import DataLoader

PATH_TO_MODEL = "C:/Users/skaif/Documents/CNN/Models/beastv1 saves/beastv1 24-01-2024--06-23-14/" \
                "beastv124-01-2024--06-23-14(80%)e50lr0.001bs32"


def main():
    train_dir = TRAIN_DIR
    test_dir = TEST_DIR
    shuffle = False
    batch_size = 32
    num_workers = 1
    pin_memory = True

    # Determine the device for the calculations
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    # What device are we using
    print("Using ", device)
    print(torch.cuda.get_device_name(0))

    # Creating datasets and their respective dataloader
    dataset = BenignAndMalignantDataset(train_dir, train_dir + '/train.csv', transform=valid_transform)
    train_set, validation_set = torch.utils.data.random_split(dataset, [1036, 115])
    train_loader = DataLoader(dataset=train_set,
                              shuffle=shuffle,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    validation_loader = DataLoader(dataset=validation_set,
                                   shuffle=shuffle,
                                   batch_size=batch_size,
                                   num_workers=num_workers,
                                   pin_memory=pin_memory)
    # Testing dataset
    test_set = BenignAndMalignantDataset(test_dir, test_dir + "/test.csv", transform=valid_transform)
    test_loader = DataLoader(dataset=test_set,
                             shuffle=shuffle,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    # Show example of the image. TODO
    # show_image_after_transforms(test_set)

    # Model class must be defined somewhere
    model = torch.load(PATH_TO_MODEL)
    model.eval()
    model_name = os.path.basename(PATH_TO_MODEL)
    real_name = type(model).__name__
    print("Testing " + model_name)
    print("Class: " + real_name)

    # Check model on the validation set
    val_acc = check_accuracy(validation_loader, model, device)
    print("Validation Accuracy: " + str(round(val_acc[0])))
    # Check model on the test set
    test_acc = check_accuracy(test_loader, model, device)
    print("Test Accuracy: " + str(round(test_acc[0])))

    # Check model on the training set
    train_acc = check_accuracy(train_loader, model, device)
    print("Training Accuracy: " + str(round(train_acc[0])))


if __name__ == "__main__":
    main()
