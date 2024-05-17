import os
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from models import *
from training import train
from datasets import BenignAndMalignantDataset
from transforms import *
from utils import *
from local_variables import *
import torch
import torch.nn as nn
from torchinfo import summary
import matplotlib.pyplot as plt

# Ignore deprecated warnings
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def main():
    # Paths to relevant files
    path_to_truth = TRUTH_FILE
    image_dir = IMAGE_DIR
    train_dir = TRAIN_DIR
    test_dir = TEST_DIR
    models_dir = MODELS_DIR

    # Determine the device for the calculations
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    # What device are we using
    print("Using ", device)
    print(torch.cuda.get_device_name(0))

    # Hyperparameters
    num_epochs = 100
    learning_rate = 0.0001
    batch_size = 16
    # batch_size = 32
    # batch_size = 64
    num_workers = 1
    pin_memory = True
    shuffle = False
    test_set_ratio = 0.1
    weight_multiplier = 1.0

    # Model to train
    model = CNNv3().to(device)

    # Transform to use
    transform = transform_v1

    # Optimizer to use
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)

    # Scheduler for the learning rate
    scheduler = None
    epoch_step = 25
    scheduler_step = 0.1
    scheduler = lr_scheduler.StepLR(optimizer, epoch_step, gamma=scheduler_step, verbose=True)

    # Split the data into training and testing sets while stratifying based on class labels
    train_df, test_df = prepare_dataframes(path_to_truth, test_set_ratio)

    # Display the shape of the training and testing sets
    print("Training set shape:", train_df.shape)
    print("Testing set shape:", test_df.shape)

    # Creating a csv file out of the generated dataframes, if the csv file already exists, then change the dataframe
    # based on the existing csv file
    if create_csv_from_dataframe(train_df, train_dir + '/train.csv') is False:
        create_dataframe_from_csv(train_dir + '/train.csv')
    if create_csv_from_dataframe(test_df, test_dir + '/test.csv') is False:
        create_dataframe_from_csv(test_dir + '/test.csv')

    # Create directories with images
    create_image_dir(train_df, image_dir, train_dir)
    create_image_dir(test_df, image_dir, test_dir)

    # Creating relevant arrays
    # y_validation_loss = []
    y_validation_f2 = []
    y_validation_acc = []
    y_training_loss = []
    # y_training_f2 = []
    # y_training_acc = []

    # Defining loss functions and the optimizer
    # Since the dataset is imbalanced, we need to set weights
    # TODO: deal with this mess later
    train_distribution_df = train_df.groupby('label').count()
    print(train_distribution_df)
    test_distribution = test_df.groupby('label').count()
    print(test_distribution)
    train_distribution = [train_distribution_df.iloc[0][0], train_distribution_df.iloc[1][0]]
    train_weights_array = [1.0, train_distribution[0] / train_distribution[1]]
    # Modify weights by the weight multiplier
    train_weights_array[1] = train_weights_array[1] * weight_multiplier
    train_weights = torch.FloatTensor(train_weights_array).to(device)
    print(train_weights)

    # Loss function
    criterion = nn.CrossEntropyLoss(weight=train_weights).to(device)

    # Naming
    CNN_VERSION = type(model).__name__ + "_"
    TRANSFORM_NAME = "transform_"
    OPTIMIZER = type(optimizer).__name__ + "_"
    if scheduler is not None:
        SCHEDULER = type(scheduler).__name__ + "-step-" + str(epoch_step) + "-gamma-" + str(scheduler_step) + "_"
    else:
        SCHEDULER = ""
    WEIGHTS = "training_weights-" + str(round(train_weights_array[0], 2)) + "-" + str(round(train_weights_array[1], 2))
    MODEL_NAME = CNN_VERSION + OPTIMIZER + SCHEDULER + TRANSFORM_NAME + WEIGHTS

    # Print out the summary of the model
    model_summary = summary(model, (batch_size, 3, 28, 28), verbose=False, device=device)

    # Modify models directory based on the model's name and optimizer
    models_dir = models_dir + "/" + type(model).__name__ + "/" + type(optimizer).__name__

    # Create saves directory for the models
    saves_dir_path = create_model_dir(MODEL_NAME, models_dir)

    # Create a txt file with the model summary
    save_txt(str(model_summary), saves_dir_path)

    # Creating datasets and their respective dataloader
    dataset = BenignAndMalignantDataset(train_dir, train_dir + '/train.csv', transform=transform)
    train_set, validation_set = torch.utils.data.random_split(dataset, [0.9, 0.1])
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

    # TRAINING THE MODEL
    best = train(model, criterion, optimizer, train_loader, validation_loader, num_epochs, y_validation_acc,
                 y_training_loss, y_validation_f2, device, scheduler)

    # Testing dataset
    test_set = BenignAndMalignantDataset(test_dir, test_dir + "/test.csv", transform=valid_transform)
    test_loader = DataLoader(dataset=test_set,
                             shuffle=shuffle,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    # Trial on the test dataset
    accuracy, f2_score = check_accuracy(test_loader, model, device)
    print("Model's Accuracy: " + str(round(accuracy)))

    accuracy_best, f2_score_best = check_accuracy(test_loader, best, device)
    print("Best Model's Acc: " + str(round(accuracy_best)))

    # Get script's name
    # script_name = os.path.basename(__file__)

    # Make dir specifically for the current iteration of the model
    iteration_dir_path = create_model_iteration_dir(MODEL_NAME, num_epochs, learning_rate, batch_size, saves_dir_path)

    # Save the model
    save_model(model, MODEL_NAME, accuracy, f2_score, num_epochs, learning_rate, batch_size, iteration_dir_path)

    # Save the best model
    save_model(best, "best "+MODEL_NAME, accuracy_best, f2_score_best, num_epochs, learning_rate, batch_size,
               iteration_dir_path)

    # Graphs
    # x_validation_loss = range(0, len(y_validation_loss))
    x_validation_f2 = range(0, len(y_validation_f2))
    x_validation_acc = range(0, len(y_validation_acc))
    x_training_loss = range(0, len(y_training_loss))
    # x_training_acc = range(0, len(y_training_acc))
    # x_training_f2 = range(0, len(y_training_f2))

    # plot the accuracy function graph
    plt.plot(x_validation_acc, y_validation_acc, color='r', label='Validation Accuracy')
    plt.grid(True)
    plt.title('Accuracy for Validation Dataset')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy %')
    plt.savefig(os.path.join(iteration_dir_path, "ValAcc.png"))
    plt.figure()

    # plot the loss function graph
    plt.plot(x_training_loss, y_training_loss, color='b', label='Training Loss')
    plt.grid(True)
    plt.title('Loss for the Training Dataset')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(iteration_dir_path, "TrainLoss.png"))
    plt.figure()

    # plot the f2 score graph
    plt.plot(x_validation_f2, y_validation_f2, color='g', label='Validation F2 Score')
    plt.grid(True)
    plt.title('F2 for the Validation Dataset')
    plt.xlabel('Epochs')
    plt.ylabel('F2')
    plt.savefig(os.path.join(iteration_dir_path, "ValF2.png"))
    plt.figure()

    return


if __name__ == "__main__":
    main()
