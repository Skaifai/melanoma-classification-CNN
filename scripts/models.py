import torch
import torch.nn as nn
from Code.scripts.custom_functions import ProposedActivation


# The original model proposed in "An accurate and noninvasive skin cancer screening based on imaging technique"
class CNNv0(nn.Module):
    def __init__(self):
        super(CNNv0, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)  # 26
        self.af1 = ProposedActivation()
        self.max1 = nn.MaxPool2d(kernel_size=3, stride=1)  # 24
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)  # 22
        self.af2 = ProposedActivation()
        self.max2 = nn.MaxPool2d(kernel_size=3, stride=1)  # 20
        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)  # 18
        self.af3 = ProposedActivation()
        # Fourth convolutional layer
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1)  # 16
        self.af4 = ProposedActivation()
        self.max3 = nn.MaxPool2d(kernel_size=3, stride=2)  # 7
        # Flatten the tensor to use the FC layer
        self.flatten = nn.Flatten()
        # Fully connected layer
        self.dense1 = nn.Linear(in_features=3136, out_features=512)
        self.af5 = ProposedActivation()
        self.dropout = nn.Dropout1d()
        self.dense2 = nn.Linear(in_features=512, out_features=2)

    def forward(self, x):
        # Pass data to the first convolutional layer
        x = self.conv1(x)
        # Use the proposed activation function
        x = self.af1(x)
        # Run the first max pooling over x
        x = self.max1(x)

        x = self.conv2(x)
        x = self.af2(x)
        x = self.max2(x)

        x = self.conv3(x)
        x = self.af3(x)

        x = self.conv4(x)
        x = self.af4(x)
        x = self.max3(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = self.af5(x)
        x = self.dropout(x)
        x = self.dense2(x)

        return x


# v0 but with xavier weight initialization and batch normalization layers
class CNNv0a1(nn.Module):
    def __init__(self):
        super(CNNv0a1, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)  # 26
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(32)
        self.af1 = ProposedActivation()
        self.max1 = nn.MaxPool2d(kernel_size=3, stride=1)  # 24
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)  # 22
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(64)
        self.af2 = ProposedActivation()
        self.max2 = nn.MaxPool2d(kernel_size=3, stride=1)  # 20
        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)  # 18
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        self.af3 = ProposedActivation()
        # Fourth convolutional layer
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1)  # 16
        torch.nn.init.xavier_uniform_(self.conv4.weight)
        self.bn3 = nn.BatchNorm2d(64)
        self.af4 = ProposedActivation()
        self.max3 = nn.MaxPool2d(kernel_size=3, stride=2)  # 7
        # Flatten the tensor to use the FC layer
        self.flatten = nn.Flatten()
        # Fully connected layer
        self.dense1 = nn.Linear(in_features=3136, out_features=512)
        self.af5 = ProposedActivation()
        self.dropout = nn.Dropout1d()
        self.dense2 = nn.Linear(in_features=512, out_features=2)

    def forward(self, x):
        # Pass data to the first convolutional layer
        x = self.conv1(x)
        x = self.bn1(x)
        # Use the proposed activation function
        x = self.af1(x)
        # Run the first max pooling over x
        x = self.max1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.af2(x)
        x = self.max2(x)

        x = self.conv3(x)
        x = self.af3(x)

        x = self.conv4(x)
        x = self.bn3(x)
        x = self.af4(x)
        x = self.max3(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = self.af5(x)
        x = self.dropout(x)
        x = self.dense2(x)

        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)  # 26
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.af1 = ProposedActivation()
        self.bn1 = nn.BatchNorm2d(32)
        self.max1 = nn.MaxPool2d(kernel_size=3, stride=2)  # 12
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)  # 10
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.af2 = ProposedActivation()
        self.bn2 = nn.BatchNorm2d(64)
        self.max2 = nn.MaxPool2d(kernel_size=3, stride=1)  # 8
        # Third convolutional layer
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2) # 9
        # torch.nn.init.xavier_uniform_(self.conv3.weight)
        # self.af3 = ProposedActivation()
        # self.bn3 = nn.BatchNorm2d(128)
        # self.max3 = nn.MaxPool2d(kernel_size=3, stride=2) # 4
        # Flatten the tensor to use the FC layer
        self.flatten = nn.Flatten()
        # Fully connected layer
        self.dense1 = nn.Linear(in_features=4096, out_features=2048)
        self.af5 = ProposedActivation()
        self.dense2 = nn.Linear(in_features=2048, out_features=1024)
        self.af6 = ProposedActivation()
        self.dropout = nn.Dropout1d()
        self.dense3 = nn.Linear(in_features=1024, out_features=2)

    def forward(self, x):
        # Pass data to the first convolutional layer
        x = self.conv1(x)
        # Use the proposed activation function
        x = self.af1(x)
        x = self.bn1(x)
        # Run the first max pooling over x
        x = self.max1(x)

        x = self.conv2(x)
        x = self.af2(x)
        x = self.bn2(x)
        x = self.max2(x)

        # x = self.conv3(x)
        # x = self.af3(x)
        # x = self.bn3(x)
        # x = self.max3(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = self.af5(x)
        x = self.dense2(x)
        x = self.af6(x)
        x = self.dropout(x)
        x = self.dense3(x)

        return x


class CNNv1(nn.Module):
    def __init__(self):
        super(CNNv1, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)  # 26
        self.af1 = ProposedActivation()
        self.max1 = nn.MaxPool2d(kernel_size=3, stride=1)  # 24
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)  # 11
        self.af2 = ProposedActivation()
        self.max2 = nn.MaxPool2d(kernel_size=3, stride=2)  # 5
        # Flatten the tensor to use the FC layer
        self.flatten = nn.Flatten()
        # Fully connected layer
        self.dense1 = nn.Linear(in_features=1600, out_features=512)
        self.af3 = ProposedActivation()
        self.dropout = nn.Dropout1d()
        self.dense2 = nn.Linear(in_features=512, out_features=2)

    def forward(self, x):
        # Pass data to the first convolutional layer
        x = self.conv1(x)
        # Use the proposed activation function
        x = self.af1(x)
        # Run the first max pooling over x
        x = self.max1(x)

        x = self.conv2(x)
        x = self.af2(x)
        x = self.max2(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = self.af3(x)
        x = self.dropout(x)
        x = self.dense2(x)

        return x


class CNNv2(nn.Module):
    def __init__(self):
        super(CNNv2, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)  # 26
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.af1 = ProposedActivation()
        self.bn1 = nn.BatchNorm2d(32)
        self.max1 = nn.MaxPool2d(kernel_size=3, stride=1)  # 24
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1)  # 21
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.af2 = ProposedActivation()
        self.bn2 = nn.BatchNorm2d(64)
        self.max2 = nn.MaxPool2d(kernel_size=3, stride=1)  # 19
        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)  # 9
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        self.af3 = ProposedActivation()
        self.bn3 = nn.BatchNorm2d(128)
        self.max3 = nn.MaxPool2d(kernel_size=3, stride=2)  # 4
        # Flatten the tensor to use the FC layer
        self.flatten = nn.Flatten()
        # Fully connected layer
        self.dense1 = nn.Linear(in_features=2048, out_features=1024)
        self.af5 = ProposedActivation()
        self.dense2 = nn.Linear(in_features=1024, out_features=512)
        self.af6 = ProposedActivation()
        self.dropout = nn.Dropout1d()
        self.dense3 = nn.Linear(in_features=512, out_features=2)

    def forward(self, x):
        # Pass data to the first convolutional layer
        x = self.conv1(x)
        # Use the proposed activation function
        x = self.af1(x)
        x = self.bn1(x)
        # Run the first max pooling over x
        x = self.max1(x)

        x = self.conv2(x)
        x = self.af2(x)
        x = self.bn2(x)
        x = self.max2(x)

        x = self.conv3(x)
        x = self.af3(x)
        x = self.bn3(x)
        x = self.max3(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = self.af5(x)
        x = self.dense2(x)
        x = self.af6(x)
        x = self.dropout(x)
        x = self.dense3(x)

        return x


class CNNv3(nn.Module):
    def __init__(self):
        super(CNNv3, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)  # 26
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(32)
        self.af1 = ProposedActivation()
        self.max1 = nn.MaxPool2d(kernel_size=3, stride=1)  # 24
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1)  # 21
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(64)
        self.af2 = ProposedActivation()
        self.max2 = nn.MaxPool2d(kernel_size=3, stride=1)  # 19
        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)  # 9
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        self.bn3 = nn.BatchNorm2d(128)
        self.af3 = ProposedActivation()
        self.max3 = nn.MaxPool2d(kernel_size=3, stride=2)  # 4
        # Flatten the tensor to use the FC layer
        self.flatten = nn.Flatten()
        # Fully connected layer
        self.dense1 = nn.Linear(in_features=2048, out_features=1024)
        self.af5 = ProposedActivation()
        self.dense2 = nn.Linear(in_features=1024, out_features=512)
        self.af6 = ProposedActivation()
        self.dropout = nn.Dropout1d()
        self.dense3 = nn.Linear(in_features=512, out_features=2)

    def forward(self, x):
        # Pass data to the first convolutional layer
        x = self.conv1(x)
        # Use the proposed activation function
        x = self.af1(x)
        x = self.bn1(x)
        # Run the first max pooling over x
        x = self.max1(x)

        x = self.conv2(x)
        x = self.af2(x)
        x = self.bn2(x)
        x = self.max2(x)

        x = self.conv3(x)
        x = self.af3(x)
        x = self.bn3(x)
        x = self.max3(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = self.af5(x)
        x = self.dense2(x)
        x = self.af6(x)
        x = self.dropout(x)
        x = self.dense3(x)

        return x


class CNNv4(nn.Module):
    def __init__(self):
        super(CNNv4, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)  # 26
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(32)
        self.af1 = ProposedActivation()
        self.max1 = nn.MaxPool2d(kernel_size=3, stride=2)  # 12
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)  # 10
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(64)
        self.af2 = ProposedActivation()
        self.max2 = nn.MaxPool2d(kernel_size=3, stride=1)  # 8
        # Flatten the tensor to use the FC layer
        self.flatten = nn.Flatten()
        # Fully connected layer
        self.dense1 = nn.Linear(in_features=4096, out_features=2048)
        self.af5 = ProposedActivation()
        self.dense2 = nn.Linear(in_features=2048, out_features=1024)
        self.af6 = ProposedActivation()
        self.dropout = nn.Dropout1d()
        self.dense3 = nn.Linear(in_features=1024, out_features=2)

    def forward(self, x):
        # Pass data to the first convolutional layer
        x = self.conv1(x)
        # Use the proposed activation function
        x = self.af1(x)
        x = self.bn1(x)
        # Run the first max pooling over x
        x = self.max1(x)

        x = self.conv2(x)
        x = self.af2(x)
        x = self.bn2(x)
        x = self.max2(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = self.af5(x)
        x = self.dense2(x)
        x = self.af6(x)
        x = self.dropout(x)
        x = self.dense3(x)

        return x


class VGG16(nn.Module):
    def __init__(self, num_classes=2):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
