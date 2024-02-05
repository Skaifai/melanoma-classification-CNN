import copy
import math
from Code.scripts.utils import check_accuracy
from tqdm import tqdm


def train(model, criterion, optimizer, train_loader, validation_loader, num_epochs, y_validation_acc,
          y_training_loss, y_validation_f2, device, scheduler=None):
    best = copy.deepcopy(model)
    best_epoch = 0
    max_acc = 0.0
    max_f2 = 0.0
    print("START THE EPOCH LOOP")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print("BEGIN TRAINING")
        loop = tqdm(train_loader, total=len(train_loader), leave=True)
        for images, labels in loop:
            # Pass the images and labels to the device
            images = images.to(device)
            labels = labels.to(device)
            # Zero grads
            optimizer.zero_grad()
            # Make predictions
            outputs = model(images)
            # Compute loss and gradients
            loss = criterion(outputs, labels.long())
            # Check if loss is nan and stop the training process
            if math.isnan(loss):
                print("Loss is NaN!")
                print(loss)
                return best
            # Backpropagation
            loss.backward()
            # Adjust learning weights
            optimizer.step()
            running_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader)
        print('Epoch loss {:.3f}'.format(epoch_loss))
        y_training_loss.append(epoch_loss)

        # Check validation accuracy every epoch
        print("### CHECKING VALIDATION ACCURACY ###")
        temp_acc, temp_f2 = check_accuracy(validation_loader, model, device)
        if temp_acc > max_acc and epoch >= 5 and temp_acc > 70.0:
            if temp_f2 > max_f2 and temp_f2 >= 0.4:
                max_acc = temp_acc
                max_f2 = temp_f2
                best = copy.deepcopy(model)
                best_epoch = epoch+1
        y_validation_acc.append(temp_acc)
        y_validation_f2.append(temp_f2)
        print('Current best epoch:', best_epoch)
        print('Current best accuracy:', max_acc)
        print('Current best F2:', max_f2)
        # Learning rate step
        if scheduler is not None:
            # For step schedulers
            scheduler.step()
            # For ReduceLROnPlateau
            if type(scheduler).__name__ == "ReduceOnPlateau":
                scheduler.step(temp_acc)
    print("BEST EPOCH:", best_epoch)
    return best
