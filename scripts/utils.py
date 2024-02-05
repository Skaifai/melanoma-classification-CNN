import shutil
import pandas as pd
import os.path
from sklearn.model_selection import train_test_split
import torch
from datetime import datetime


def prepare_dataframes(truth_path, test_size):
    df = pd.read_csv(truth_path, header=0)
    train_dataframe, test_dataframe = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=42)
    return train_dataframe, test_dataframe


def create_csv_from_dataframe(dataframe, target_path):
    path_to_file = target_path.replace('\\', '/')
    if os.path.isfile(target_path):
        print('File at ' + '\'' + target_path + '\'' + ' already exists!')
        return False
    else:
        dataframe.to_csv(path_to_file, index=False, header=True)
        return True


def create_dataframe_from_csv(path_to_file, header=0):
    path_to_file = path_to_file.replace('\\', '/')
    if os.path.isfile(path_to_file):
        df = pd.read_csv(path_to_file, header=header)
        return df
    else:
        raise FileNotFoundError("File not found at "+path_to_file+"!")


def create_image_dir(dataframe, image_dir_path, target_dir_path):
    # Replace \ with / for consistency
    image_dir_path = image_dir_path.replace('\\', '/')
    target_dir_path = target_dir_path.replace('\\', '/')
    # Check if directory already exists
    isdir = os.path.isdir(target_dir_path)
    if not isdir:
        # Iterate through the training set and copy/move images
        for index, row in dataframe.iterrows():
            image_filename = row['img_name']
            image_class = row['label']
            src_path = os.path.join(image_dir_path, image_filename)
            dst_path = os.path.join(target_dir_path, str(image_class), image_filename)
            os.makedirs(os.path.join(target_dir_path, str(image_class)), exist_ok=True)
            shutil.copy(src_path, dst_path)
    else:
        print('Target directory at ' + '\'' + target_dir_path + '\'' + ' already exists!')


def create_model_dir(model_name, target_dir_path):
    saves_folder_name = model_name + " saves"
    saves_dir_path = os.path.join(target_dir_path, saves_folder_name).replace('\\', '/')

    isdir = os.path.isdir(saves_dir_path)
    if not isdir:
        os.makedirs(saves_dir_path.replace('/', '\\'), exist_ok=True)
    else:
        print("Directory at " + saves_dir_path + " already exists!")
    return saves_dir_path


def save_txt(contents, target_dir_path):
    target_dir_path = target_dir_path.replace('\\', '/')
    file_path = target_dir_path+"/configuration.txt"
    isdir = os.path.isdir(target_dir_path)
    if not isdir:
        raise NotADirectoryError("Directory at " + target_dir_path + " does not exist!")
    else:
        if os.path.isfile(file_path):
            print("Configuration file already exists at {}.".format(file_path))
        else:
            with open(target_dir_path+"/configuration.txt", 'w', encoding="utf-8") as f:
                f.write(contents)
                f.close()
                print("Written the following contents to the file at " + file_path)
                print(contents)


def create_model_iteration_dir(model_name, num_epochs, learning_rate, batch_size, saves_dir_path):
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y--%H-%M-%S")
    iteration_folder_name = model_name + ' ' + dt_string + " e" + str(num_epochs) + " lr" + str(learning_rate) + \
        " bs" + str(batch_size)
    iteration_dir_path = os.path.join(saves_dir_path, iteration_folder_name).replace('\\', '/')
    os.mkdir(iteration_dir_path.replace('/', '\\'))
    return iteration_dir_path


def save_model(model, model_name, accuracy, f2_score, num_epochs, learning_rate, batch_size, iteration_dir_path):
    model_name = model_name + " (" + str(round(accuracy)) + "%) " + " F2_" + "{:.2f}".format(f2_score) + " e" + \
        str(num_epochs) + " lr" + str(learning_rate) + " bs" + str(batch_size)
    torch.save(model, os.path.join(iteration_dir_path, model_name).replace("\\", "/"))


def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    num_true_pos = 0
    num_false_pos = 0
    num_true_neg = 0
    num_false_neg = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            predictions = scores.argmax(1)
            # noinspection PyTypeChecker
            num_correct += torch.sum(predictions == y)

            for idx in range(len(predictions)):
                if predictions[idx] == 1 and y[idx] == predictions[idx]:
                    num_true_pos += 1
                elif predictions[idx] == 1 and y[idx] != predictions[idx]:
                    num_false_pos += 1
                elif predictions[idx] == 0 and y[idx] == predictions[idx]:
                    num_true_neg += 1
                elif predictions[idx] == 0 and y[idx] != predictions[idx]:
                    num_false_neg += 1

            num_samples += predictions.size(0)
    accuracy = float(num_correct) / float(num_samples) * 100
    if float(num_true_pos) + float(num_false_pos) != 0.0:
        precision = float(num_true_pos) / (float(num_true_pos) + float(num_false_pos))
    else:
        precision = 0.0
    if (float(num_true_pos) + float(num_false_neg)) != 0.0:
        recall = float(num_true_pos) / (float(num_true_pos) + float(num_false_neg))
    else:
        recall = 0.0
    if num_true_pos + 0.8 * num_false_neg + 0.2 * num_false_pos != 0.0:
        f2_score = num_true_pos / (num_true_pos + 0.8 * num_false_neg + 0.2 * num_false_pos)
    else:
        f2_score = 0.0
    # print("True Positive", num_true_pos)
    # print("False Positive", num_false_pos)
    # print("True Negative", num_true_neg)
    # print("False Negative", num_false_neg)
    print(f"Precision Score {precision:.2f}")
    print(f"Recall Score {recall:.2f}")
    print(f"F2 Score {f2_score:.2f}")
    print("#########")
    print(f"Got {num_correct} / {num_samples} with accuracy {accuracy:.2f}.")
    print("#########")
    return accuracy, f2_score
