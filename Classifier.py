import os
import cv2
import sys
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import random
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def main():
    # path to the directory
    path = sys.argv[1]

    # path to all sub directories
    subdirs = os.listdir(path)

    for subdir in subdirs:
        # list of files
        filenames = os.listdir(os.path.join(path, subdir))



        for filename in filenames:

            image = cv2.imread(os.path.join(path, subdir, filename))


            image_gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


            # dimension min of the img
            min_dim = min(image_gris.shape)

            # square size
            marge_h = (image_gris.shape[0] - min_dim) // 2
            marge_w = (image_gris.shape[1] - min_dim) // 2

            # white padding
            image_padded = cv2.copyMakeBorder(image_gris, marge_h, marge_h, marge_w, marge_w, cv2.BORDER_CONSTANT, value=255)

            # resize to (32,32)
            image_resized = cv2.resize(image_padded, (32,32))
            _, threshpic = cv2.threshold(image_resized, 95, 255, cv2.THRESH_BINARY)

            os.makedirs("all", exist_ok=True)

            image_path = os.path.join("all", subdir, filename)

            os.makedirs(os.path.dirname(image_path), exist_ok=True)

            cv2.imwrite(image_path, threshpic)

    letters_path = 'all'

    # List to stock my data that i want to use for separate it in groups
    data = []

    # to get all letter in my subdir and add it to my list (data)
    for label in os.listdir(letters_path):
        # to subdir
        label_path = os.path.join(letters_path, label)

        for image_filename in os.listdir(label_path):
            # path to the img of the letter
            image_path = os.path.join(label_path, image_filename)
            image = cv2.imread(image_path)
            # add the tuple to the list
            data.append((image, label))

    # Dictionnary to store the letter accuracies
    letter_accuracies = {}

    for label in os.listdir(letters_path):
        random.shuffle(data)

        # Split the data into the three sets (train, validation, and test)
        train_size = int(len(data) * 0.8)
        valid_size = int(len(data) * 0.1)
        test_size = int(len(data) * 0.1)
        train_data = data[:train_size]
        validation_data = data[train_size:train_size + valid_size]
        test_data = data[train_size + valid_size:]

        # Extract the features and labels from the sets
        X_train = [np.array(d[0]).flatten() for d in train_data]
        y_train = [d[1] for d in train_data]
        X_valid = [np.array(d[0]).flatten() for d in validation_data]
        y_valid = [d[1] for d in validation_data]
        X_test = [np.array(d[0]).flatten() for d in test_data]
        y_test = [d[1] for d in test_data]

        scores = []

        # Creation of our best model

        for k in range(1, 15, 2):
            model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')

            # Training
            model.fit(X_train, y_train)


            # Evaluate our model with each K
            score = model.score(X_valid, y_valid)

            scores.append(score)


        # Find the best score with scores of  validate group
        best_k_index = scores.index(max(scores))
        best_k = list(range(1, 15, 2))[best_k_index]


        # New model with our best K

        best_model = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')

        # Training the model on all our training value
        best_model.fit(X_train + X_valid, y_train + y_valid)

        # Evaluate the model on the test set
        test_accuracy = best_model.score(X_test, y_test)

        # Stock our results
        letter_accuracies[label] = test_accuracy


        f = open("results.txt", "w")
        f.write("the best value of k for this test is :{}\n".format(best_k))
        f.write("precision for each letter :\n")

        # sort by letters (numbers)
        sorted_letters = sorted(letter_accuracies.items(), key=lambda x: int(x[0]))

        for letter, accuracy in sorted_letters:
            f.write("{}: {}\n".format(letter, accuracy))

        f.close()

        predictions = model.predict(X_test)

        # calcul confusion matrix
        confusion_mat = confusion_matrix(y_test, predictions)

        # save matrix
        np.savetxt("confusion_matrix.csv", confusion_mat, delimiter=",")


if __name__ == "__main__":
    main()