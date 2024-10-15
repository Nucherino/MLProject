from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import matplotlib.pyplot as plt

def SVM(split):
    x_train, x_test, y_train, y_test = split

    # kernels = ["poly", "sigmoid", "rbf", "linear"]
    # def try_kernel(kernel: str, iterations: int):
    #     accuracy = 0
    #     precision = 0
    #     recall = 0
    #     f1score = 0
 
    #     for i in range(iterations):
    #         print(f"iteration {i} in {kernel}")
    #         svm = SVC(kernel=kernel)
    #         svm.fit(x_train, y_train)

    #         y_pred = svm.predict(x_test)

    #         precision += precision_score(y_test, y_pred, average="micro")
    #         recall += recall_score(y_test, y_pred, average="micro")
    #         f1score += f1_score(y_test, y_pred, average="micro")
    #         accuracy += accuracy_score(y_test, y_pred)

    #     return f"{kernel}: [Accuracy: {accuracy/iterations}, Recall: {recall/iterations}, F1: {f1score/iterations}, Precision: {precision/iterations}]"
    
    # iterations = 10
    # results = [try_kernel(kernel, iterations) for kernel in kernels]


    svm = SVC(kernel="linear")
    svm.fit(x_train, y_train)

    y_pred = svm.predict(x_test)

    print(y_pred)

    decision_values = svm.decision_function(x_train)
    plt.plot(decision_values)
    plt.title('SVM Decision Function Values over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Decision Function Value')
    plt.show()

def NN(split):
    x_train, x_test, y_train, y_test = split

    model = Sequential([
        Dense(18, activation="relu", input_shape=(x_train.shape[1],)),
        Dense(10, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    
    output = model.fit(x_train, y_train, epochs=10000, batch_size=64, validation_split=0.3)

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

    y_pred_prob = model.predict(x_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    print("Predicted labels:", y_pred)
    print("Actual Labels: ", y_test)

    plt.plot(output.history['accuracy'])
    plt.title(f'Neural Network Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()


def main():
    data = pd.read_csv("cirrhosis.csv")
    data = data.drop(columns=["ID"]) # this is literally a useless column and would only make training worse

    encoders = {}

    def encode(value, name):
        le = LabelEncoder()
        le.fit(value)
        encoders[name] = le
        return le.transform(value)

    def clean_it(x):
        try:
            return x.strip()
        except:
            return x

    for i in data.columns:
        if i in ["Drug", "Age", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema"]:
            data[i] = encode(data[i].apply(clean_it), i)
        if i == "Status":
            data[i] = data[i].apply(lambda x: 1 if x in ["C", "CL"] else 0)

    data.dropna(inplace=True)
    data.isnull().sum()
    x, y = data.iloc[:, [0, 1] + list(range(3, 19))].values, data.iloc[:, 2].values
    split = train_test_split(x, y, test_size=0.3)

    # SVM(split)
    NN(split)


if __name__ == "__main__": main()
