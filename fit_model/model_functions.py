import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_ground_heigth(df):
    groundHeight = df['1464341275']
    groundHeight_reset = groundHeight.reset_index(drop=True)
    groundHeight_reset.plot()
    return


def plot_alpha_scores(alpha_scores):
    alphas, scores = zip(*alpha_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(alphas, scores, marker='o', linestyle='-', color='b')
    plt.xscale('log')
    plt.xlabel('Alpha (Regularization Strength)')
    plt.ylabel('Cross-Validation Score')
    plt.title('Alpha vs. Cross-Validation Score')
    plt.grid(True)
    plt.show()
    return


def load_and_prepare_data(file_path):
    df = pd.read_parquet(file_path)
    y = np.array(df['Label'].reset_index(drop=True)) # remove the multi-index
    X = df.drop(columns=['Label'])

    return X, y


def fit_model_with_files(model, files, classes):
    for i, file in enumerate(files):
        X, y = load_and_prepare_data(file)

        # partial fit on each file
        if i == 0:
            model.partial_fit(X, y, classes=classes)
            #first_file = False
        else:
            model.partial_fit(X, y)
    
    return X.columns


def calculate_maneuver_recall():
    return