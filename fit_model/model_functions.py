import pandas as pd

def plot_ground_heigth(df):
    groundHeight = df[1464341275]
    groundHeight_reset = groundHeight.reset_index(drop=True)
    groundHeight_reset.plot()
    return


def calculate_maneuver_recall():
    return