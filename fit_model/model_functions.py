import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

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
    y = np.array(df['Label'].reset_index(drop=True).str.lower()) # remove the multi-index
    X = df.drop(columns=['Label'])

    return X, y


def fit_model_with_files(model, files, classes):
    for i, file in enumerate(files):
        X, y = load_and_prepare_data(file)

        # partial fit on each file
        if i == 0:
            model.partial_fit(X, y, classes=classes)
        else:
            model.partial_fit(X, y)
    
    return X.columns


def calculate_maneuver_recall(data_path, flight_id, data, predictions, threshhold = 0.8):
    maneuvers_path = os.path.join(data_path, f"{flight_id}.csv")
    maneuvers = pd.read_csv(maneuvers_path)

    # removing 'No maneuvers' entries
    maneuvers = maneuvers.loc[maneuvers['maneuver'] != "no maneuver"]
    
    maneuvers['startTimestamp'] = pd.to_datetime(maneuvers['startTimestamp'])
    maneuvers['endTimeStamp'] = pd.to_datetime(maneuvers['endTimeStamp'])

    timestamps = np.array(data.index.get_level_values(0))
    timestamp_predictions = list(zip(timestamps, predictions))

    maneuver_scores = {}
    grouped_maneuvers = maneuvers.groupby('maneuver')

    total_detected = 0
    total_maneuvers = len(maneuvers)
    
    for label, current_maneuvers in grouped_maneuvers:
        detected_count = 0
        label = label.lower()

        # Iterate over each maneuver in the ground truth
        for _, maneuver in current_maneuvers.iterrows():
            start_time = maneuver['startTimestamp'].tz_localize("UTC")
            end_time = maneuver['endTimeStamp'].tz_localize("UTC")

            duration_s = (end_time - start_time).total_seconds()
            detection_threshold_s = threshhold * duration_s
            
            maneuver_predictions = [t for t in timestamp_predictions if start_time <= t[0] <= end_time]

            correct_predictions = [p for p in maneuver_predictions if p[1] == label]
            current_datetimes = pd.Series([p[0] for p in maneuver_predictions])
            correct_duration = len(correct_predictions) * (current_datetimes.diff().dt.total_seconds().median())

            if correct_duration >= detection_threshold_s:
                detected_count += 1
        
        total_label_maneuvers = len(maneuvers)
        maneuver_scores[label] = detected_count / total_label_maneuvers if total_label_maneuvers > 0 else 0.0
        total_detected += detected_count
    
    overall_recall = total_detected / total_maneuvers if total_maneuvers > 0 else 0.0
    
    return maneuver_scores, overall_recall