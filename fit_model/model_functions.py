import pandas as pd
import numpy as np

def plot_ground_heigth(df):
    groundHeight = df[1464341275]
    groundHeight_reset = groundHeight.reset_index(drop=True)
    groundHeight_reset.plot()
    return


def convert_column_to_datetime_utc(df, column):
    datetime_col = pd.to_datetime(df[column])
    if datetime_col.dt.tz is None:
        return datetime_col.dt.tz_localize('UTC')
    else:
        return datetime_col.dt.tz_convert('UTC')


def add_row_label(data, labels):
    data['Label'] = None

    labels['startTimestamp'] = convert_column_to_datetime_utc(labels, 'startTimestamp')
    labels['endTimeStamp'] = convert_column_to_datetime_utc(labels, 'endTimeStamp')

    # calculate the label (maneuver) by the timestamp
    for _, row in labels.iterrows():
        start, end, label = row['startTimestamp'], row['endTimeStamp'], row['maneuver']
        
        idx = (data.index.get_level_values('TimeStamp') >= start) & (data.index.get_level_values('TimeStamp') <= end)
        data.loc[idx, 'Label'] = label
    
    data.loc[data['Label'].isnull(), 'Label'] = "No Maneuver"
    return data


def convert_value(val):
        if isinstance(val, bool):
            return int(val)
        elif isinstance(val, (int, float)):
            return val
        elif isinstance(val, (list, np.ndarray)):  # take the mean of arrays
            return np.mean([v for v in val if isinstance(v, (int, float))])
        elif isinstance(val, str):  # gnore text
            return 0
        else:
            return 0
        

def clean_dataframe(df):
    df_cleaned = df.map(convert_value)
    return df_cleaned.dropna(axis=1)


def partial_fit_SGDClassifier(model, df):
    
    return


def calculate_maneuver_recall():
    return