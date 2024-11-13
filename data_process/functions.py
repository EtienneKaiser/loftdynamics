import pandas as pd
import json
import os
import numpy as np

def get_flight_ids(data_dir):
    files = os.listdir(data_dir)
    flight_ids = set()
    
    for file in files:
        flight_id = os.path.splitext(file)[0]
        flight_ids.add(flight_id)
    
    return sorted(list(flight_ids))


def iterate_data():
    files = sorted('./data')
    pairs = {}

    for file in files:
        if file.endswith('.parquet') or file.endswith('.json'):
            uuid = file.split('-')[0]
            if uuid not in pairs:
                pairs[uuid] = {}
            if file.endswith('.parquet'):
                pairs[uuid]['parquet'] = file
            elif file.endswith('.json'):
                pairs[uuid]['json'] = file

    return pairs



def load_json_file(json_path):
    # load json file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    df = pd.json_normalize(data, 'annotations', ['flight_session_annotation_id'], errors='ignore')
    return df


def process_json_file(df):
    # timestamp -> datetime 
    df['startTimestamp'] = pd.to_datetime(df['startTimestamp'])
    df['endTimeStamp'] = pd.to_datetime(df['endTimeStamp'])
    
    # overall_grade 
    if 'grading.overall' in df.columns:
        df['overall_grade'] = df['grading.overall']
    else:
        df['overall_grade'] = np.nan
    
    # remove
    df = df.drop(columns=['grading.overall'], errors='ignore')
    df = df.drop(columns=['flight_session_annotation_id'], errors='ignore')
    
    # comment change name to maneuver
    if 'comment' in df.columns:
        df.rename(columns={'comment': 'maneuver'}, inplace=True)
    else:
        df['maneuver'] = 'No Maneuver'  # 如果没有 'comment' 列，则默认所有为 "No Maneuver"
    
    # No Maneuver
    no_maneuver_segments = []
    for i in range(len(df) - 1):
        end_current = df.loc[i, 'endTimeStamp']
        start_next = df.loc[i + 1, 'startTimestamp']

        if start_next > end_current:
            no_maneuver_segments.append({
                'startTimestamp': end_current,
                'endTimeStamp': start_next,
                'maneuver': 'No Maneuver',
                'overall_grade': np.nan  # No grade for 'No Maneuver'
            })

    df_no_maneuver = pd.DataFrame(no_maneuver_segments)
    df = pd.concat([df, df_no_maneuver], ignore_index=True).sort_values(by='startTimestamp').reset_index(drop=True)

    df['duration'] = (df['endTimeStamp'] - df['startTimestamp']).dt.total_seconds()
    df['maneuver_category'] = df['maneuver'].astype('category').cat.codes

    return df

def load_parquet_file(parquet_path):
    df = pd.read_parquet(parquet_path)
    return df

def process_parquet_file(df):

    state_desc_df = pd.read_csv('./state_descriptions_analysis.csv')

    persistent_cols = state_desc_df[state_desc_df['Persistence'] == True]
    persistent_col_names = persistent_cols['Name'].tolist()

    #drop the columns that is persistent
    df = df.drop(columns=[col for col in persistent_col_names if col in df.columns], errors='ignore')

    if 'TimeStamp' in df.columns:
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    
    # if the column is empty, remove it
    df = df.dropna(axis=1, how='all')

    #fill up the none/ nan values
    df = df.fillna(method='ffill')

    return df

def remove_constant_columns(df):
    original_index = df.index
    constant_columns = []

    for col in df.columns:
        # in case of array
        if df[col].apply(lambda x: isinstance(x, (np.ndarray))).any():
            if df[col].apply(lambda x: np.array_equal(x, df[col].iloc[0])).all():
                constant_columns.append(col)
        # in case of 
        elif df[col].nunique() == 1:
            constant_columns.append(col)

    # drop it
    df = df.drop(columns=constant_columns)

    df.index = original_index

    return df



