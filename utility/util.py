import json
import os
import pandas as pd
from joblib import Memory
from tqdm import tqdm

from utility import constants as const

memory = Memory('./cachedir', verbose=0)

def load_json_file(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def load_parquet_file(parquet_path):
    return pd.read_parquet(parquet_path)

def iterate_data():
    files = sorted(os.listdir(const.DATA_DIR))
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

@memory.cache
def assign_meaningful_flight_names(pairs):
    data_dict = {}
    loaded_var_names = []
    idx = 0

    for uuid, pair in tqdm(pairs.items(), desc="Loading flight data", unit="flight"):
        if 'parquet' in pair and 'json' in pair:
            var_name = const.ARR_FLIGHT_NAMES[idx]
            loaded_var_names.append(var_name)
            idx += 1

            parquet_path = os.path.join(const.DATA_DIR, pair['parquet'])
            json_path = os.path.join(const.DATA_DIR, pair['json'])
            data_parquet = load_parquet_file(parquet_path)
            data_json = load_json_file(json_path)

            data_dict[var_name] = {'parquet': data_parquet, 'json': data_json}

            if idx >= len(const.ARR_FLIGHT_NAMES):
                print("Reached variable name limit.")
                break

    print("Loaded data variables:", loaded_var_names)
    return data_dict

def get_flight_data():
    pairs = iterate_data()
    return assign_meaningful_flight_names(pairs)
