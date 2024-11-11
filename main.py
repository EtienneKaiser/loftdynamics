import os
import joblib
import pandas as pd

from utility.constants import CACHE_FILE
from utility.util import get_flight_data, load_json_file, load_parquet_file

flight_data_cache = None

# Done optimization in case of extension of this project
def load_data():
    """Load flight data either from disk or memory."""
    global flight_data_cache

    if os.path.exists(CACHE_FILE):
        print(f"LOADING EXISTING cache from {CACHE_FILE}...")
        flight_data_cache = joblib.load(CACHE_FILE)
        print("Loaded flight names:", list(flight_data_cache.keys()))  # Optionally list loaded flight names

    else:
        print("Cache file not found. CREATING NEW cache... Take a ☕")
        flight_data_cache = get_flight_data()

        print(f"SAVING cache to {CACHE_FILE}...")
        joblib.dump(flight_data_cache, CACHE_FILE)

    return flight_data_cache

if __name__ == '__main__':

    flights = load_data()
    print("Loaded all flight data successfully.")
    
    #TODO get the data you need
    # here's an example
    for flight_name, paths in flights.items():
        print(f"\nData for flight '{flight_name}':")

        json_df = pd.DataFrame(load_json_file(paths['json_path']))
        parquet_df = load_parquet_file(paths['parquet_path'])

        print("JSON DataFrame:")
        print(json_df.head())

        print("\nParquet DataFrame:")
        print(parquet_df.head())
