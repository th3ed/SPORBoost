import requests
import pandas as pd
import yaml
import numpy as np
from dask import delayed, compute
from dask.distributed import LocalCluster, Client

def save_all_datasets(yaml_path, out_path):
    # Read in the data
    meta = parse_dataset_metadata(yaml_path)

    @delayed
    def process_data(name, out_path, url, columns, dtype, missing_ind):
        # Get the dataset
        df = get_dataset(url, columns, dtype, missing_ind)

        # Write the data locally along with it's metadata to read later
        df.to_parquet(f"{out_path}/{name}.parquet", engine='pyarrow', compression='snappy')

    tasks = [process_data(name=name, out_path=out_path, **info) for name, info in meta.items()]
    _ = compute(tasks)

def parse_dataset_metadata(path):
    # Read yaml
    dfs = yaml.safe_load(open(path, 'rb'))

    # Structure dictionary so we can read this data in later
    out = {}
    for df, meta in dfs.items():
        out[df] = {
            'url' : meta['url'],
            'missing_ind' : meta['missing_ind'],
            'columns' : list(meta['columns'].keys()),
            'dtype' : meta['columns']
        }

    return out

def get_dataset(url, columns, dtype, missing_ind):
    # Grab data from web
    data = requests.get(url).content

    # Decode the data, split on rows, remove leading/trailing whitespace, and then finally split on cols
    rows = [l.strip().split(" ") for l in data.decode('utf-8').split('\n') if l.strip() != ""]

    # Convert to a pandas dataframe
    dtype_mapping = {
        'ordinal' : 'float32',
        'target' : 'category',
        'continuous' : 'float32',
        "category" : "category"
    }
    # Validate we only have valid input types
    input_dtypes = set(dtype.values())
    bad_dtypes = input_dtypes - dtype_mapping.keys()
    if len(bad_dtypes) > 0:
        raise ValueError(f"Bad dtypes found in schema: {bad_dtypes}")
    pd_dtypes = {col : dtype_mapping[t] for col, t in dtype.items()}
    
    # Convert to a pandas dataframe with appropriate schema
    df = pd.DataFrame(rows, columns = columns) \
    .replace(missing_ind, np.nan) \
    .astype(pd_dtypes)
    
    return df

if __name__ == '__main__':
    # Set up a dask cluster so we can process in parallel
    cluster = LocalCluster()
    client = Client(cluster)

    # Process all datasets given the expected yaml file
    save_all_datasets("datasets.yaml", "data")