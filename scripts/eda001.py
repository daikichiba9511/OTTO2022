# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # EDA
#
# ## Reference
#
# [1] [OTTO: Basic EDA, Alifia Ghantiwala ](https://www.kaggle.com/code/aliphya/otto-basic-eda)

# %%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pathlib import Path
from tqdm import tqdm

# %%
#Code reference to read json data into pandas: 
#https://www.kaggle.com/code/edwardcrookenden/otto-getting-started-eda-baseline
TRAIN_PATH = '../input/otto-recommender-system/train.jsonl'
sample_size = 100_000

print(Path(TRAIN_PATH).exists())

parquet_dir = Path("../input/train_parquet")
parquet_dir.mkdir(parents=True, exist_ok=True)

def convert_parquet(parquet_dir: Path) -> None:
    chunks = pd.read_json(TRAIN_PATH, lines=True, orient="records", chunksize = sample_size)
    train_df = []
    for i, c in enumerate(tqdm(chunks)):
        event_dict = {
            "session": [],
            "aid": [],
            "ts": [],
            "type": [],
        }
        for session, events in zip(c["session"].tolist(), c["events"].tolist()):
            for event in events:
                event_dict["session"].append(session)
                event_dict["aid"].append(event["aid"])
                event_dict["ts"].append(event["ts"])
                event_dict["type"].append(event["type"])

        start = str(i * sample_size).zfill(9)
        end = str(i * sample_size + sample_size)
        event_df = pd.DataFrame(event_dict)
        event_df.to_parquet(parquet_dir / f"{start}_{end}.parquet")
        del event_df
    
# convert_parquet(parquet_dir)


# %%
# %%time
files = sorted(Path("../input/train_parquet").glob("*"))
print(len(files))
dfs = []

for path in files[0:10]:
    dfs.append(pd.read_parquet(path))

dfs = pd.concat(dfs).reset_index(drop=True)

# %%
dfs

# %%
session_cnt_df = dfs.groupby("session").apply(len).sort_values(ascending=False)

# %%
session_cnt_df.head()

# %%
session_cnt_df.index

# %%
tmp = session_cnt_df.tolist()
print(len(tmp))

# %%
plt.hist(session_cnt_df.tolist())

# %%
dfs.groupby(["session", "type"])["aid"].shift(-1)

# %%
dfs.iloc[:40, :].head()

# %%

# %%
