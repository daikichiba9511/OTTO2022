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
# # Co-visitation matrix - simplified, impoved logic
#
# - [1]のNotebookを参考. baseは[2]でそれに[3]でいわれてるtest dataのleakと新しいロジックを追加したもの
# - datasetは最適化したものを使ってる[4]
#
# ## Reference
#
# [1] [co-visitation matrix - simplified, imprvd logic 🔥, @radek1](https://www.kaggle.com/code/radek1/co-visitation-matrix-simplified-imprvd-logic)
#
# [2] [Co-visitation Matrix, @vslaykovsky](https://www.kaggle.com/code/vslaykovsky/co-visitation-matrix)
#
# [3] [Test Data Leak - LB Boost, @cdeotte](https://www.kaggle.com/code/cdeotte/test-data-leak-lb-boost)
#
# [4] [Full dataset processed to CSV/parquet files with optimized memory footprint](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363843)


# %%
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle5 as pickle
from beartype import beartype
from tqdm import tqdm


@dataclass(frozen=True)
class Data:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    id2type: dict[int, str]
    type2id: dict[str, int]
    sample_sub: pd.DataFrame


def load_data(root: Path = Path("./")) -> Data:
    train = pd.read_parquet(f"{root}/input/otto-full-optimized-memory-footprint/train.parquet")  # type: ignore
    test = pd.read_parquet(f"{root}/input/otto-full-optimized-memory-footprint/test.parquet")

    with open(f"{root}/input/otto-full-optimized-memory-footprint/id2type.pkl", "rb") as fh:  # type: ignore
        id2type = pickle.load(fh)
    with open(f"{root}/input/otto-full-optimized-memory-footprint/type2id.pkl", "rb") as fh:
        type2id = pickle.load(fh)

    sample_sub = pd.read_csv(f"{root}/input/otto-recommender-system/sample_submission.csv")

    assert isinstance(train, pd.DataFrame) and isinstance(test, pd.DataFrame)
    return Data(train_df=train, test_df=test, id2type=id2type, type2id=type2id, sample_sub=sample_sub)


data = load_data()
train = data.train_df
test = data.test_df
id2type = data.id2type
type2id = data.type2id
sample_sub = data.sample_sub

# %%
fraction_of_sessions_to_use = 1.0


# %%
class PreprocessModule:
    @staticmethod
    @beartype
    def optimize_train_df(train: pd.DataFrame) -> None:
        train.loc[:, "session"] = train.loc[:, "session"].astype(np.int32)
        train.loc[:, "aid"] = train.loc[:, "aid"].astype(np.int32)
        train.loc[:, "ts"] = train.loc[:, "ts"].astype(np.int32)

    @staticmethod
    @beartype
    def _sampled_df_by_session(df: pd.DataFrame, sampling_ratio: float) -> pd.DataFrame:
        dropped_duplicated_df = df.drop_duplicates(["session"])
        if dropped_duplicated_df is None:
            return df
        remained_df_session = dropped_duplicated_df.sample(frac=sampling_ratio)["session"]
        sub_df = df.loc[df.loc[:, "session"].isin(remained_df_session), :]
        return sub_df

    @staticmethod
    @beartype
    def preprocess(df: pd.DataFrame, fraction_of_sessions_to_use: float) -> pd.DataFrame:
        if fraction_of_sessions_to_use < 0.0 or fraction_of_sessions_to_use > 1.0:
            raise ValueError(f"invalid range")
        subset_of_df = PreprocessModule._sampled_df_by_session(df=df, sampling_ratio=fraction_of_sessions_to_use)
        subset_of_df.index = pd.MultiIndex.from_frame(subset_of_df[["session"]])
        assert isinstance(subset_of_df, pd.DataFrame)
        return subset_of_df


PreprocessModule.optimize_train_df(train)
subset_of_train = PreprocessModule.preprocess(df=train, fraction_of_sessions_to_use=fraction_of_sessions_to_use)
subset_of_test = PreprocessModule.preprocess(df=test, fraction_of_sessions_to_use=fraction_of_sessions_to_use)

del train

# %%
print(subset_of_train.head())

# %%

next_aids = defaultdict(Counter)

chunk_size = 30_000
uni_sessions = subset_of_train.loc[:, "session"].unique()
for i in tqdm(range(0, len(uni_sessions), chunk_size)):
    consecuitive_aids = subset_of_train.loc[
        uni_sessions[i] : uni_sessions[min(len(uni_sessions) - 1, i + chunk_size - 1)], :
    ].reset_index(drop=True)
    consecuitive_aids = consecuitive_aids.groupby("session").apply(lambda g: g.tail(30)).reset_index(drop=True)
    consecuitive_aids = consecuitive_aids.merge(consecuitive_aids, on="session")
    consecuitive_aids = consecuitive_aids[consecuitive_aids["aid_x"] != consecuitive_aids["aid_y"]]
    consecuitive_aids.loc[:, "days_elapsed"] = (consecuitive_aids["ts_y"] - consecuitive_aids["ts_x"]) / (24 * 60 * 60)
    consecuitive_aids = consecuitive_aids[
        (consecuitive_aids["days_elapsed"] > 0) & (consecuitive_aids["days_elapsed"] <= 1)
    ]
    for row in consecuitive_aids.drop_duplicates(["session", "aid_x", "aid_y"]).itertuples():
        next_aids[row.aid_x][row.aid_y] += 1

uni_sessions = subset_of_test.loc[:, "session"].unique()
for i in tqdm(range(0, len(uni_sessions), chunk_size)):
    consecuitive_aids = subset_of_test.loc[
        uni_sessions[i] : uni_sessions[min(len(uni_sessions) - 1, i + chunk_size - 1)]
    ].reset_index(drop=True)
    consecuitive_aids = consecuitive_aids.groupby("session").apply(lambda g: g.tail(30)).reset_index(drop=True)
    consecuitive_aids = consecuitive_aids.merge(consecuitive_aids, on="session")
    consecuitive_aids = consecuitive_aids[consecuitive_aids["aid_x"] != consecuitive_aids["aid_y"]]
    consecuitive_aids.loc[:, "days_elapsed"] = (consecuitive_aids["ts_y"] - consecuitive_aids["ts_x"]) / (24 * 60 * 60)
    consecuitive_aids = consecuitive_aids[
        (consecuitive_aids["days_elapsed"] > 0) & (consecuitive_aids["days_elapsed"] <= 1)
    ]
    for row in consecuitive_aids.drop_duplicates(["session", "aid_x", "aid_y"]).itertuples():
        next_aids[row.aid_x][row.aid_y] += 1


# %%

len(next_aids)

# %%
# %%time

test_session_aids = test.groupby("session")["aid"].apply(list)
session_types = ["clicks", "carts", "orders"]
print(test_session_aids.head())


# %%
type(test_session_aids)

# %%
# %%time


@beartype
def extracted_unique_preserved_order_labels(
    test_session_aids: pd.Series, next_aids: dict[np.int32, Counter]
) -> list[list[np.int32]]:
    if not isinstance(test_session_aids, pd.Series):
        raise ValueError()

    labels = []
    for aids in test_session_aids:
        unique_preserved_order_aids: list[np.int32] = list(dict.fromkeys(aids[::-1]))
        if len(unique_preserved_order_aids) >= 20:
            labels.append(unique_preserved_order_aids[:20])
        else:
            counter = Counter()

            for aid in unique_preserved_order_aids:
                subsequent_aid_counter = next_aids.get(aid)
                if subsequent_aid_counter is not None:
                    counter += subsequent_aid_counter

            unique_preserved_order_aids += [
                aid for aid, cnt in counter.most_common(40) if aid not in unique_preserved_order_aids
            ]
            labels.append(unique_preserved_order_aids[:20])

    assert isinstance(labels, list)
    return labels


if isinstance(test_session_aids, pd.DataFrame):
    raise TypeError()

labels = extracted_unique_preserved_order_labels(test_session_aids, next_aids=next_aids)

# %%

plt.hist([len(l) for l in labels])
plt.show()

# %%


# %%

labels_as_strings = [" ".join(map(str, l)) for l in labels]
predictions = pd.DataFrame({"session_type": test_session_aids.index, "labels": labels_as_strings})

prediction_dfs = []
for st in session_types:
    modified_predictions = predictions.copy()
    modified_predictions.loc[:, "session_type"] = modified_predictions["session_type"].astype("str") + f"_{st}"
    prediction_dfs.append(modified_predictions)

submission = pd.concat(prediction_dfs).reset_index(drop=True)
print(submission.head())
submission.to_csv("./output/submission.csv", index=False)
