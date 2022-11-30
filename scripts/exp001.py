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
# - validationの追加[5]
#
# ## Reference
#
# 1. [co-visitation matrix - simplified, imprvd logic 🔥, @radek1](https://www.kaggle.com/code/radek1/co-visitation-matrix-simplified-imprvd-logic)
#
# 2. [Co-visitation Matrix, @vslaykovsky](https://www.kaggle.com/code/vslaykovsky/co-visitation-matrix)
#
# 3. [Test Data Leak - LB Boost, @cdeotte](https://www.kaggle.com/code/cdeotte/test-data-leak-lb-boost)
#
# 4. [Full dataset processed to CSV/parquet files with optimized memory footprint](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363843)
#
# 5. [local validation tracks public LB perfecty -- here is the setup](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364991)
#
# 6. [Compute Validation Score - [CV 563] ](https://www.kaggle.com/code/cdeotte/compute-validation-score-cv-563)


# %%
from __future__ import annotations

import csv
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle5 as pickle
from beartype import beartype
from tqdm import tqdm


class UtilsModule:
    @staticmethod
    def timeit(func):
        """
        Reference
        1. https://dev.to/kcdchennai/python-decorator-to-measure-execution-time-54hk
        """

        @wraps(func)
        def timeit_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            total_time = end_time - start_time
            print(f"Function {func.__name__} Took {total_time:.4f} seconds")
            return result

        return timeit_wrapper


@dataclass(frozen=True)
class Config:
    exp_name: str = "exp001"
    debug: bool = __debug__ if __debug__ is not None else True
    seed: int = 42


@dataclass(frozen=True)
class Data:
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df: pd.DataFrame
    test_labels: pd.DataFrame
    id2type: dict[int, str]
    type2id: dict[str, int]
    sample_sub: pd.DataFrame


LABEL2IDS = {
    "clicks": 0,
    "carts": 1,
    "orders": 2,
}

IDS2LABEL = {
    0: "clicks",
    1: "carts",
    2: "orders",
}


def load_data(root: Path = Path("./"), debug: bool = False) -> Data:
    # dataset only for validation from https://www.kaggle.com/datasets/radek1/otto-train-and-test-data-for-local-validation
    # 'ts' columns data were devided by 1000, because the creator thinks 'ts' columns is enough orders of seconds.
    # So, If you need, must restore 'ts' columns data by multiplying 1000
    # Ref: 5.
    # TODO: when I make final submit, change this directory to original dataset / otto-full-optimized-memory-footprint

    # train = pd.read_parquet(f"{root}/input/otto-validation/train.parquet")  # type: ignore
    # valid = pd.read_parquet(f"{root}/input/otto-validation/test.parquet")

    @beartype
    def _load_train() -> pd.DataFrame:
        train_dfs = []
        for e, chunk_file in enumerate(Path(f"{root}/input/otto-validation/train_parquet").glob("*")):
            chunk = pd.read_parquet(chunk_file)
            chunk["type"] = chunk["type"].map(lambda x: LABEL2IDS[x])
            train_dfs.append(chunk)
        # return pd.concat(train_dfs).reset_index(drop=True).astype({"ts": "datetime64[ms]"})
        return pd.concat(train_dfs).reset_index(drop=True)

    @beartype
    def _load_valid() -> pd.DataFrame:
        valid_dfs = []
        for e, chunk_file in enumerate(Path(f"{root}/input/otto-validation/test_parquet").glob("*")):
            chunk = pd.read_parquet(chunk_file)
            chunk["type"] = chunk["type"].map(lambda x: LABEL2IDS[x])
            valid_dfs.append(chunk)
        # return pd.concat(valid_dfs).reset_index(drop=True).astype({"ts": "datetime64[ms]"})
        return pd.concat(valid_dfs).reset_index(drop=True)

    @beartype
    def _load_test() -> pd.DataFrame:
        test = pd.read_parquet(f"{root}/input/otto-full-optimized-memory-footprint/test.parquet")
        test = test.reset_index(drop=True)
        # test["type"] = test["type"].map(lambda x: IDS2LABEL[x])
        # test = test.astype({"ts": "datetime64[ms]", "type": "int64"})
        test = test.astype({"type": "int64"})
        return test

    @beartype
    def _load_test_labels() -> pd.DataFrame:
        """validationの評価用のdata"""
        test_labels = pd.read_parquet(f"{root}/input/otto-validation/test_labels.parquet")
        return test_labels

    train = _load_train()
    valid = _load_valid()
    valid_labels = _load_test_labels()
    test = _load_test()

    sample_sub = pd.read_csv(f"{root}/input/otto-recommender-system/sample_submission.csv")

    assert isinstance(train, pd.DataFrame) and isinstance(test, pd.DataFrame)
    return Data(
        train_df=train,
        valid_df=valid,
        test_labels=valid_labels,
        test_df=test,
        id2type=IDS2LABEL,
        type2id=LABEL2IDS,
        sample_sub=sample_sub,
    )


data = load_data(debug=Config.debug)
train = data.train_df
valid = data.valid_df
test = data.test_df
id2type = data.id2type
type2id = data.type2id
sample_sub = data.sample_sub

print(f"{'#'*20} {Config.exp_name}: debug mode: {Config.debug} {'#'*20}")
print("train = \n", train.head())
print("valid = \n", valid.head())
print("test = \n", test.head())
print("-------- info ----------")
print("train.info() = \n", train.info())
print("valid.info() = \n", valid.info())
print("test.info() = \n", test.info())

# %%


# %%
class PreprocessModule:
    @staticmethod
    @beartype
    def optimize_train_df(train: pd.DataFrame) -> None:
        train.loc[:, "session"] = train.loc[:, "session"].astype(np.int32)
        train.loc[:, "aid"] = train.loc[:, "aid"].astype(np.int32)
        # train.loc[:, "ts"] = train.loc[:, "ts"].astype(np.int32)

    @staticmethod
    @beartype
    def _sampled_df_by_session(df: pd.DataFrame, sampling_ratio: float, seed: int = 42) -> pd.DataFrame:
        if sampling_ratio < 0.0 or sampling_ratio > 1.0:
            raise ValueError(f"Invalid Value: {sampling_ratio}")

        dropped_duplicated_df = df.drop_duplicates(["session"])
        if dropped_duplicated_df is None:
            return df

        remained_df_session = dropped_duplicated_df.sample(frac=sampling_ratio, random_state=seed)["session"]
        sub_df = df.loc[df.loc[:, "session"].isin(remained_df_session), :]
        return sub_df

    @staticmethod
    @beartype
    def preprocess(df: pd.DataFrame, fraction_of_sessions_to_use: float) -> pd.DataFrame:
        if fraction_of_sessions_to_use < 0.0 or fraction_of_sessions_to_use > 1.0:
            raise ValueError("invalid range")

        subset_of_df = PreprocessModule._sampled_df_by_session(df=df, sampling_ratio=fraction_of_sessions_to_use)
        subset_of_df.index = pd.MultiIndex.from_frame(subset_of_df[["session"]])
        return subset_of_df.reset_index(drop=True)


print("------------ preprocess ----------- ")
fraction_of_sessions_to_use = 1.0 if not Config.debug else 0.01
PreprocessModule.optimize_train_df(train)
subset_of_train = PreprocessModule.preprocess(df=train, fraction_of_sessions_to_use=fraction_of_sessions_to_use)
subset_of_valid = PreprocessModule.preprocess(df=valid, fraction_of_sessions_to_use=fraction_of_sessions_to_use)
subset_of_test = PreprocessModule.preprocess(df=test, fraction_of_sessions_to_use=fraction_of_sessions_to_use)

del train

# %%
print(subset_of_train.head())

# %%

# {"aid_x": {"aid_c": count}}
# aid_xに対してaid_cが何回起こってるか -> co-occurance
next_aids = defaultdict(Counter)

chunk_size = 30_000

print(" --- processing train --- ")
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

print(" --- processing valid --- ")
uni_sessions = subset_of_valid.loc[:, "session"].unique()
for i in tqdm(range(0, len(uni_sessions), chunk_size)):
    consecuitive_aids = subset_of_valid.loc[
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

# NOTE:
# only 'test' is not same data source, so 'ts' column was devided by 1000. so, you need to multiply 1000
# <-> but, you don't need to do train and valid
# Referrence
# 1. https://www.kaggle.com/competitions/otto-recommender-system/discussion/364991#2023514
# 2. https://www.kaggle.com/code/radek1/a-robust-local-validation-framework
print(" --- processing test --- ")
# Procedure.
# 1. chunkにわける
# 2. session毎に最後の30個の履歴を最後尾から取ってきてindexをreset
# 3. それらをsession毎にmerge
# 4. aid_x != aid_yのものを取ってくる
# 5. ts_yとts_xが0日以上1日以下の履歴を抽出する
uni_sessions = subset_of_test.loc[:, "session"].unique()
for i in tqdm(range(0, len(uni_sessions), chunk_size)):
    current_chunk = subset_of_test.loc[
        uni_sessions[i] : uni_sessions[min(len(uni_sessions) - 1, i + chunk_size - 1)]
    ].reset_index(drop=True)
    current_chunk = current_chunk.groupby("session").apply(lambda g: g.tail(30)).reset_index(drop=True)
    consecuitive_aids = current_chunk.merge(current_chunk, on="session")
    consecuitive_aids = consecuitive_aids[consecuitive_aids["aid_x"] != consecuitive_aids["aid_y"]]
    consecuitive_aids["days_elapsed"] = (consecuitive_aids["ts_y"] - consecuitive_aids["ts_x"]) / (24 * 60 * 60 * 1000)
    consecuitive_aids = consecuitive_aids[
        (consecuitive_aids["days_elapsed"] > 0) & (consecuitive_aids["days_elapsed"] <= 1)
    ]
    for row in consecuitive_aids.drop_duplicates(["session", "aid_x", "aid_y"]).itertuples():
        next_aids[row.aid_x][row.aid_y] += 1


# %%

print(f"{len(next_aids) = }")

# %%
# %%time

valid_session_aids = valid.groupby("session")["aid"].apply(list)
test_session_aids = test.groupby("session")["aid"].apply(list)
print(test_session_aids.head())


# %%
type(test_session_aids)

# %%
# %%time


class SuggestModule:
    @staticmethod
    @beartype
    def extracted_unique_preserved_order_labels(
        test_session_aids: pd.Series, next_aids: dict[np.int32, Counter]
    ) -> list[list[int]]:
        """suggest items"""

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

        return labels


if isinstance(test_session_aids, pd.DataFrame):
    raise TypeError()

if isinstance(valid_session_aids, pd.DataFrame):
    raise TypeError()

valid_labels = SuggestModule.extracted_unique_preserved_order_labels(valid_session_aids, next_aids=next_aids)
labels = SuggestModule.extracted_unique_preserved_order_labels(test_session_aids, next_aids=next_aids)

# %%

plt.hist([len(l) for l in labels])
plt.show()


# %%
class MetricsModule:
    @staticmethod
    @UtilsModule.timeit
    @beartype
    def computed_metric(submission_df: pd.DataFrame, gt_df: pd.DataFrame) -> float:
        print(" ----------- start to computation of metrics ------------ \n")
        if "session_type" not in submission_df.columns or "labels" not in submission_df.columns:
            raise ValueError(f"invalid columns in submission_df: {submission_df.columns}")

        # if "aid" not in gt_df.columns:
        #     raise ValueError(f"invalid columns in gt_df: {gt_df.columns}")
        score = 0
        weights = {"clicks": 0.10, "carts": 0.30, "orders": 0.60}
        for t in weights.keys():
            sub = submission_df.loc[submission_df["session_type"].str.contains(t)].copy()
            sub["session"] = sub["session_type"].apply(lambda x: int(x.split("_")[0]))
            sub["labels"] = sub["labels"].apply(lambda x: [int(i) for i in x.split()[:20]])

            # -- gt label
            test_labels = gt_df.loc[gt_df["type"] == t]

            test_labels = test_labels.merge(sub, how="left", on=["session"])
            test_labels["hits"] = test_labels.apply(
                lambda df: len(set(df["ground_truth"]).intersection(set(df["labels"]))), axis=1
            )
            test_labels["gt_count"] = test_labels["ground_truth"].str.len().clip(0, 20)
            recall = test_labels["hits"].sum() / test_labels["gt_count"].sum()
            score += weights[t] * recall
            print(f"{t} recall = {recall}")

        assert 0.0 <= score <= 1.0, f"invalid score: {score}"
        return score


# %%


@beartype
def make_predictions(session_aids: pd.Series, labels: list, session_types: list[str]) -> pd.DataFrame:
    labels_as_strings = [" ".join(map(str, l)) for l in labels]
    predictions = pd.DataFrame({"session_type": session_aids.index, "labels": labels_as_strings})

    prediction_dfs = []
    for st in session_types:
        modified_predictions = predictions.copy()
        modified_predictions.loc[:, "session_type"] = modified_predictions["session_type"].astype("str") + f"_{st}"
        prediction_dfs.append(modified_predictions)

    submission = pd.concat(prediction_dfs).reset_index(drop=True)
    return submission


session_types = ["clicks", "carts", "orders"]
submission = make_predictions(session_aids=test_session_aids, labels=labels, session_types=session_types)
valid_submission = make_predictions(session_aids=valid_session_aids, labels=valid_labels, session_types=session_types)

metrics_score = MetricsModule.computed_metric(submission_df=valid_submission, gt_df=data.test_labels)
print("Recall@20 => ", metrics_score)

print(submission.head())
sub_save_path = Path("./output") / Config.exp_name / "submission.csv"
sub_save_path.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(sub_save_path, index=False)
with (sub_save_path.parent / "metrics.csv").open("w") as fp:
    writer = csv.writer(fp, delimiter=",")
    writer.writerow(["Recall@20", metrics_score])
