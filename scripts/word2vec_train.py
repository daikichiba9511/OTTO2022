from __future__ import annotations

"""word2vec training for create more candidates

Reference:
1. https://www.kaggle.com/code/radek1/word2vec-how-to-training-and-submission
"""

import glob
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from annoy import AnnoyIndex
from gensim.models import Word2Vec
from gensim.test.utils import common_texts

ROOT = "."
INPUT_DIR = Path(f"{ROOT}/input")
OUTPUT_DIR = Path(f"{ROOT}/output")
TYPE_LABELS = {"clicks": 0, "carts": 1, "orders": 2}


def load_test() -> pl.DataFrame:
    dfs = []
    for e, chunk_file in enumerate(glob.glob(f"{ROOT}/input/otto-chunk-data-in-parquet-format/test_parquet/*")):
        chunk = pl.read_parquet(chunk_file)
        chunk = chunk.with_column((pl.col("ts") / 1000).cast(pl.Int32))
        chunk = chunk.with_column(pl.col("type").apply(lambda x: TYPE_LABELS[x]).cast(pl.UInt8))
        dfs.append(chunk)
    return pl.concat(dfs)


def load_train(is_all: bool) -> pl.DataFrame:
    if is_all:
        dfs = []
        for e, chunk_file in enumerate(glob.glob(f"{ROOT}/input/otto-chunk-data-in-parquet-format/train_parquet/*")):
            chunk = pl.read_parquet(chunk_file)
            chunk = chunk.with_column((pl.col("ts") / 1000).cast(pl.Int32))
            chunk = chunk.with_column(pl.col("type").apply(lambda x: TYPE_LABELS[x]).cast(pl.UInt8))
            dfs.append(chunk)
        return pl.concat(dfs)
    else:
        train = pl.read_parquet(f"{INPUT_DIR}/otto-train-and-test-data-for-local-validation/train.parquet")
        return train


def main() -> None:
    n_jobs = os.cpu_count()
    if n_jobs is None:
        n_jobs = 1
    save_path = OUTPUT_DIR / "w2vec_train"
    save_path.mkdir(parents=True, exist_ok=True)

    train_df = load_train(is_all=True)
    test_df = load_test()
    sentence_df = pl.concat([train_df, test_df]).groupby("session").agg(pl.col("aid").alias("sentence"))
    sentences = sentence_df["sentence"].to_list()

    w2vec = Word2Vec(sentences=sentences, vector_size=32, min_count=1, workers=n_jobs)
    w2vec.save(f"{OUTPUT_DIR}/w2vec_train/w2vec")

    aid2idx = {aid: i for i, aid in enumerate(w2vec.wv.index_to_key)}
    index = AnnoyIndex(f=32, metric="euclidean")
    for _, idx in aid2idx.items():
        index.add_item(idx, w2vec.wv.vectors[idx])
    index.build(n_trees=10, n_jobs=n_jobs)
    index.save(fn=f"{OUTPUT_DIR}/w2vec_train/index.ann")

    session_type = ["clicks", "carts", "orders"]
    test_session_aids = test_df.to_pandas().reset_index(drop=True).groupby("session")["aid"].apply(list)
    test_session_types = test_df.to_pandas().reset_index(drop=True).groupby("session")["type"].apply(list)

    labels = []
    type_weight_multipliers = {0: 1, 1: 6, 2: 3}
    for aids, types in zip(test_session_aids, test_session_types):
        if len(aids) >= 20:
            weights = np.logspace(0.1, 1.0, len(aids), base=2, endpoint=True) - 1
            aids_temp: defaultdict[int, int] = defaultdict(lambda: 0)
            for aid, w, t in zip(aids, weights, types):
                aids_temp[aid] += w * type_weight_multipliers[t]

            sorted_aids = [k for k, v in sorted(aids_temp.items(), key=lambda item: -item[1])]
            labels.append(sorted_aids[:20])
        else:
            aids = list(dict.fromkeys(aids[::-1]))
            most_recent_aid = aids[0]
            nns = [w2vec.wv.index_to_key[i] for i in index.get_nns_by_item(aid2idx[most_recent_aid], 21)[1:]]
            labels.append((aids + nns)[:20])

    labels_as_strings = [" ".join([str(ll) for ll in lls]) for lls in labels]
    predictions = pd.DataFrame(data={"session_type": test_session_aids.index, "labels": labels_as_strings})
    predictions_dfs = []
    for st in session_type:
        modifiled_predictions = predictions.copy()
        modifiled_predictions["session_type"] = modifiled_predictions["session_type"].astype("str") + f"_{st}"
        predictions_dfs.append(modifiled_predictions)
    submission = pd.concat(predictions_dfs).reset_index(drop=True)
    submission.to_csv(save_path / "submission.csv", index=False)


if __name__ == "__main__":
    main()
