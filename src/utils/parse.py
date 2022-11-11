import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # must
    parser.add_argument("config", type=str, help="config path")

    # about experiments
    parser.add_argument("--train_fold", default=-1, type=int, nargs="*")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--exp", action="store_true")

    # about tpu
    parser.add_argument("--tpu")
    parser.add_argument("--tpu_cores", default=-10, type=int)

    args = parser.parse_args()
    return args
