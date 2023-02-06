from os import getenv
from pathlib import Path

import yaml

with open("globals.yml", "r") as stream:
    data = yaml.safe_load(stream)

ROOT_DIR = getenv("ROOT_DIR", data["ROOT_DIR"])
if not ROOT_DIR:
    raise ValueError(
        "ROOT_DIR must be set, either as environment variable or in globals.yml"
    )

(RESULTS_DIR, DATA_DIR, STATS_DIR, HPARAMS_DIR, KV_DIR) = (
    Path(ROOT_DIR) / z
    for z in [
        data["RESULTS_DIR"],
        data["DATA_DIR"],
        data["STATS_DIR"],
        data["HPARAMS_DIR"],
        data["KV_DIR"],
    ]
)

REMOTE_ROOT_URL = data["REMOTE_ROOT_URL"]
SEED = data["SEED"]

for var, path in (
        ("RESULTS_DIR", RESULTS_DIR),
        ("DATA_DIR", DATA_DIR),
        ("STATS_DIR", STATS_DIR),
        ("HPARAMS_DIR", HPARAMS_DIR),
        ("KV_DIR", KV_DIR),
):
    print(f"{var}: {path.resolve()}")
