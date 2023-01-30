from pathlib import Path

import yaml

with open("globals.yml", "r") as stream:
    data = yaml.safe_load(stream)

(RESULTS_DIR, DATA_DIR, STATS_DIR, HPARAMS_DIR, KV_DIR) = (
    Path(data["ROOT_DIR"]) / Path(z)
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
