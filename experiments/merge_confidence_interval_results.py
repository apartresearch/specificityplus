"""
Find all files in the results folder named {metric}_ci.csv or {metric}_ci+.csv,
extract the values for "mean", "0.5%", and "99.5%",
and merge them into csv files of this form:

```csv
,CounterFact,CounterFact+
GPT-2 M,"0.0e+00 (0.0e+00, 0.0e+00)","0.0e+00 (0.0e+00, 0.0e+00)"
FT-L,"1.4e-05 (1.3e-05, 1.4e-05)","1.4e-05 (1.3e-05, 1.4e-05)"
ROME,"1.6e-06 (1.4e-06, 1.7e-06)","2.5e-05 (2.5e-05, 2.5e-05)"
MEMIT,"nan (nan, nan)","nan (nan, nan)"
GPT-2 XL,"0.0e+00 (0.0e+00, 0.0e+00)","0.0e+00 (0.0e+00, 0.0e+00)"
FT-L,"7.2e-06 (6.9e-06, 7.4e-06)","9.5e-06 (9.3e-06, 9.7e-06)"
ROME,"1.5e-06 (1.4e-06, 1.6e-06)","3.3e-05 (3.2e-05, 3.3e-05)"
MEMIT,"2.9e-07 (2.5e-07, 3.4e-07)","9.0e-06 (8.8e-06, 9.1e-06)"
GPT-J (6B),"0.0e+00 (0.0e+00, 0.0e+00)","0.0e+00 (0.0e+00, 0.0e+00)"
FT-L,"3.2e-06 (3.1e-06, 3.4e-06)","5.2e-06 (5.1e-06, 5.3e-06)"
ROME,"3.5e-06 (3.2e-06, 3.8e-06)","1.8e-05 (1.8e-05, 1.9e-05)"
MEMIT,"9.2e-07 (8.0e-07, 1.0e-06)","9.9e-06 (9.8e-06, 1.0e-05)"
```
"""
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Tuple, DefaultDict, Dict

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results/combined"

MODELS = ["gpt2-medium", "gpt2-xl", "gpt-j-6B"]
ALGOS = ["FT-L", "ROME", "MEMIT"]
METRICS = ["S", "M", "KL"]
STATS = ["mean", "0.5%", "99.5%"]


def main():
    stat_to_metric_dataset_to_series: Dict[str, DefaultDict[Tuple[str, str], pd.Series]] = {
        stat: defaultdict(partial(pd.Series, dtype=float)) for stat in STATS
    }
    for metric in METRICS:
        for model in MODELS:
            for dataset_suffix in ["", "+"]:
                df = pd.read_csv(RESULTS_DIR / f"{model}/{metric}_ci{dataset_suffix}.csv", index_col=0)
                index_to_insert = (f"N{metric}", f"CounterFact{dataset_suffix}")
                for stat, metric_dataset_to_series in stat_to_metric_dataset_to_series.items():
                    new_values_at_index = df.loc[stat]
                    base_model = list(new_values_at_index.index.difference(ALGOS))
                    new_values_at_index = new_values_at_index.reindex(base_model + ALGOS)

                    metric_dataset_to_series[index_to_insert] = pd.concat([
                        metric_dataset_to_series[index_to_insert],
                        new_values_at_index,
                    ])

    # merge datasets into one dataframe
    for metric in METRICS:
        df_csv = pd.DataFrame()
        df_latex = pd.DataFrame()
        for col in ["CounterFact", "CounterFact+"]:
            format_strs = ('{:.1e}', '{:.1e}') if metric == "KL" else ('{:.2f}', '{:.3f}')
            mean = stat_to_metric_dataset_to_series["mean"][(f"N{metric}", col)].dropna().map(format_strs[0].format)
            lower = stat_to_metric_dataset_to_series["0.5%"][(f"N{metric}", col)].dropna().map(format_strs[1].format)
            upper = stat_to_metric_dataset_to_series["99.5%"][(f"N{metric}", col)].dropna().map(format_strs[1].format)
            df_csv[col] = mean + " (" + lower + ", " + upper + ")"
            df_latex[col] = mean + " \confint{" + lower + "}{" + upper + "}"  # use with custom latex command \confint

        df_csv.to_csv(RESULTS_DIR / f"N{metric}_from_bootstrap.csv")

        # write latex table
        with open(RESULTS_DIR / f"N{metric}_from_bootstrap.tex", "w") as f:
            f.write(df_latex.to_latex(escape=False))

        # reformat latex table for easier copy-pasting
        with open(RESULTS_DIR / f"N{metric}_from_bootstrap.tex", "r") as f:
            lines = f.readlines()
        # drop lines delimiting the tabular environment
        lines = [line for line in lines if not line.startswith("\\begin") and not line.startswith("\\end")]

        # surround every line starting with "GPT" with \midrule
        with open(RESULTS_DIR / f"N{metric}_from_bootstrap.tex", "w") as f:
            f.write(lines[0])
            for prev, next in zip(lines, lines[1:]):
                if next.startswith("\\midrule"):
                    continue
                if next.startswith("GPT"):
                    next = f"\\midrule\\midrule\n{next}\\midrule\n"
                f.write(next)


if __name__ == '__main__':
    main()
