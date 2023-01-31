"""
Script for running an experiment end-to-end on a given test dataset batch
  (for all edit algorithms and the unedited model).

What it does:
1) RUN:
    - iterate over all selected models (plus the unedited model)
        - run model on all test samples
        - store output on hard disk
2) SCORE:
    - load output for unedited model from HD
    - iterate over all selected models
        - load output from HD
        - compute scores from edited model output plus unedited model output
        - collect scores
    - store results for all models (incl. unedited model) in csv files on hard disk
"""
from typing import Union


from experiments.evaluate import main as run


def score():
    # TODO: implement
    pass


def main(
        alg_names: list[str],
        model_name: Union[str, tuple],
        hparams_fnames: list[str],
        ds_name: str,
        dataset_size_limit: int,
        skip_generation_tests: bool,
        generation_test_interval: int,
        conserve_memory: bool,
        num_edits: int = 1,
        use_cache: bool = False,
        verbose: bool = False,
        start_index: int = 0
):
    assert "IDENTITY" in alg_names, "must also run unedited model (include 'IDENTITY')"
    assert len(alg_names) == len(hparams_fnames), "..."

    for alg_name, hparams_fname in zip(alg_names, hparams_fnames):
        run(
            alg_name=alg_name,
            model_name=model_name,
            hparams_fname=hparams_fname,
            ds_name=ds_name,
            dataset_size_limit=dataset_size_limit,
            skip_generation_tests=skip_generation_tests,
            generation_test_interval=generation_test_interval,
            conserve_memory=conserve_memory,
            dir_name=alg_name,
            num_edits=num_edits,
            use_cache=use_cache,
            verbose=verbose,
            start_index=start_index,
        )

    for alg_name, hparams_fname in zip(alg_names, hparams_fnames):
        score()


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_names", # todo: fix CLI arg
        choices=["MEMIT", "ROME", "FT", "MEND", "IDENTITY"],
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<model_name>/<run_id>, "
             "where the run_id is of the form 'run_<start_index>_<dataset_size_limit>'. ",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        choices=["gpt2-medium", "gpt2-large", "gpt2-xl", "EleutherAI/gpt-j-6B", "EleutherAI/gpt-neox-20b"],
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fnames", # todo: fix CLI arg
        type=str,
        default="gpt2-xl.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["mcf", "cf", "zsre"],
        default="mcf",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to n records.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start index for the dataset. Useful for parallelizing runs.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
             "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
             "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Also print detailed information about the running edit algorithm",
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False, verbose=False)
    args = parser.parse_args()

    main(
        alg_names=args.alg_names,
        model_name=args.model_name,
        hparams_fnames=args.hparams_fnames,
        ds_name=args.ds_name,
        dataset_size_limit=args.dataset_size_limit,
        skip_generation_tests=args.skip_generation_tests,
        generation_test_interval=args.generation_test_interval,
        conserve_memory=args.conserve_memory,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
        verbose=args.verbose,
        start_index=args.start_index
    )
