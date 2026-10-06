# Working on an experiment

Install with `python -m pip install -e .` and run `python -m unittest discover -s tests -v`.

Before changing an estimator, state what quantity it estimates, why it is unbiased or consistent, what counts as one independent observation, and which cost is included in comparisons. Add a numerical check against a known limit or independent reference rather than a test that repeats the implementation.

Keep pilot and evaluation random streams separate. Preserve existing result folders and write new experiments to a named output directory. Record every case in a prespecified grid, including failures. Inspect coverage and absolute error as well as variance ratios.

Commit methodological changes separately from benchmark outputs. A result folder's metadata stores parameters, seeds and source hashes. Documentation should distinguish measured results, analytical calculations, approximations and possible future work.
