# Monte Carlo option pricing

I wanted to see how much better you can do than plain Monte Carlo when pricing a European put, and whether the improvement is still worth it once you count the time spent tuning the sampler.

This compares five estimators in Python and NumPy. The main model uses a two-component generalised-normal mixture, which allows asymmetric returns and heavier tails than a Gaussian. There are also Gaussian and stress cases. Numerical integration gives a reference price to check the simulations against.

The main finding: tuned importance sampling reduced variance by about **9.5x** in the default at-the-money case. But the proposal search took enough time to outweigh the gain for a single calculation. Reusing the proposal makes a difference.

[Walkthrough](docs/WALKTHROUGH.md) | [Full results](results/reconstruction-v3/REPORT.md) | [Experiment notes](docs/PROVENANCE.md)

## Try it

You'll need Python 3.10+ and NumPy. From the repository folder:

```sh
python -m pip install .
python -m mcpricing price --scenario mixture --strike 100 --samples 20000
```

This prints the prices, standard errors, runtimes and tuning costs, alongside the integration reference. It also estimates delta: how the price changes with the starting stock price.

For a smaller example with comments:

```sh
python examples/first_experiment.py
```

## What's being compared?

| Method | What it does |
|---|---|
| Direct Monte Carlo (IID) | Draws independent returns and averages the discounted payoffs. |
| Importance sampling (IS) | Samples losses more often, then reweights the payoffs to keep the same pricing target. |
| Antithetic variates (AV) | Pairs opposite shocks around each mixture component's centre. Both payoffs count towards the budget. |
| Control variates (CV) | Uses the discounted stock price, whose expectation is known, to reduce noise in the put estimate. |
| Metropolis-Hastings (MH) | Samples returns with a random walk. Consecutive draws are correlated. |

The IS proposal, CV coefficient and MH step size are tuned on separate pilot samples, then held fixed during evaluation. A drift correction keeps the stock's discounted expectation equal to its starting price.

## Results

The full run contains **13,500 price estimates**: three return distributions, three strikes, three sample budgets, five independently tuned pilot groups, 20 repeats per group and five methods. Raw trials, seeds, pilot choices and environment details are saved in [`results/reconstruction-v3`](results/reconstruction-v3).

Here's the default mixture with spot and strike both 100, using 20,000 payoff evaluations per estimate. The reference price is **4.28997599**.

| Method | RMSE | Variance reduction vs IID, from integration | Mean time per estimate |
|---|---:|---:|---:|
| IID | 0.05346 | 1.00x | 2.59 ms |
| IS | 0.01754 | 9.40x | 6.44 ms |
| AV | 0.05334 | 1.38x | 1.53 ms |
| CV | 0.04017 | 2.41x | 2.70 ms |
| MH | 0.14755 | Not computed | 37.59 ms |

Times exclude tuning. The variance column comes from integrating payoff moments for each fixed pilot, averaged across the five groups. MH needs an autocorrelation calculation as well, so it has no entry there. RMSE comes from repeated simulations; the two columns won't line up exactly.

![Convergence under the default mixture](results/reconstruction-v3/mixture-convergence.svg)

IS gave **9.46x lower empirical variance**, close to the **9.40x** integration result. That corresponds to roughly **3.07x lower standard error** at the same sample count. The bootstrap 95% interval for the empirical ratio was wide, **5.16-19.47x**; with only five pilot groups, it's an exploratory check.

Runtime changes the picture. Measured by variance times runtime, IS was **3.81x as efficient** as IID before tuning costs. Charging the full **28.36 ms** proposal search to one estimate brought that ratio down to **0.70x**. These numbers depend on the model, strike and machine.

One useful failure came from the first MH implementation. Choosing a step size by low pilot payoff variance could favour a chain that barely visited the loss region. It looked stable because it missed the events that mattered. Tuning by expected squared jump distance improved exploration: at strike 80 in the default mixture, with 20,000 samples, RMSE fell from **0.1165 to 0.0506** and 95% interval coverage rose from **89% to 96%**. Both rounds are kept in the repo; the [experiment notes](docs/PROVENANCE.md) explain the change.

## Run it yourself

For a quick benchmark:

```sh
python -m mcpricing benchmark --scenarios mixture --strikes 100 --sizes 1000 5000 --groups 2 --repeats 5 --pilot-samples 1000 --bootstrap 200 --output local-results
```

For the full experiment:

```sh
python -m mcpricing benchmark --output my-full-run
```

Each run saves trial data, summary tables, pilot settings, plots and a standalone `report.html`. Use a fresh output folder each time. To rebuild a report from saved data:

```sh
python -m mcpricing report my-full-run
```

The recorded runs used Python 3.12.14 and NumPy 2.3.5. To use the pinned NumPy version:

```sh
python -m pip install -r requirements-reproduce.txt
python -m pip install --no-deps .
```

## Checks and limits

```sh
python -m unittest discover -s tests -v
```

All 16 tests passed locally on Windows with Python 3.12.14. They cover the Black-Scholes limit, distribution moments, discounted-stock expectation, put-call parity, estimator calculations, numerical refinement, delta and benchmark reproducibility. The package was also built and installed separately from the source folder. GitHub Actions is configured for Linux, Windows and macOS; see the workflow runs for hosted results.

These are synthetic, single-maturity models. They aren't fitted to market data, and changing maturity doesn't rescale the return distribution. For this one-dimensional put, numerical integration is cheaper than simulation; the point is to study the estimators. Path-dependent options would need a model for the whole price path.

MH's batch-means intervals sometimes under-cover. Its Python loop also makes runtime comparisons with the vectorised methods partly a comparison of implementations.

There's an optional Docker setup, which hasn't been tested locally:

```sh
docker build -t mcpricing .
docker run --rm mcpricing price --samples 20000
```

## Finding your way around

Start with [`examples/first_experiment.py`](examples/first_experiment.py). The distribution and payoff are in `mcpricing/model.py`, the five methods in `mcpricing/estimators.py`, and the reference variance calculations in `mcpricing/theory.py`. `benchmark.py` runs the experiments; `report.py` makes the tables and plots.

The [walkthrough](docs/WALKTHROUGH.md) covers the derivations and includes exercises. [CONTRIBUTING.md](CONTRIBUTING.md) explains how to add an experiment.

This rebuilds a student project whose original code was lost, with AI assistance on the implementation. The parameters and results come from the rebuilt version; the [experiment notes](docs/PROVENANCE.md) record what changed.
