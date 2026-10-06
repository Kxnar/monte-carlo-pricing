# Understanding the project

Start with the price being estimated, then learn why the estimators behave differently. The purpose is to explain and investigate the numerical methods, not claim a profitable strategy.

## 1. A put is insurance against a low terminal price

Suppose a stock is worth 100 today. A one-year European put with strike 100 pays `max(100 - S_T, 0)` at expiry.

| Terminal stock price | Put payoff |
|---:|---:|
| 80 | 20 |
| 95 | 5 |
| 100 | 0 |
| 110 | 0 |
| 130 | 0 |

If these five outcomes were equally likely under the pricing distribution, the expected payoff would be 5. Discounting at a continuously compounded 3% rate gives `5 exp(-0.03)`, or about 4.852. Those five outcomes are just an illustration, not the model used in the benchmark.

**Check:** A put with strike 80 and spot 100 is out of the money. Why do most simulated payoffs become zero? Why does the small set of loss scenarios matter so much?

## 2. The distribution is an assumption, not a prediction

We introduce a random variable X with density

$$p(x)=\sum_{j=1}^{2} w_j\frac{\beta_j}{2\alpha_j\Gamma(1/\beta_j)}\exp\left(-\left|\frac{x-\mu_j}{\alpha_j}\right|^{\beta_j}\right).$$

The component label is sampled with probabilities `w`. The parameters control location, scale and shape. When beta is 2, a component is Gaussian with standard deviation `alpha / sqrt(2)`. A beta below 2 produces a heavier tail than that Gaussian form. Combining a central component with a wider, negatively shifted component gives more weight to adverse returns.

The default mixture has weights `(0.8, 0.2)`, locations `(0.02, -0.08)`, scales `(0.16, 0.30)` and shapes `(2.0, 1.5)`. These are synthetic choices. There is no market-data fitting.

We define a **terminal pricing measure** by

$$\kappa=\log\mathbb E[e^X],\qquad S_T=S_0\exp(rT-\kappa+X).$$

Then

$$\mathbb E[e^{-rT}S_T]=S_0e^{-\kappa}\mathbb E[e^X]=S_0.$$

That equality enforces the discounted stock expectation for this chosen one-period model. The put price is

$$P=\mathbb E[Y],\qquad Y=e^{-rT}(K-S_T)^+.$$

**Crucial distinction:** a historical return distribution is not automatically a pricing distribution. This project specifies the latter directly. Adjusting its mean does not magically calibrate it to traded options, uniquely identify a pricing measure, or construct a consistent continuous-time process. The mixture parameters are a terminal law for one maturity. Varying T does not automatically scale those parameters into a multi-maturity model.

**Check:** Why is subtracting the mean of X insufficient? Because `E[exp(X)]` is generally not `exp(E[X])`.

## 3. How direct samples are generated

For a standard generalised-normal component, draw

$$G\sim\operatorname{Gamma}(1/\beta,1),\quad B\in\{-1,+1\}\text{ with equal probability},\quad X=\mu+\alpha B G^{1/\beta}.$$

The change of variables from G to `G^(1/beta)` gives the desired absolute-value density. Sampling the component first produces an exact mixture draw. This means we do not need a Markov chain for this model.

Read `Mixture.sample` in `mcpricing/model.py`, then run:

```sh
python -m mcpricing price --scenario mixture --strike 100 --samples 20000 --seed 42
```

Different seeds give different estimates. The reference stays at about **4.28997599**. That reference comes from numerical integration.

## 4. Why more samples help slowly

For independent observations, the ordinary estimator is

$$\widehat P_N=\frac1N\sum_{i=1}^N Y_i,\qquad \operatorname{Var}(\widehat P_N)=\frac{\operatorname{Var}(Y)}N.$$

Standard error is the square root of variance, so it decreases like `1/sqrt(N)`. Four times as many samples roughly halves the sampling error. An approximate 95% interval is `estimate +/- 1.96 x estimated standard error`, relying on a normal approximation that can be poor for very rare payoffs or dependent observations.

The project also evaluates independent estimates against quadrature. This measures actual error rather than trusting an estimator's own reported error bar.

**The old 6.1x number:** it was a ratio of two sample variances from 40 independent estimates each. Ratios of noisy variance estimates fluctuate. For the default at-the-money mixture with the selected importance proposal, numerical integration gives a population ratio around **9.40**. This is a property of that model and proposal, not a universal promise.

If variance falls by 9.40x, standard error falls by `sqrt(9.40)`, about **3.07x**, at the same sample count. At approximately equal error, ordinary Monte Carlo would need 9.40x as many samples. Neither statement proves a 9.40x wall-clock speedup.

**Exercise:** Starting with a standard error of 0.06, what error would a 9.40x variance reduction imply? Approximately 0.0196.

## 5. Importance sampling changes which outcomes get attention

For a put, positive returns often produce zero payoff. Spending more samples in the loss region can be more informative. We draw from a different density q and correct for the changed probabilities:

$$P=\int h(x)p(x)\,dx=\mathbb E_q\left[h(X)\frac{p(X)}{q(X)}\right].$$

Here `h` includes discounting. We use

$$q(x)=0.1p(x)+0.9\widetilde p(x),$$

where the shifted/rescaled mixture `p_tilde` puts more mass in useful regions. The 10% defensive component guarantees `q >= 0.1p`, hence `p/q <= 10`. This prevents missing target support and bounds the weights; it does not by itself guarantee a good proposal.

Twelve candidates are compared using separate pilot samples. The chosen proposal is then frozen before the independent evaluation draws. This avoids selecting a proposal because it happened to look good on the same samples used to report the final result.

The zero-variance ideal would have `q*(x)` proportional to `h(x)p(x)` for this nonnegative payoff. Its normalising constant is the unknown price, so it is a theoretical guide rather than an implemented shortcut.

Read `Proposal`, `importance_values`, `estimate` and `tune`. Notice that this implementation uses ordinary importance weights, not weights divided by their random sample sum.

**Check:** If we oversample crashes but forget `p/q`, what happens? We price under the proposal distribution and usually overestimate the put.

## 6. Antithetic sampling pairs opposing shocks

Inside a sampled component, pair

$$X^+=\mu_j+\alpha_jZ,\qquad X^-=\mu_j-\alpha_jZ.$$

Each marginal has the right component distribution. Average the two payoffs and treat the pair as one independent observation. A budget of 20,000 payoffs gives 10,000 independent pairs. Counting both members as independent would give an incorrect standard error.

Reflecting the whole asymmetric mixture around zero would change its distribution. That is why the reflection is around the selected component's centre.

For a single Gaussian component and monotone payoff, pairing opposite shocks gives negative covariance. A mixture introduces shared component-label randomness. By total covariance,

$$\operatorname{Cov}(Y^+,Y^-)=\mathbb E[\operatorname{Cov}(Y^+,Y^-\mid J)]+\operatorname{Var}(\mathbb E[Y\mid J]).$$

The second term is nonnegative. It can weaken or even overwhelm the within-component benefit. Do not assume every antithetic construction always improves every mixture model.

**Check:** Why does the code calculate standard error from the pair averages rather than the raw list of 20,000 payoffs?

## 7. A control variate uses something whose expectation we know

Let `C = exp(-rT) S_T`, so `E[C] = S_0`. Estimate

$$P=\mathbb E[Y-b(C-S_0)].$$

The extra term has zero expectation. Its optimal constant coefficient is

$$b^*=\frac{\operatorname{Cov}(Y,C)}{\operatorname{Var}(C)}.$$

Puts and stock values are negatively correlated, so b is typically negative. Subtracting `b(C-S_0)` offsets some of the payoff fluctuation.

We fit b on a separate pilot, then freeze it. Numerical integration computes the population-optimal coefficient only for comparison; the simulation is not given an oracle coefficient. A poor pilot coefficient can lose some of the benefit while the estimator remains unbiased conditional on that coefficient.

**Check:** What if we used an incorrect known expectation for C? The adjustment would introduce bias. The model's martingale check therefore matters for both pricing and variance reduction.

## 8. Why Metropolis-Hastings is a weaker baseline here

MH proposes a Gaussian step from the current X and accepts it with probability

$$\min\left(1,\frac{p(X_{new})}{p(X_{old})}\right).$$

Its long-run target is p, but consecutive draws are correlated. Many apparently different samples may contain much less independent information. The standard error must account for autocorrelation; the code uses non-overlapping batch means as an approximate diagnostic.

Each chain is initialised with an exact target draw. This avoids manufacturing a disadvantage through a bad starting point. The retained 1,000-step burn-in is redundant for stationarity but included in the measured MH cost. Direct IID sampling is available, so MH is an educational comparison, not the recommended production solver.

An initial benchmark tuned MH by estimated payoff variance. This can be misleading for a rare put: a pilot that barely encounters losses may report artificially small variance. The revised method maximises **expected squared jump distance**, averaging squared accepted moves, to tune exploration of the target law. Payoff variance and acceptance remain recorded as diagnostics. This does not make every confidence interval reliable; read the observed coverage.

**Check:** Is a 99% acceptance rate always good? No. Tiny steps can almost always be accepted while moving very slowly through the distribution.

## 9. Benchmark design and cost

The final experiment has three synthetic laws, three strikes, three budgets, five independent pilot groups, 20 evaluation repeats per group and five estimators: **13,500 price estimates**.

For each method and case we record:

- Empirical variance and RMSE against quadrature.
- Nominal 95% interval coverage and the ratio of reported error variance to observed variance.
- Method runtime, separate pilot cost and variance-times-cost efficiency.
- Population variance constants for the four independent methods.
- IS weight ESS, MH payoff ESS, acceptance rates and pilot candidates.
- Hierarchical bootstrap intervals over pilot groups and evaluation repeats.

IS weight ESS and MH payoff ESS answer different questions. A high IS weight ESS does not guarantee a small payoff variance; a proposal that emphasises rare losses can estimate a put well even when the weights themselves are uneven.

Five pilot groups give limited evidence about tuning variability. Bootstrap intervals are exploratory and not corrected for simultaneous comparisons across the whole grid. Timings depend on machine load, Python versus vectorised loops, and diagnostic overhead.

For a single 20,000-sample at-the-money price, searching 12 importance proposals can cost more than the saved variance justifies. Reusing the proposal changes the economics. The report includes both views; do not cherry-pick the one that sounds largest.

**Exercise:** Compare IS and CV at strike 80 and strike 120. Which wins on variance? Does that answer change when pilot time is included?

## 10. Delta and a useful derivative trap

Holding the terminal distribution parameters fixed, the put's pathwise delta is

$$\frac{\partial h}{\partial S_0}=-e^{-rT}\frac{S_T}{S_0}\mathbf1\{S_T<K\}.$$

The code estimates its expectation and checks it against numerical integration and finite differences using the same random shocks. Common random numbers stop unrelated simulation noise from dominating the finite difference.

This does not automatically extend to gamma by differentiating the same indicator naively: the payoff kink requires more care. A likelihood-ratio or smoothing method would be a defensible extension, but is not claimed as implemented.

## Suggested learning sessions

1. **20 minutes:** Run `price`, explain each payoff and derive the discount/martingale correction. Read `model.py`.
2. **30 minutes:** Derive the IS and CV identities. Explain the antithetic pair budget. Read `estimators.py` and change the CLI strike.
3. **30 minutes:** Read the final benchmark report. Explain variance versus standard error, pilot amortisation and the MH failure. Reproduce a small grid.
4. **Your own extension:** Make one controlled change, predict its effect, rerun held-out experiments and write what happened. Good candidates are component stratification or a path-dependent payoff under a separately specified dynamic model.

## What makes the work defensible on a CV?

Describe the question, methods, checks and limitations. A defensible summary is: 'Compared five Monte Carlo estimators under synthetic mixture return models; validated prices and population variance against quadrature and evaluated 13,500 estimates with independent pilot tuning.' Use a numerical performance claim only with its model, baseline and cost assumptions.

The repository was reconstructed with AI implementation assistance. Understand the choices, verify the experiments and add your own reasoned extension before presenting it as evidence of expertise. The source code and measured results are inspectable; no results from the lost implementation are claimed to have been recovered.

## References

- [SciPy: generalised-normal parameterisation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gennorm.html).
- [NumPy: Generator.gamma](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.gamma.html).
- [Art Owen: Monte Carlo Theory, Methods and Examples](https://artowen.su.domains/mc/), especially the variance reduction chapters.
- [Boyle, Broadie and Glasserman: Monte Carlo methods for security pricing](https://business.columbia.edu/sites/default/files-efs/pubfiles/4336/monte_carlo_methods_security_pricing.pdf).

The implementation uses NumPy and Python's standard library. It does not copy code from those references.
