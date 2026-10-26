# Formative 3: Probability, Bayes and Gradient Descent

**Group 4** · Uwineza Bonesha Kevine · Esther Kirabo · Isheja Ladouce
Repository: https://github.com/Bonesha-5/Group4_Formative3_Assignment

This project answers four questions with code we wrote ourselves:

| Part | Question | Notebook |
|---|---|---|
| 1 | Can we separate two hidden groups of heights with no labels? | `EM_algorithm.ipynb` |
| 2 | How much does one word tell us about a movie review? | `bayesian.ipynb` |
| 3 | What does gradient descent do, step by step, by hand? | Handwritten PDF |
| 4 | Does the code reproduce our hand calculations exactly? | `gradient_desent.ipynb` |

---

## How to run

1. Open any notebook with the **Open in Colab** badge at the top, or run it locally with Jupyter.
2. Keep `GaltonFamilies.csv` and `IMDB Dataset.csv` in the same folder as the notebooks.
3. Requirements: Python 3, `numpy`, `matplotlib`, and **SciPy 1.15 or newer** (Part 4 uses `scipy.differentiate.derivative`). Upgrade with `pip install --upgrade scipy` if needed.
4. Run all cells from top to bottom.

---

## Decisions we made (and why)

| Decision | Choice | Reason |
|---|---|---|
| Part 1 populations | Mothers + children | Assignment allowed mothers or fathers; we chose mothers. |
| Mothers per family | One per family (205) | A mother is repeated on every row of her family. Counting her once avoids duplicates. |
| Units | Inches | The dataset is in inches. Test heights in cm are divided by 2.54. |
| EM starting means | 60th and 70th percentiles (67.0 and 68.0 in) | Two different starting points inside the data, so the groups can separate. |
| EM stopping rule | Change in log-likelihood < 0.000001, max 1,000 iterations | Stop when the fit no longer improves, with a safety limit. |
| Part 2 direction | P(Positive \| keyword) | Assignment asked for one direction only. |
| Keyword matching | Whole words only | "excellent" should not match inside another word. |
| Part 3 bias | **One shared scalar b**, written as b·[1, 1] | In standard linear regression the bias is a single intercept. See "Why b is one number" below. |
| Learning rate | α = 0.01 | Values above about 0.0157 make the error explode for this data; 0.01 is safe and still fast. |
| Iterations in Part 3 | 3 | One per group member. |

---

## Part 1: Expectation-Maximization (EM)

### The problem
We mixed **205 mothers' heights** with **934 children's heights** (1,139 values) and removed the labels. EM must find two bell curves that explain the data, without ever seeing who is who. The true labels were kept aside **only to check the results at the end**.

### How EM works in our code

| Step | Function / cell | What it does |
|---|---|---|
| Bell curve | `gaussian_pdf(x, mean, variance)` | How typical a height is for a group |
| E-step | `e_step(...)` | Responsibility of each group for each height: γ₁ = π₁N₁ ÷ (π₁N₁ + π₂N₂) |
| M-step | update cells | New means, variances and weights, using the responsibilities as weights |
| Fit score | log-likelihood | Σ log(π₁N₁ + π₂N₂); higher is better |
| Stop | convergence loop | Stops when the log-likelihood changes by less than 0.000001 |

### Optimization tracking table

| Iteration | μ₁ | μ₂ | σ₁² | σ₂² | π₁ | π₂ | Log-likelihood |
|---|---|---|---|---|---|---|---|
| 0 (initialization) | 67.00 | 68.00 | 12.58 | 12.58 | 0.500 | 0.500 | −3127.43 |
| 1 | 65.78 | 66.77 | 11.99 | 12.73 | 0.524 | 0.476 | −3057.74 |
| 2 | 65.77 | 66.78 | 11.71 | 13.01 | 0.524 | 0.476 | −3057.46 |

**Converged after 428 iterations.** Final values:

| | Mean (in) | Variance | Weight |
|---|---|---|---|
| Group 1 (shorter) | 64.30 | 5.79 | 0.644 |
| Group 2 (taller) | 69.79 | 5.47 | 0.356 |

Final log-likelihood: **−3035.75**. Across all 428 iterations, the log-likelihood **never decreased**, which is EM's guarantee and our check that the code is correct.

### Should we just split at the global mean? No.

We tested it against the hidden labels:

| | Naive split at 66.25 in | Truth |
|---|---|---|
| Mean of the "mother" pile | 63.58 | 64.00 |
| Mean of the "child" pile | 69.47 | 66.75 |
| Accuracy | **57.6%** | — |
| Children wrongly called "mother" | **48.2%** | — |

Why it fails:
1. **The groups overlap.** Every mother's height (58 to 70.5 in) also appears among the children, so no single line can separate them.
2. **The pile means are biased.** Cutting each bell curve chops off its tail, so the "child" pile mean (69.47) is almost 3 inches above the true children's mean (66.75).
3. **The line is pulled toward the bigger group.** There are 4.5 times more children than mothers, so the global mean (66.25) sits close to the children's mean.
4. **A line gives no probabilities.** EM gives each height a probability for each group, which is what we need for the test height.

### What EM actually found
EM's groups are **64.3 in** and **69.8 in**. Compare with the real subgroups:

| | Mean height (in) |
|---|---|
| Mothers | 64.00 |
| Daughters | 64.10 |
| Sons | 69.23 |

Galton's "children" are grown adults. Daughters are as tall as mothers, and sons are much taller. So the clearest split in the numbers is **shorter people (mothers + daughters)** vs **taller people (sons)**, and that is what EM found. EM maximizes how well the curves fit the data; it does not know what "mother" or "child" means.

### Classifying a test height
`classify(height, unit="in")` runs one E-step with the final parameters:

```
classify(66)          → P(Group 1, shorter) = 0.8353   P(Group 2, taller) = 0.1647
classify(170, "cm")   → 66.93 in → P(Group 1) = 0.6711   P(Group 2) = 0.3289
```

The two probabilities always add up to 1.

---

## Part 2: Bayesian probability (IMDb reviews)

### Keywords

| Positive | Negative |
|---|---|
| excellent, superb, masterpiece, outstanding | terrible, awful, horrible, disappointing |

We chose strong, unambiguous opinion words that reviewers use to judge a film as a whole, rather than mild words like "good" or "bad" that often appear in mixed reviews ("good acting, bad plot").

### Bayes' theorem

P(Positive | keyword) = P(keyword | Positive) × P(Positive) ÷ P(keyword)

### Results (50,000 reviews: 25,000 positive, 25,000 negative)

| Keyword | Prior P(Positive) | Likelihood P(keyword \| Positive) | Marginal P(keyword) | Posterior P(Positive \| keyword) |
|---|---|---|---|---|
| excellent | 0.5000 | 0.1147 | 0.0710 | **0.8074** |
| superb | 0.5000 | 0.0410 | 0.0240 | **0.8557** |
| masterpiece | 0.5000 | 0.0351 | 0.0241 | **0.7274** |
| outstanding | 0.5000 | 0.0247 | 0.0149 | **0.8318** |
| terrible | 0.5000 | 0.0153 | 0.0540 | **0.1418** |
| awful | 0.5000 | 0.0114 | 0.0577 | **0.0985** |
| horrible | 0.5000 | 0.0130 | 0.0420 | **0.1552** |
| disappointing | 0.5000 | 0.0068 | 0.0157 | **0.2176** |

### What the numbers tell us
- **Positive words push the probability up** from 50% to 73–86%, and **negative words push it down** to 10–22%.
- **"superb" is the strongest positive signal** (85.6%), stronger than "excellent," even though "excellent" appears three times more often. How often a word appears and how reliable it is are different things.
- **"masterpiece" is the weakest positive word** (72.7%). A likely reason is that it also appears in negative reviews, for example "this is no masterpiece."
- **Check:** each posterior equals (positive reviews with the word) ÷ (all reviews with the word), counted directly. Both methods match for every keyword.

### Implementation
Basic Python only: the `csv` module to read the file, loops and counters for the probabilities, and `re` (Python's standard library) for whole-word matching. No machine learning libraries.

---

## Part 3: Gradient descent by hand

### Setup

X = [[1, 3], [4, 10]], y = [5, 6], m = [−1, 2], b = 1, α = 0.01, n = 2

Model: ŷ = Xm + b·[1, 1] · Cost: J = (1/n)·eᵀe, where e = ŷ − y

### Gradients (chain rule)
J depends on e, e depends on ŷ, and ŷ depends on m and b:

- ∂J/∂m = (2/n) · Xᵀ · e (Xᵀ puts each feature's values in a row, so each weight collects its own feature × error)
- ∂J/∂b = (2/n) · 1ᵀ · e = (2/n) · Σe (the same b is in every prediction, so its gradient adds up all the errors)

### Iterations (one per member)

| Iteration | Done by | ŷ | e | MSE | ∂J/∂m | ∂J/∂b | New m | New b |
|---|---|---|---|---|---|---|---|---|
| 1 | Isheja Ladouce | [6, 17] | [1, 11] | 61 | [45, 113] | 12 | [−1.45, 0.87] | 0.88 |
| 2 | Esther Kirabo | [2.04, 3.78] | [−2.96, −2.22] | 6.845 | [−11.84, −31.08] | −5.18 | [−1.3316, 1.1808] | 0.9318 |
| 3 | Uwineza Bonesha Kevine | [3.1426, 7.4134] | [−1.8574, 1.4134] | 2.7238 | [3.7962, 8.5618] | −0.444 | [−1.369562, 1.095182] | 0.93624 |

After the 3rd update, the MSE is **2.3904**.

### The trend
- **The error falls every time:** 61 → 6.845 → 2.724 → 2.390. The parameters move in a direction that reduces the error.
- **The gradients change sign.** In iteration 1 the predictions are too high (all gradients positive), in iteration 2 too low (all negative), in iteration 3 mixed. Gradient descent corrects itself in both directions.
- **m₂ overshoots, then settles:** 2 → 0.87 → 1.18 → 1.10. Its gradient is largest because x₂ has the largest values (3 and 10).
- **b moves the least**, because its gradient is not multiplied by the x values.

### Why b is one number
The bias is the **y-intercept**: the prediction when every input is 0, like a taxi's starting fee. A line (or a plane, with two inputs) crosses the y-axis at only **one** point, so the model has one bias, shared by all data points. b = [1, 1] is that one value written once per row so it can be added to Xm. A separate bias per data point could not predict a new point and would let the model memorize the answers.

---

## Part 4: Gradient descent in code

`gradient_desent.ipynb` repeats Part 3 in Python, with every step visible.

| Function | Job |
|---|---|
| `predict`, `compute_errors`, `mse` | ŷ = Xm + b, e = ŷ − y, J = (1/n)·eᵀe |
| `cost_equation` | The MSE as one equation of [m₁, m₂, b] |
| `scipy_derivative(equation, params, index)` | Accepts any equation and returns its derivative for one parameter, using `scipy.differentiate.derivative` |
| `gradients_with_scipy` | Gradients from SciPy (used for the updates) |
| `gradients_with_matrices` | Gradients from our Part 3 formulas (used as a check) |
| `update` | m = m − α·∂J/∂m, b = b − α·∂J/∂b |

**How the SciPy derivative works:** our cost has three inputs, but SciPy differentiates one-input functions. So `scipy_derivative` freezes the other parameters and lets SciPy move one of them slightly to measure the slope. It is first tested on f(x) = x², where f′(3) must be 6.

### Results
- **The SciPy gradients match the matrix formulas at every iteration** (for example [45, 113] and 12 in iteration 1), and iterations 1 to 3 reproduce the hand calculations exactly. Two different methods, one answer.
- The code runs 20 iterations. The MSE goes from 61 to 2.00, with final m = [−1.4309, 1.1123] and b = 1.1768.
- Two plots: m₁, m₂ and b over the iterations, and the MSE over the iterations.

---

## Presentation plan

Every member speaks in every part, presenting the piece they built.

| Part | Uwineza Bonesha Kevine | Esther Kirabo | Isheja Ladouce |
|---|---|---|---|
| 1 EM | Data, global-mean question | E-step, test-height demo | M-step, tracking table, convergence |
| 2 Bayes | Keywords and why | Prior and likelihood | Marginal, posterior, insights |
| 3 Manual | Iteration 1, MSE | Chain rule, iteration 2 | Iteration 3, trend |
| 4 Code | Setup and model functions | SciPy derivative | Table, plots, conclusion |

### Questions we prepared for
- **Why not split at the global mean?** Overlap, biased pile means, the pull of the bigger group, and no probabilities (Part 1 table).
- **What is a responsibility?** The probability that a height belongs to a group, given the current parameters.
- **Why multiply by the responsibility in the M-step?** It is our best guess of the missing label; with true 0/1 labels the formulas become ordinary averages.
- **Why the log of the likelihood?** Multiplying 1,139 small probabilities rounds to 0 on a computer; the log turns the product into a sum.
- **When does EM stop?** When the log-likelihood changes by less than the tolerance. It never decreases.
- **Why did EM find shorter vs taller people?** Adult daughters are as tall as mothers; the clearest split is by sex, not generation.
- **Why Xᵀ in ∂J/∂m?** Each weight needs its own feature's values multiplied by the errors; transposing puts each feature in a row.
- **Why is b one number?** It is the y-intercept, and the gradient (2/n)·Σe is a single number.
- **Why α = 0.01?** 0.1 explodes (61 → 8,090 → 1.1 million), 0.001 is too slow, 0.01 is safe and fast.

---

## Team and contributions

| Member | Part 1 | Part 2 | Part 3 | Part 4 |
|---|---|---|---|---|
| Uwineza Bonesha Kevine | Data preparation, initialization | Data cleaning, keyword selection | Predictions, MSE, iteration 1 | Model setup and initialization |
| Esther Kirabo | E-step, responsibilities | Prior and likelihood | Chain-rule derivatives, iteration 2 | SciPy derivative, gradient updates |
| Isheja Ladouce | M-step, log-likelihood, convergence | Marginal and posterior, tables | Parameter updates, iteration 3 | Tracking table, plots, interpretation |

All three members reviewed every part together. Details are in the team tracking sheet and the contributions PDF.

---

## References
- Galton Families dataset (heights of parents and adult children, in inches).
- IMDb Dataset of 50K Movie Reviews.
- Google, *Machine Learning Crash Course: Linear Regression*: https://developers.google.com/machine-learning/crash-course/linear-regression
- scikit-learn, `LinearRegression` documentation (single `intercept_` for one output): https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
- James, Witten, Hastie and Tibshirani, *An Introduction to Statistical Learning*, Chapter 3.
