# stochastix

A generic, vectorized Monte Carlo option pricing engine in Python, built around a strict separation of concerns between **stochastic dynamics**, **discretization schemes**, and **payoff contracts**.

Most pricing scripts hardcode a single model, a single scheme, and a single payoff together. stochastix decouples the three so that adding a new model, a new discretization scheme, or a new contract requires **zero changes** to the other two — each is injected as an independent strategy at runtime.

## Why this exists

Research code that mixes model dynamics, numerical schemes, and payoff logic into one script is fast to write and impossible to extend. stochastix applies the Strategy pattern (Gang of Four) to pricing: `Dynamics`, `SimulationScheme`, `OptionContract`, and `Pricer` are four independent abstractions composed at runtime, not four responsibilities bundled into one class.

## Features (v1)

- **Stochastic dynamics**: Geometric Brownian Motion, Heston (stochastic volatility, correlated via Cholesky decomposition)
- **Discretization schemes**: Euler-Maruyama, Milstein
- **Payoff contracts**: European, Asian (path-average), and two path-dependent autocallable barrier structures (fixed double-barrier knockout, adaptive two-observation-window barrier)
- **Pricer**: vectorized Monte Carlo estimator with standard error reporting
- Fully tested: convergence to closed-form solutions, martingale property, Feller condition, put-call parity, deterministic edge-case validation on hand-built paths

## Architecture

```
stochastix/
├── src/stochastix/
│   ├── time_grid.py          # TimeGrid: observation dates + simulation grid
│   ├── dynamics/              # StochasticProcess: GBM, Heston
│   ├── schemes/                # SimulationScheme: Euler-Maruyama, Milstein
│   ├── contracts/              # OptionContract: European, Asian, Autocall
│   └── pricers/                # Pricer: Monte Carlo estimator
└── tests/                      # pytest, mirrors src/ structure
```

Each layer depends only on the interface of the layer below it, never on a concrete implementation. The full design rationale — including the data-flow diagram used to freeze the architecture before implementation — is documented in [`docs/architecture.rst`](docs/architecture.rst) and rendered in the [Sphinx documentation](#documentation).

## Detailed Architecture

```
stochastix/
├── pyproject.toml
├── README.md
├── src/
│   └── stochastix/
│       ├── __init__.py
│       ├── time_grid.py          # TimeGrid
│       ├── dynamics/
│       │   ├── __init__.py
│       │   ├── base.py           # ABC StochasticProcess
│       │   ├── gbm.py
│       │   └── heston.py
│       ├── schemes/
│       │   ├── __init__.py
│       │   ├── base.py           # ABC SimulationScheme
│       │   ├── euler.py
│       │   └── milstein.py
│       ├── contracts/
│       │   ├── __init__.py
│       │   ├── base.py           # ABC OptionContract
│       │   ├── european.py
│       │   ├── asian.py
│       │   └── autocall.py       # tes deux contrats à barrière
│       ├── pricers/
│       │   ├── __init__.py
│       │   ├── base.py           # ABC Pricer
│       │   └── monte_carlo.py    # EuropeanAsianPricer, branche 4a
│       └── exceptions.py
├── tests/
│   ├── conftest.py                # fixtures partagées (seed fixe, grilles standard)
│   ├── test_time_grid.py
│   ├── dynamics/
│   │   ├── test_gbm.py
│   │   └── test_heston.py
│   ├── schemes/
│   │   ├── test_euler.py
│   │   └── test_milstein.py
│   ├── contracts/
│   │   ├── test_european.py
│   │   ├── test_asian.py
│   │   └── test_autocall.py
│   └── pricers/
│       └── test_monte_carlo.py
└── docs/                          # Sphinx, plus tard
```

### Design principle: orthogonal axes of variation

Before writing any concrete class, each pair of components was tested against one question: *if I change X, am I forced to change Y?* If no, they are separate abstractions composed via dependency injection. If yes, they belong in the same class. This is what keeps, for example, adding a third discretization scheme a one-file change rather than a rewrite across the codebase.

## Validation methodology

Correctness is checked at three levels, not just "does it run":

| Level | Criteria |
|---|---|
| **Mathematical** | Convergence to Black-Scholes closed-form (O(1/√N) Monte Carlo error), strong convergence order of Milstein vs. Euler-Maruyama, risk-neutral martingale property (E[S_T] = S_0 e^{rT}), Feller condition on Heston variance |
| **Financial** | Put-call parity, payoff non-negativity, price bounds, deterministic validation of autocallable trigger logic on hand-constructed paths |
| **Software** | Abstract base classes enforce interface contracts; instantiating an incomplete subclass fails at construction, not at runtime |

## Installation

```bash
git clone https://github.com/TeebooGH/stochastix.git
cd stochastix
uv sync
```

## Quick example

```python
from stochastix.dynamics import Heston
from stochastix.schemes import Milstein
from stochastix.contracts import EuropeanCall
from stochastix.pricers import MonteCarloPricer
from stochastix.time_grid import TimeGrid

grid = TimeGrid(maturity=1.0, n_steps=252)
process = Heston(
    s0=100,
    v0=0.04,
    kappa=2.0,
    theta=0.04,
    sigma=0.3,
    rho=-0.7,
    r=0.02,
    scheme=Milstein(),
)
contract = EuropeanCall(strike=100)
pricer = MonteCarloPricer(process=process, grid=grid, n_paths=100_000)

price, stderr = pricer.price(contract)
```

## Testing

```bash
uv run pytest
```

## Documentation

Full API reference and architecture rationale built with Sphinx (autodoc, Napoleon): [link once published on GitHub Pages].

## Roadmap (v2.0)

- Longstaff-Schwartz least-squares Monte Carlo for early-exercise (discretionary) American options
- `Calibrator`: parameter calibration against market data (with Common Random Numbers for stable optimizer convergence)
- `DataLoader`: market data ingestion layer feeding the calibrator
- Interactive UI for exploring convergence and pricing surfaces

## Author

Built by Thibaud OU, M1 at Télécom SudParis (Institut Polytechnique de Paris), applied mathematics and computer science.