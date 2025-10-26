#  A/B Testing with Multi-Armed Bandits



This repository implements and compares two adaptive **A/B Testing** algorithms | **Epsilon-Greedy** and **Thompson Sampling** , within the **Multi-Armed Bandit** framework.


---

##  Objective

Traditional A/B testing splits users equally between variants, which wastes traffic on suboptimal options.  
Bandit algorithms improve this by **learning dynamically** from each user interaction.

In this homework, we:
1. Simulate multiple ads (arms) with different true conversion probabilities.
2. Apply **Epsilon-Greedy** and **Thompson Sampling** algorithms.
3. Run experiments over 10,000–20,000 trials.
4. Compare results via cumulative reward and regret.
5. Save visualizations and reports automatically.

---

##  Implementation Details

###  Classes

| Class | Description |
|--------|-------------|
| **`Bandit`** | Abstract base class with template methods for all algorithms (`pull`, `update`, `experiment`, `report`). |
| **`EpsilonGreedy`** | Implements ε-greedy with decaying ε = max(0.01, 1 / √t) : high initial exploration that slowly decreases to 0.01. |
| **`ThompsonSampling`** | Bayesian algorithm using Beta priors to sample uncertainty and select arms probabilistically. |
| **`Visualization`** | Generates reward/regret plots and learning curves. |
| **`Comparison`** | Loads reports, summarizes mean and total rewards, ranks algorithms, and identifies the best performer. |

---

##  Experimental Setup

| Parameter | Value | Description |
|------------|--------|-------------|
| **Number of Arms (Ads)** | 3 | Each ad has a true reward probability |
| **True Rewards** | `[1,2,3,4]` | Used for both algorithms |
| **Trials (N)** | 20,000 (configurable) | Number of iterations per algorithm |
| **Algorithms Tested** | Epsilon-Greedy, Thompson Sampling | |
| **Epsilon Values** | `[0.01, 0.1, 0.3]` | Fixed and decaying variations |
| **Logger** | `loguru` | Tracks events with timestamps and levels |
| **Output Folders** | `/experiments/exp#/reports` and `/experiments/exp#/plots` | Created automatically |

##  Installation and Setup

### 1. Clone the repository

git clone https://github.com/shhushann/A_B_Testing.git

## Navigate to the correct directory
cd A_B_Testing

## 2. Create and activate a virtual environment
  -  Windows
python -m venv venv
venv\Scripts\activate

  - macOS / Linux
python3 -m venv venv
source venv/bin/activate


## 3.  Install dependencies
pip install -r requirements.txt

#  Running the Experiment
To start a new experiment:
python test.py
Each run will automatically:
- Create a new experiment folder under /experiments/exp#
- Run all Epsilon-Greedy and Thompson Sampling tests
- Save plots in /plots/ and reports in /reports/
- Log progress in /logs/




** The Repository already includes 4 done experiments as workflow examples.
