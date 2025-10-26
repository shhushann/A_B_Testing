"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""
############################### LOGGER
from abc import ABC, abstractmethod
from loguru import logger  
import pyment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("logs", exist_ok=True)
os.makedirs("reports", exist_ok=True)



logger.remove()

logger.add(
    "logs/bandit_{time}.log",
    rotation="5 MB",
    retention="7 days",
    compression="zip",
    level="DEBUG",
    enqueue=True,
    backtrace=True,
    diagnose=True
)


logger.add(
    sink=lambda msg: print(msg, end=""),
    level="INFO"
)

NUM_TRIALS = 200000
EPS = 0.1
BANDIT_REWARDS = [1,2,3,4]
class Bandit(ABC):
    """ """

    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        self.p = p
        self.N = 0
        self.p_estimate = 0


    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        """ """
        pass

    @abstractmethod
    def update(self):
        """ """
        pass

    @abstractmethod
    def experiment(self):
        """ """
        pass

    @abstractmethod
    def report(self):
        """ """
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#


class Visualization:
    """ """
    def plot1(self, cumulative_avg, title="Epsilon-Greedy Learning Curve", save_path=None):
        """

        :param cumulative_avg: 
        :param title: (Default value = "Epsilon-Greedy Learning Curve")
        :param save_path: (Default value = None)

        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(cumulative_avg)
        plt.title(title)
        plt.xlabel("Trial")
        plt.ylabel("Cumulative Average Reward")
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(cumulative_avg)
        plt.xscale("log")
        plt.title(title + " (Log Scale)")
        plt.xlabel("Trial (log scale)")
        plt.ylabel("Cumulative Average Reward")
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            base, ext = os.path.splitext(save_path)
            plt.savefig(base + "_log" + ext)
            plt.close()
        else:
            plt.show()

    def plot2(self, rewards_eg, rewards_ts, regrets_eg=None, regrets_ts=None, save_path=None):
        """

        :param rewards_eg: 
        :param rewards_ts: 
        :param regrets_eg: (Default value = None)
        :param regrets_ts: (Default value = None)
        :param save_path: (Default value = None)

        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(np.cumsum(rewards_eg), label="Epsilon-Greedy", alpha=0.7)
        plt.plot(np.cumsum(rewards_ts), label="Thompson Sampling", alpha=0.7)
        plt.title("Cumulative Rewards Comparison")
        plt.xlabel("Trial")
        plt.ylabel("Cumulative Reward")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

        if regrets_eg is not None and regrets_ts is not None:
            plt.figure(figsize=(10, 5))
            plt.plot(np.cumsum(regrets_eg), label="Epsilon-Greedy", alpha=0.7)
            plt.plot(np.cumsum(regrets_ts), label="Thompson Sampling", alpha=0.7)
            plt.title("Cumulative Regret Comparison")
            plt.xlabel("Trial")
            plt.ylabel("Cumulative Regret")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            if save_path:
                base, ext = os.path.splitext(save_path)
                plt.savefig(base + "_regret" + ext)
                plt.close()
            else:
                plt.show()

#--------------------------------------#

class EpsilonGreedy(Bandit):
    """ """
    

    def __init__(self,p, epsilon):
        super().__init__(p)
        self.epsilon = epsilon
        self.rewards = []



    def __repr__(self):
        return f"EpsilonGreedy Bandit(p={self.p:.2f}, estimate={self.p_estimate:.2f})"



    def pull(self):
        """ """
        return np.random.random() < self.p

    def update(self, reward):
        """

        :param reward: 

        """

        self.N += 1
        self.p_estimate = self.p_estimate + (1 / self.N) * (reward - self.p_estimate)

    @staticmethod
    def experiment(bandit_probs, epsilon, N=10000):
        """

        :param bandit_probs: 
        :param epsilon: 
        :param N: (Default value = 10000)

        """

        bandits = [EpsilonGreedy(p, epsilon) for p in bandit_probs]
        true_means = np.array(bandit_probs)
        optimal_index = np.argmax(true_means)

        count_suboptimal = 0
        rewards = np.empty(N)

        for i in range(N):
            epsilon = max(0.01, 1 / np.sqrt(i + 1))

            if np.random.random() < epsilon:
                j = np.random.choice(len(bandits))  # explore
            else:
                j = np.argmax([b.p_estimate for b in bandits])  # exploit


            reward = bandits[j].pull()
            bandits[j].update(reward)
            rewards[i] = reward


            if j != optimal_index:
                count_suboptimal += 1

        cumulative_average = np.cumsum(rewards) / (np.arange(N) + 1)
        percent_suboptimal = (count_suboptimal / N) * 100


        for b in bandits:
            logger.info(f"Bandit true p={b.p:.2f}, estimated p={b.p_estimate:.2f}")
        logger.warning(f"Epsilon={epsilon:.2f} → {percent_suboptimal:.2f}% suboptimal choices")
        logger.info("--------------------------------------------------")

        return rewards, cumulative_average, percent_suboptimal, [b.p_estimate for b in bandits]

    def report(self):
        """ """

        avg_reward = np.mean(self.rewards)
        optimal_p = max(BANDIT_REWARDS) / max(BANDIT_REWARDS)
        avg_regret = (max(self.true_ps) - avg_reward) if hasattr(self, "true_ps") else None


        logger.info(f"Average reward for {self.__class__.__name__}: {avg_reward:.3f}")
        if avg_regret is not None:
            logger.warning(f"Average regret for {self.__class__.__name__}: {avg_regret:.3f}")

        df = pd.DataFrame({
            "Trial": np.arange(1, len(self.rewards) + 1),
            "Reward": self.rewards,
            "Algorithm": self.__class__.__name__
        })
        report_path = os.path.join("reports", f"{self.__class__.__name__.lower()}_results.csv")
        df.to_csv(report_path, index=False)
        logger.info(f"Results saved to {report_path}")


class ThompsonSampling(Bandit):
    """ """
    def __init__(self, p):
        super().__init__(p)
        self.alpha = 1
        self.beta = 1
        self.rewards = []

    def __repr__(self):
        return f"ThompsonSampling Bandit(p={self.p:.2f}, alpha={self.alpha}, beta={self.beta})"

    def pull(self):
        """ """
        return np.random.random() < self.p

    def update(self, reward):
        """

        :param reward: 

        """
        self.alpha += reward
        self.beta += 1 - reward
        self.rewards.append(reward)

    @staticmethod
    def experiment(bandit_probs, N=10000):
        """

        :param bandit_probs: 
        :param N: (Default value = 10000)

        """
        bandits = [ThompsonSampling(p) for p in bandit_probs]
        true_means = np.array(bandit_probs)
        optimal_index = np.argmax(true_means)

        rewards = np.empty(N)
        count_suboptimal = 0

        for i in range(N):
            sampled_theta = [np.random.beta(b.alpha, b.beta) for b in bandits]
            j = np.argmax(sampled_theta)

            reward = bandits[j].pull()
            bandits[j].update(reward)
            rewards[i] = reward

            if j != optimal_index:
                count_suboptimal += 1

        cumulative_average = np.cumsum(rewards) / (np.arange(N) + 1)
        percent_suboptimal = (count_suboptimal / N) * 100
        estimates = [b.alpha / (b.alpha + b.beta) for b in bandits]

        for b in bandits:
            logger.info(
                f"Bandit true p={b.p:.2f}, posterior mean={b.alpha / (b.alpha + b.beta):.2f}"
            )
        logger.warning(f"Thompson Sampling → {percent_suboptimal:.2f}% suboptimal choices")
        logger.info("--------------------------------------------------")

        return rewards, cumulative_average, percent_suboptimal, estimates

    def report(self):
        """ """
        if not hasattr(self, "rewards") or len(self.rewards) == 0:
            logger.error("No rewards recorded for Thompson Sampling instance.")
            return

        avg_reward = np.mean(self.rewards)
        optimal_p = max(BANDIT_PROBABILITIES)
        avg_regret = optimal_p - avg_reward

        logger.info(f"Average reward for ThompsonSampling: {avg_reward:.3f}")
        logger.warning(f"Average regret for ThompsonSampling: {avg_regret:.3f}")

        df = pd.DataFrame({
            "Trial": np.arange(1, len(self.rewards) + 1),
            "Reward": self.rewards,
            "Algorithm": "ThompsonSampling"
        })

        os.makedirs("reports", exist_ok=True)
        file_path = os.path.join("reports", f"{self.__class__.__name__.lower()}_results.csv")
        df.to_csv(file_path, index=False)
        logger.info(f"Report temporarily saved to {file_path}")


class Comparison:
    def __init__(self, exp_path):
        self.exp_path = exp_path
        self.report_dir = os.path.join(exp_path, "reports")
        self.plot_dir = os.path.join(exp_path, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)

    def load_reports(self):
        import pandas as pd
        reports = {}
        for file in os.listdir(self.report_dir):
            if file.endswith(".csv") and "epsilongreedy" in file or "thompsonsampling" in file:
                algo_name = file.replace(".csv", "")
                df = pd.read_csv(os.path.join(self.report_dir, file))
                if "Reward" in df.columns:
                    reports[algo_name] = df
                else:
                    logger.warning(f"Skipping {file}: no 'Reward' column found.")
            else:
                logger.debug(f"Skipping non-algorithm file: {file}")
        return reports

    def summarize(self):
        import numpy as np
        reports = self.load_reports()
        summary = {}
        for name, df in reports.items():
            avg_reward = df["Reward"].mean()
            final_reward = df["Reward"].iloc[-1]
            cumulative_reward = np.sum(df["Reward"])
            summary[name] = {
                "mean_reward": avg_reward,
                "final_reward": final_reward,
                "total_reward": cumulative_reward,
                "num_trials": len(df)
            }
        return summary

    def visualize(self):
        import matplotlib.pyplot as plt
        import numpy as np

        reports = self.load_reports()
        plt.figure(figsize=(10, 6))
        for name, df in reports.items():
            cumulative = np.cumsum(df["Reward"]) / np.arange(1, len(df) + 1)
            plt.plot(cumulative, label=name, alpha=0.8)

        plt.title("Algorithm Comparison: Average Reward Over Time")
        plt.xlabel("Trial")
        plt.ylabel("Cumulative Average Reward")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        save_path = os.path.join(self.plot_dir, "comparison.png")
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Comparison plot saved to {save_path}")

    def best_algorithm(self):
        summary = self.summarize()

        # Sort by mean reward (or choose another metric)
        best = max(summary.items(), key=lambda x: x[1]["mean_reward"])
        algo_name, stats = best

        logger.success(
            f" Best algorithm: {algo_name} | Mean Reward={stats['mean_reward']:.3f}, "
            f"Final Reward={stats['final_reward']:.3f}, Total={stats['total_reward']:.1f}"
        )
        return algo_name, stats

    def rank_algorithms(self):
        summary = self.summarize()
        df = (
            pd.DataFrame(summary).T
            .sort_values("mean_reward", ascending=False)
            .reset_index()
            .rename(columns={'index': 'Algorithm'})
        )
        logger.info("Algorithm ranking by mean reward:")
        for i, row in df.iterrows():
            logger.info(f"{i + 1}. {row['Algorithm']} — Mean={row['mean_reward']:.3f}, Final={row['final_reward']:.3f}")
        return df

    def report_summary(self):
        import pandas as pd
        summary = self.summarize()

        def barplot_summary(self):
            import matplotlib.pyplot as plt
            import pandas as pd

            df = pd.DataFrame(self.summarize()).T
            df = df.sort_values("mean_reward", ascending=False)

            plt.figure(figsize=(8, 5))
            plt.bar(df.index, df["mean_reward"], color='skyblue')
            plt.ylabel("Mean Reward")
            plt.title("Mean Reward per Algorithm")
            plt.xticks(rotation=20)
            plt.tight_layout()

            save_path = os.path.join(self.plot_dir, "comparison_summary_bar.png")
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Bar plot summary saved to {save_path}")

        # Log to console
        logger.info("Summary of experiment performance:\n")
        for algo, stats in summary.items():
            logger.info(
                f"{algo}: Mean={stats['mean_reward']:.3f}, Final={stats['final_reward']:.3f}, "
                f"Total={stats['total_reward']:.1f}, Trials={stats['num_trials']}"
            )

        # Save summary as CSV
        df_summary = pd.DataFrame(summary).T.reset_index().rename(columns={'index': 'Algorithm'})
        summary_path = os.path.join(self.report_dir, "comparison_summary.csv")
        df_summary.to_csv(summary_path, index=False)
        logger.info(f"Comparison summary saved to {summary_path}")

        # Also save as JSON for potential reuse
        import json
        json_path = os.path.join(self.report_dir, "comparison_summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=4)
        logger.info(f"Comparison summary JSON saved to {json_path}")




if __name__=='__main__':
   
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")
