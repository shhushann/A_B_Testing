from Bandit import EpsilonGreedy, ThompsonSampling, Visualization, Comparison, logger
import numpy as np
import os


def get_next_experiment_folder(base="experiments"):
    os.makedirs(base, exist_ok=True)
    existing = [d for d in os.listdir(base) if d.startswith("exp")]
    next_idx = len(existing) + 1
    exp_dir = os.path.join(base, f"exp{next_idx}")
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "reports"), exist_ok=True)
    logger.info(f"Created new experiment directory: {exp_dir}\n")
    return exp_dir


def run_epsilon_greedy_tests(exp_dir):
    bandit_probs = [0.2, 0.5, 0.75]
    epsilons = [0.01, 0.1, 0.3]
    N = 10_000

    results = {}
    viz = Visualization()
    logger.info("Starting Epsilon-Greedy experiments...\n")

    for eps in epsilons:
        logger.info(f"Running Epsilon-Greedy experiment with epsilon={eps}")

        rewards, cumulative_avg, subopt, estimates = EpsilonGreedy.experiment(
            bandit_probs, eps, N=N
        )

        results[eps] = {"rewards": rewards, "cumulative_avg": cumulative_avg}

        # Save report directly inside experiment folder
        report_path = os.path.join(exp_dir, "reports", f"epsilongreedy_eps{eps}.csv")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        df = np.column_stack((np.arange(1, N + 1), rewards))
        np.savetxt(report_path, df, delimiter=",", header="Trial,Reward", comments="")
        logger.info(f"Report saved to {report_path}")

        # Plot
        plot_path = os.path.join(exp_dir, "plots", f"epsilongreedy_eps{eps}.png")
        viz.plot1(cumulative_avg, title=f"Epsilon-Greedy (ε={eps})", save_path=plot_path)

        logger.info(
            f"Completed ε={eps} | Final estimates={np.round(estimates, 3)} | "
            f"Suboptimal choices={subopt:.2f}%\n"
        )

    logger.info("All Epsilon-Greedy experiments completed.\n")
    return results


def run_thompson_sampling_test(exp_dir):
    bandit_probs = [0.2, 0.5, 0.75]  # Multi-armed setup
    N = 10_000

    logger.info("Starting Thompson Sampling (multi-armed) experiment...\n")
    viz = Visualization()

    rewards, cumulative_avg, subopt, estimates = ThompsonSampling.experiment(
        bandit_probs, N=N
    )


    report_path = os.path.join(exp_dir, "reports", "thompsonsampling_multiarmed.csv")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    df = np.column_stack((np.arange(1, N + 1), rewards))
    np.savetxt(report_path, df, delimiter=",", header="Trial,Reward", comments="")
    logger.info(f"Report saved to {report_path}")


    plot_path = os.path.join(exp_dir, "plots", "thompsonsampling_multiarmed.png")
    viz.plot1(cumulative_avg, title="Thompson Sampling (Multi-Armed)", save_path=plot_path)

    logger.info(
        f"Thompson Sampling completed | Final estimates={np.round(estimates, 3)} | "
        f"Suboptimal choices={subopt:.2f}%\n"
    )

    return {"rewards": rewards, "cumulative_avg": cumulative_avg}


if __name__ == "__main__":
    exp_dir = get_next_experiment_folder()

    logger.info("=== Running Epsilon-Greedy Tests ===\n")
    eg_results = run_epsilon_greedy_tests(exp_dir)

    logger.info("=== Running Thompson Sampling Test ===\n")
    ts_result = run_thompson_sampling_test(exp_dir)


    logger.info("=== Running Comparison ===\n")
    comp = Comparison(exp_dir)
    comp.visualize()
    rank_df = comp.rank_algorithms()
    best_algo, best_stats = comp.best_algorithm()
    comp.report_summary()

    rank_path = os.path.join(exp_dir, "reports", "algorithm_ranking.csv")
    rank_df.to_csv(rank_path, index=False)
    best_path = os.path.join(exp_dir, "reports", "best_algorithm.txt")
    with open(best_path, "w") as f:
        f.write(f"Best Algorithm: {best_algo}\n")
        for k, v in best_stats.items():
            f.write(f"{k}: {v}\n")

    logger.info(f" Experiment completed. All outputs saved in {exp_dir}\n")
