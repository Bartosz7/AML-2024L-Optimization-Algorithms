import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# TODO: type annotations


def plot_figures_for_cv(l_iwls_vals_list, l_sgd_vals_list, l_adam_vals_list) -> None:
    """Function plots Minus log likelihood vs Iteration number for each model and split from CV."""
    n_splits = len(l_iwls_vals_list)
    for i in range(n_splits):

        l_iwls_vals = l_iwls_vals_list[i]
        l_sgd_vals = l_sgd_vals_list[i]
        l_adam_vals = l_adam_vals_list[i]

        plt.figure(figsize=(15, 8))
        plt.plot(
            np.linspace(1, len(l_iwls_vals), len(l_iwls_vals)),
            l_iwls_vals,
            label="IWLS",
        )
        plt.plot(
            np.linspace(1, len(l_sgd_vals), len(l_sgd_vals)), l_sgd_vals, label="SGD"
        )
        plt.plot(
            np.linspace(1, len(l_adam_vals), len(l_adam_vals)),
            l_adam_vals,
            label="ADAM",
        )
        plt.legend()
        plt.rc("xtick", labelsize=10)
        plt.rc("ytick", labelsize=10)
        plt.rc("legend", fontsize=15)
        plt.xlabel("Iterations", fontsize=20)
        plt.ylabel("Minus log-likelihood", fontsize=20)
        plt.title(f"Comparison between models for split {i+1}", fontsize=20)
        plt.show()


def plot_acc_boxplots(sgd_acc_list, adam_acc_list, iwls_acc_list) -> None:
    """TODO"""
    df = pd.DataFrame(
        {
            "acc": sgd_acc_list + adam_acc_list + iwls_acc_list,
            "model": ["sgd" for i in range(len(sgd_acc_list))]
            + ["adam" for i in range(len(adam_acc_list))]
            + ["iwls" for i in range(len(iwls_acc_list))],
        }
    )

    sns.boxplot(data=df, x="model", y="acc").set(
        title=f"Models accuracy for {len(sgd_acc_list)} train test splits",
        xlabel="Models",
        ylabel="Accuracy",
    )
