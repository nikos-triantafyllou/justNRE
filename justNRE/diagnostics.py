import numpy as np
import matplotlib.pyplot as plt
import corner


def plot_confusion_matrix(X_norm, y, model):
    """
    Computes and plots a confusion matrix for a binary NRE classifier.

    Returns:
        fig: the matplotlib Figure (caller decides whether to show/save it)
        conf_matrix: the raw confusion matrix array
    """
    # Generate predictions for validation or test set
    y_pred = model.predict(X_norm)

    # For binary classification, convert probabilities to class labels
    y_pred_classes = (y_pred > 0.5).astype(int)

    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    # Compute the confusion matrix. sklearn sorts labels ascending, i.e. [0, 1].
    # In this codebase flag_0 = independently drawn, flag_1 = jointly drawn
    # (see nre.data.prepare_for_NRE), so row/col order here is
    # [Independently drawn, Jointly drawn] -- NOT [Jointly, Independently] as
    # the original tick labels claimed.
    conf_matrix = confusion_matrix(y, y_pred_classes)
    labels = ["Independently drawn", "Jointly drawn"]

    # Plot the confusion matrix onto an explicit fig/ax (not global pyplot state)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="bone",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix")

    return fig, conf_matrix


def cornerplot1(data, theta_true, param_names, fig=None, color=None, name=None, weights=None):
    if len(theta_true) == 1:
        theta_true_ = None
    else:
        theta_true_ = theta_true

    fig = corner.corner(
        data,
        bins=40,
        hist_bin_factor=0.5,
        weights=weights,
        fig=fig,
        color=color,
        levels=(0.68, 0.95),
        plot_contour=True,
        fill_contours=True,
        plot_density=True,
        plot_datapoints=False,
        labels=param_names,
        smooth=1,
        truths=theta_true_,
        truth_color='black',
        linestyle='--',
        truth_kwargs={"linestyle": (0, (5, 5)), "linewidth": 1},
        label_kwargs={"fontsize": 12},
        contour_kwargs={"linestyles": "-", "alpha": 0.2},
        hist_kwargs={"linewidth": 2, "density": True},
        data_kwargs={"ms": 10},
    )
    if len(theta_true) == 1:
        # Draw on the corner plot's own axes rather than whatever pyplot
        # currently considers the "current axes" (avoids surprises if the
        # caller has other figures open).
        fig.axes[0].axvline(theta_true[0], color='black')
    return fig
