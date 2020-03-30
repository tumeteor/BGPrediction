import numpy as np
import os
import ExperimentData
import matplotlib.pyplot as plt
import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cmap = plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    try:
        thresh = cm.max() / 2.
    except ValueError:  # raised if `y` is empty.
        pass
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def save_confusion_matrix(cnf_matrix, classes, patientId, desc, model):
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes,
                          title='Confusion matrix, without normalization')
    dir = "prediction/tmp/" + str(ExperimentData.curExpId)
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(dir + "/cm_{}_{}_{}.png".format(model, patientId, desc))

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes, normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig(dir + "/cm_{}_{}_norm{}.png".format(model, patientId, desc))
    plt.close()
