import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize


def plot_cm(all_true, all_preds, classes):
    all_true = [classes[t] for t in all_true]
    all_preds = [classes[t] for t in all_preds]
    cm = confusion_matrix(all_true, all_preds, labels=classes)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')


def plot_prc(all_true, all_preds, classes):
    # Binarize the labels
    y_test_binarized = label_binarize(all_true, classes=[0, 1, 2])
    y_score_binarized = label_binarize(all_preds, classes=[0, 1, 2])

    fpr = dict()
    tpr = dict()

    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score_binarized[:, i])
        plt.plot(fpr[i], tpr[i], lw=2, label='Class {}'.format(i))

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="best")
