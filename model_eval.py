"""
Feb 15 2019
Methods for model evaluation.
"""
import numpy as np
from sklearn import metrics


def auc(actual, pred_prob, pos_label):
    fpr, tpr, thresholds = metrics.roc_curve(
            actual,
            pred_prob,
            pos_label=pos_label)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc
