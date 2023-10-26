#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.stats import sem
from sklearn.metrics import roc_auc_score
import scipy.stats
from scipy import stats
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
from keras.models import load_model
import os, os.path
import cv2
import glob
import pandas as pd
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
# from keras.preprocessing import image
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout
from datetime import datetime
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, f_regression
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
import pickle
import json
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from scipy.stats import norm
from sklearn.metrics import auc
import matplotlib.pyplot as plt





def fastDeLong_no_weights(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating
              Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov






def fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight):
        """
        The fast version of DeLong's method for computing the covariance of
        unadjusted AUC.
        Args:
           predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
              sorted such as the examples with label "1" are first
        Returns:
           (AUC value, DeLong covariance)
        Reference:
         @article{sun2014fast,
           title={Fast Implementation of DeLong's Algorithm for
                  Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
           author={Xu Sun and Weichao Xu},
           journal={IEEE Signal Processing Letters},
           volume={21},
           number={11},
           pages={1389--1393},
           year={2014},
           publisher={IEEE}
         }
        """
        # Short variables are named as they are in the paper
        m = label_1_count
        n = predictions_sorted_transposed.shape[1] - m
        positive_examples = predictions_sorted_transposed[:, :m]
        negative_examples = predictions_sorted_transposed[:, m:]
        k = predictions_sorted_transposed.shape[0]

        tx = np.empty([k, m], dtype=np.float)
        ty = np.empty([k, n], dtype=np.float)
        tz = np.empty([k, m + n], dtype=np.float)
        for r in range(k):
            tx[r, :] = compute_midrank_weight(positive_examples[r, :], sample_weight[:m])
            ty[r, :] = compute_midrank_weight(negative_examples[r, :], sample_weight[m:])
            tz[r, :] = compute_midrank_weight(predictions_sorted_transposed[r, :], sample_weight)
        total_positive_weights = sample_weight[:m].sum()
        total_negative_weights = sample_weight[m:].sum()
        pair_weights = np.dot(sample_weight[:m, np.newaxis], sample_weight[np.newaxis, m:])
        total_pair_weights = pair_weights.sum()
        aucs = (sample_weight[:m]*(tz[:, :m] - tx)).sum(axis=1) / total_pair_weights
        v01 = (tz[:, :m] - tx[:, :]) / total_negative_weights
        v10 = 1. - (tz[:, m:] - ty[:, :]) / total_positive_weights
        sx = np.cov(v01)
        sy = np.cov(v10)
        delongcov = sx / m + sy / n
        return aucs, delongcov



def compute_ground_truth_statistics(ground_truth, sample_weight):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight

def fastDeLong(predictions_sorted_transposed, label_1_count, sample_weight):
    if sample_weight is None:
        return fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)
    else:
        return fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight)

def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2

def compute_midrank_weight(x, sample_weight):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]

    cumulative_weight = np.cumsum(sample_weight[J])

    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j
    T2 = np.empty(N, dtype=np.float)
    T2[J] = T
    return T2

def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)





def delong_roc_variance(ground_truth, predictions, sample_weight=None):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
        ground_truth, sample_weight)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count, ordered_sample_weight)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


def ci_(tp, n, alpha=0.05):
    """ Estimates confidence interval for Bernoulli p
    Args:
      tp: number of positive outcomes, TP in this case
      n: number of attemps, TP+FP for Precision, TP+FN for Recall
      alpha: confidence level
    Returns:
      Tuple[float, float]: lower and upper bounds of the confidence interval
    """
    p_hat = float(tp) / n
    z_score = norm.isf(alpha * 0.5)  # two sides, so alpha/2 on each side
    variance_of_sum = p_hat * (1-p_hat) / n
    std = variance_of_sum ** 0.5
    return p_hat - z_score * std, p_hat + z_score * std


def auc_keras_(fpr_keras, tpr_keras):
    auc_keras = auc(fpr_keras, tpr_keras)

    return auc_keras

class Uncertainty:
    


    def set_data (y_pred ,y_test ,visualization=True ) :
            ## Autoencoder
       # y_pred = np.array([0.21, 0.32, 0.63, 0.35, 0.92, 0.79, 0.82, 0.99, 0.04])
       # y_test = np.array([0,    1,    0,    0,    1,    1,    0,    1,    0   ])

        #y_pred = np.array(y_pred)
        #y_test = np.array(y_test)

        tag = 'test'


        cmtx = pd.DataFrame(
            confusion_matrix(y_test.round(), y_pred.round(), labels=[1, 0]), 
            index=['true:yes', 'true:no'], 
            columns=['pred:yes', 'pred:no']
        )


        ###############  Precision
        precision = precision_score(y_test.round(), y_pred.round()) #  average=Nonefor precision from each class
        ############### Recall
        recall = recall_score(y_test.round(), y_pred.round())
        # ############### F1 score
        f1_score1 = f1_score(y_test.round(), y_pred.round())
        # ############### Cohen's kappa
        cohen_kappa_score1 = cohen_kappa_score(y_test.round(), y_pred.round())

        confusion_m = confusion_matrix(y_test.round(), y_pred.round())

        TP = confusion_m[1][1]
        FP = confusion_m[0][1]
        TN = confusion_m[0][0]
        FN = confusion_m[1][0]

        n = TP+FP
        Precision_CI = ci_(TP, n, alpha=0.05)

        n = TP+FN
        Recall_CI = ci_(TP, n, alpha=0.05)

        fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred)

        auc_keras = auc_keras_(fpr_keras, tpr_keras)

       


        alpha = .95

        y_true = np.array(y_test)#np.array([0,    1,    0,    0,    1,    1,    0,    1,    0   ])


        auc, auc_cov = delong_roc_variance(
            y_true,
            y_pred)

        auc_std = np.sqrt(auc_cov)
        lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)

        ci = stats.norm.ppf(
            lower_upper_q,
            loc=auc,
            scale=auc_std)

        ci[ci > 1] = 1

        #out_ = (y_pred, y_test,tag,precision,Precision_CI,recall,Recall_CI, auc ,auc_cov ,ci , fpr_keras, tpr_keras ,auc_keras ,cmtx)
        
        if visualization ==True :
            # ### Bootstrap


        

            y_true = np.array(y_test)#np.array([0,    1,    0,    0,    1,    1,    0,    1,    0   ])

            print("Original ROC area: {:0.3f}".format(roc_auc_score(y_true, y_pred)))

            n_bootstraps = 1000
            rng_seed = 42  # control reproducibility
            bootstrapped_scores = []

            rng = np.random.RandomState(rng_seed)
            for i in range(n_bootstraps):
                # bootstrap by sampling with replacement on the prediction indices
                indices = rng.randint(0, len(y_pred), len(y_pred))
                if len(np.unique(y_true[indices])) < 2:
                    # We need at least one positive and one negative sample for ROC AUC
                    # to be defined: reject the sample
                    continue

                score = roc_auc_score(y_true[indices], y_pred[indices])
                bootstrapped_scores.append(score)
                #print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))


            plt.hist(bootstrapped_scores, bins=50)
            plt.title('Histogram of the bootstrapped ROC AUC scores')
            plt.show()

            sorted_scores = np.array(bootstrapped_scores)
            sorted_scores.sort()

            # Computing the lower and upper bound of the 90% confidence interval
            # You can change the bounds percentiles to 0.025 and 0.975 to get
            # a 95% confidence interval instead.
            confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
            confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]

            print("Bootstrap")
            print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
                confidence_lower, confidence_upper))


            # ### Classification Report




            ########## Print All Together
            print('####################')
            print("Results for "+tag)
            print(cmtx)
            print("Precision: ",precision)
            print("Precision_CI: ",Precision_CI)
            print("Recall: ",recall)
            print("Recall_CI: ",Recall_CI)
            # print("F1 Score: ",f1_score1)
            # print("Cohen_Kappa Score: ",cohen_kappa_score1)
            print("Delong Method")
            print('AUC:', auc)
            print('AUC COV:', auc_cov)
            print('95% AUC CI:', ci)


            # ### AUC Plot




            plt.figure(1)
            plt.plot([0, 1], [0, 1], 'k--', label="chance level (AUC = 0.5)")
            plt.plot(fpr_keras, tpr_keras, label='keras (AUC = {:.3f})'.format(auc_keras))
            # plt.plot(precision, recall, label='Keras (area = {:.3f})'.format(auc_keras))
            # plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')
            plt.show()
            
        else :
            return (y_pred, y_test,tag,precision,Precision_CI,recall,Recall_CI, auc ,auc_cov ,ci , fpr_keras, tpr_keras ,auc_keras ,cmtx)
            