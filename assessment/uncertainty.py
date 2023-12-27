"""
---------------------------------------------------------------------------
Created on Fri April  10 10:32:32 2023

----------------------------------------------------------------------------

**Title:**        ValidPath Toolbox - Uncertainty Analysis module

**Description:**  This is the Uncertainty Analysis module of the ValidPath toolbox. It is includes Uncertainty_Analysis class and several methods
              
**Classes:**      Uncertainty_Analysis
              

**Methods:**      get_report, auc_keras_, ci_, Delong_CI, compute_midrank, compute_midrank_weight, calc_pvalue, compute_ground_truth_statistics, delong_roc_variance, bootstrapping

---------------------------------------------------------------------------
Author: SeyedM.MousaviKahaki (seyed.kahaki@fda.hhs.gov)
Version ='1.0'
---------------------------------------------------------------------------
"""
import numpy as np
from sklearn.metrics import roc_auc_score
import scipy.stats
from scipy import stats
import os, os.path
import pandas as pd
from sklearn.metrics import roc_curve
#from sklearn.metrics import precision_recall_curve
#from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from scipy.stats import norm
from sklearn.metrics import auc
import matplotlib.pyplot as plt 


class Uncertainty_Analysis:
    def __init__(self):
        self.perform_Delong = False
        self.perform_Bootstrap = False
        self.plot_roc = False
        self.tag = 'My Results'
        self.n_bootstraps = 1000
        pass


    def get_report (self, y_pred ,y_truth) :
        """ This method recieve the machine learning prediction output and the ground truth and report several metrics. This is the main metod of the Uncertainty_Analysis class which calls other methods to procude results.
        
        :Parameters:
           y_truth: ground_truth -  np.array of 0 and 1
           y_pred:  predictions - np.array of floats of the probability of being class 1
        :Returns:
            precision
            Precision Conficenc Interval
            Recall
            Recall Conficenc Interval
            AUC based on delong method and its Conficenc Interval and COV 
            False Positive Rate
            True Positive Rate
            AUC
            Confusion Matrix
                        
        """

        cmtx = pd.DataFrame(
            confusion_matrix(y_truth.round(), y_pred.round(), labels=[1, 0]), 
            index=['true:yes', 'true:no'], 
            columns=['pred:yes', 'pred:no']
        )


        ###############  Precision
        precision = precision_score(y_truth.round(), y_pred.round())
        ############### Recall
        recall = recall_score(y_truth.round(), y_pred.round())
        # ############### F1 score
        f1_score1 = f1_score(y_truth.round(), y_pred.round())
        # ############### Cohen's kappa
        cohen_kappa_score1 = cohen_kappa_score(y_truth.round(), y_pred.round())

        confusion_m = confusion_matrix(y_truth.round(), y_pred.round())

        TP = confusion_m[1][1]
        FP = confusion_m[0][1]
        TN = confusion_m[0][0]
        FN = confusion_m[1][0]

        n = TP+FP
        Precision_CI = self.ci_(TP, n, alpha=0.05)

        n = TP+FN
        Recall_CI = self.ci_(TP, n, alpha=0.05)

        fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_truth, y_pred)

        auc_keras = self.auc_keras_(fpr_keras, tpr_keras)

        if self.perform_Delong:
            auc_delong, ci_delong, lower_upper_q, auc_delong_cov, auc_std = self.Delong_CI(y_pred ,y_truth)
        else:
            auc_delong = ci_delong = lower_upper_q = auc_delong_cov = auc_std = []

        ########## Print All Together
        print('####################')
        print("Results for "+self.tag)
        print(cmtx)
        print("Precision: ",precision)
        print("Precision_CI: ",Precision_CI)
        print("Recall: ",recall)
        print("Recall_CI: ",Recall_CI)
        if self.perform_Delong:
            print("Delong Method")
            print('AUC:', auc_delong)
            print('AUC COV:', auc_delong_cov)
            print('95% AUC CI:', ci_delong)
        # ### Bootstrap
        if self.perform_Bootstrap:
            bootstrapped_scores, confidence_lower, confidence_upper = self.bootstrapping(y_truth, y_pred)
        else:
            bootstrapped_scores =  confidence_lower =  confidence_upper = {}
                
        if self.plot_roc ==True :
            # ### AUC Plot
            plt.figure(1)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr_keras, tpr_keras, label='keras (AUC = {:.3f})'.format(auc_keras))
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')
            plt.show()
            
        
        return (precision,Precision_CI,recall,Recall_CI, auc_delong ,auc_delong_cov ,ci_delong , fpr_keras, tpr_keras ,auc_keras ,cmtx)

    def auc_keras_(self, fpr_keras, tpr_keras):
        
        """ Estimates confidence interval for Bernoulli p
        
        :Parameters:
          fpr_keras: False Positive Rate Values
          tpr_keras: True Positive Rate Values

        
        :Returns:
          AUC: Area Under the ROC Curve
        """
        
        auc_keras = auc(fpr_keras, tpr_keras)

        return auc_keras
    
    def ci_(self, tp, n, alpha=0.05):
        """ Estimates confidence interval for Bernoulli p
        
        :Parameters:
          tp: number of positive outcomes, TP in this case
          n: number of attemps, TP+FP for Precision, TP+FN for Recall
          alpha: confidence level
        
        :Returns:
          Tuple[float, float]: lower and upper bounds of the confidence interval
        """
        p_hat = float(tp) / n
        z_score = norm.isf(alpha * 0.5)  # two sides, so alpha/2 on each side
        variance_of_sum = p_hat * (1-p_hat) / n
        std = variance_of_sum ** 0.5
        return p_hat - z_score * std, p_hat + z_score * std
    
    def Delong_CI(self, y_pred ,y_truth):
        """ A Python implementation of an algorithm for computing the statistical significance of comparing two sets of predictions by ROC AUC. 
        Also can compute variance of a single ROC AUC estimate. X. Sun and W. Xu, "Fast Implementation of DeLong’s Algorithm for Comparing the 
        Areas Under Correlated Receiver Operating Characteristic Curves," in IEEE Signal Processing Letters, vol. 21, no. 11, pp. 1389-1393, Nov. 2014, 
        doi: 10.1109/LSP.2014.2337313.
        
        :Parameters:
           y_truth: ground_truth -  np.array of 0 and 1
           y_pred:  predictions - np.array of floats of the probability of being class 1
        :Returns:
            auc, ci, lower_upper_q, auc_cov, auc_std
        
        """
        alpha = .95
        
        y_true = np.array(y_truth)

        auc, auc_cov = self.delong_roc_variance(
            y_true,
            y_pred)

        auc_std = np.sqrt(auc_cov)
        lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)

        ci = stats.norm.ppf(
            lower_upper_q,
            loc=auc,
            scale=auc_std)

        ci[ci > 1] = 1
        return auc, ci, lower_upper_q, auc_cov, auc_std
    
    # stackoverflow.com/questions/19124239/scikit-learn-roc-curve-with-confidence-intervals
    def compute_midrank(self,x):
        """Computes midranks.
        
        :Parameters:
           x - a 1D numpy array
        
        :Returns:
           array of midranks
        """
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=float)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5*(i + j - 1)
            i = j
        T2 = np.empty(N, dtype=float)
        # Note(kazeevn) +1 is due to Python using 0-based indexing
        # instead of 1-based in the AUC formula in the paper
        T2[J] = T + 1
        return T2
        
    def compute_midrank_weight(self, x, sample_weight):
        """Computes midranks.
        
        :Parameters:
           x - a 1D numpy array
           
        :Returns:
           array of midranks
        """
        J = np.argsort(x)
        Z = x[J]
        cumulative_weight = np.cumsum(sample_weight[J])
        N = len(x)
        T = np.zeros(N, dtype=float)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = cumulative_weight[i:j].mean()
            i = j
        T2 = np.empty(N, dtype=float)
        T2[J] = T
        return T2
    
    
    def fastDeLong(self, predictions_sorted_transposed, label_1_count, sample_weight):
        if sample_weight is None:
            return self.fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)
        else:
            return self.fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight)
    
    def fastDeLong_weights(self, predictions_sorted_transposed, label_1_count, sample_weight):
        """
        The fast version of DeLong's method for computing the covariance of
        unadjusted AUC.
        
        :Parameters:
           predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
              sorted such as the examples with label "1" are first
        :Returns:
           (AUC value, DeLong covariance)
           
        :Reference:
         @article{sun2014fast,
           title={Fast Implementation of DeLong's Algorithm for Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
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

        tx = np.empty([k, m], dtype=float)
        ty = np.empty([k, n], dtype=float)
        tz = np.empty([k, m + n], dtype=float)
        for r in range(k):
            tx[r, :] = self.compute_midrank_weight(positive_examples[r, :], sample_weight[:m])
            ty[r, :] = self.compute_midrank_weight(negative_examples[r, :], sample_weight[m:])
            tz[r, :] = self.compute_midrank_weight(predictions_sorted_transposed[r, :], sample_weight)
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
    
    
    def fastDeLong_no_weights(self, predictions_sorted_transposed, label_1_count):
        """
        The fast version of DeLong's method for computing the covariance of
        unadjusted AUC.
        
        :Parameters:
           predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
              sorted such as the examples with label "1" are first
        
        :Returns:
           (AUC value, DeLong covariance)
        
        Reference:
         @article{sun2014fast,
           title={Fast Implementation of DeLong's Algorithm for Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
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

        tx = np.empty([k, m], dtype=float)
        ty = np.empty([k, n], dtype=float)
        tz = np.empty([k, m + n], dtype=float)
        for r in range(k):
            tx[r, :] = self.compute_midrank(positive_examples[r, :])
            ty[r, :] = self.compute_midrank(negative_examples[r, :])
            tz[r, :] = self.compute_midrank(predictions_sorted_transposed[r, :])
        aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
        v01 = (tz[:, :m] - tx[:, :]) / n
        v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
        sx = np.cov(v01)
        sy = np.cov(v10)
        delongcov = sx / m + sy / n
        return aucs, delongcov
    
    
    def calc_pvalue(self, aucs, sigma):
        """Computes log(10) of p-values.
        
        :Parameters:
           aucs: 1D array of AUCs
           sigma: AUC DeLong covariances
        
        :Returns:
           log10(pvalue)
        """
        l = np.array([[1, -1]])
        z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
        return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)
        
        
    def compute_ground_truth_statistics(self, ground_truth, sample_weight):
        assert np.array_equal(np.unique(ground_truth), [0, 1])
        order = (-ground_truth).argsort()
        label_1_count = int(ground_truth.sum())
        if sample_weight is None:
            ordered_sample_weight = None
        else:
            ordered_sample_weight = sample_weight[order]

        return order, label_1_count, ordered_sample_weight
        
        
    def delong_roc_variance(self, ground_truth, predictions, sample_weight=None):
        """
        Computes ROC AUC variance for a single set of predictions
        
        :Parameters:
           ground_truth: np.array of 0 and 1
           predictions: np.array of floats of the probability of being class 1
        """
        order, label_1_count, ordered_sample_weight = self.compute_ground_truth_statistics(
            ground_truth, sample_weight)
        predictions_sorted_transposed = predictions[np.newaxis, order]
        aucs, delongcov = self.fastDeLong(predictions_sorted_transposed, label_1_count, ordered_sample_weight)
        assert len(aucs) == 1
        return aucs[0], delongcov
        
        
    def bootstrapping(self, y_true, y_pred):
        """
        Computes ROC AUC variance for a single set of predictions
        
        :Parameters:
           ground_truth: np.array of 0 and 1
           predictions: np.array of floats of the probability of being class 1
        """
        print("Original ROC area: {:0.3f}".format(roc_auc_score(y_true, y_pred)))
        
        rng_seed = 42  # control reproducibility
        bootstrapped_scores = []

        rng = np.random.RandomState(rng_seed)
        for i in range(self.n_bootstraps):
            # bootstrap by sampling with replacement on the prediction indices
            indices = rng.randint(0, len(y_pred), len(y_pred))
            if len(np.unique(y_true[indices])) < 2:
                # We need at least one positive and one negative sample for ROC AUC
                # to be defined: reject the sample
                continue

            score = roc_auc_score(y_true[indices], y_pred[indices])
            bootstrapped_scores.append(score)
            print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))


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
        return bootstrapped_scores, confidence_lower, confidence_upper
        
    