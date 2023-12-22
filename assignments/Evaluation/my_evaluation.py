import numpy as np
import pandas as pd
from collections import Counter

class my_evaluation:
    # Binary class or multi-class classification evaluation
    # Each data point can only belong to one class

    def __init__(self, predictions, actuals, pred_proba=None):
        # inputs:
        # predictions: list of predicted classes
        # actuals: list of ground truth
        # pred_proba: pd.DataFrame of prediction probability of belonging to each class
        self.predictions = np.array(predictions)
        self.actuals = np.array(actuals)
        self.pred_proba = pred_proba
        if type(self.pred_proba) == pd.DataFrame:
            self.classes_ = list(self.pred_proba.keys())
        else:
            self.classes_ = list(set(list(self.predictions) + list(self.actuals)))
        self.confusion_matrix = None

    def confusion(self):
        # compute confusion matrix for each class in self.classes_
        # self.confusion = {self.classes_[i]: {"TP":tp, "TN": tn, "FP": fp, "FN": fn}}
        # no return variables
        # write your own code below
        self.confusion_matrix = {}
        for target in self.classes_:
            tp = np.sum((self.predictions == target) & (self.actuals == target))
            tn = np.sum((self.predictions != target) & (self.actuals != target))
            fp = np.sum((self.predictions == target) & (self.actuals != target))
            fn = np.sum((self.predictions != target) & (self.actuals == target))
            self.confusion_matrix[target] = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}

    def precision(self, target=None, average="macro"):
        # compute precision
        # target: target class (str). If not None, then return precision of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average precision
        # output: prec = float
        # note: be careful for divided by 0
        # write your own code below
        if self.confusion_matrix is None:
            self.confusion()
        
        if target is not None:
            tp = self.confusion_matrix[target]["TP"]
            fp = self.confusion_matrix[target]["FP"]
            if tp + fp == 0:
                return 0.0
            else:
                return tp / (tp + fp)
        else:
            precisions = []
            for target in self.classes_:
                tp = self.confusion_matrix[target]["TP"]
                fp = self.confusion_matrix[target]["FP"]
                if tp + fp == 0:
                    precisions.append(0.0)
                else:
                    precisions.append(tp / (tp + fp))
            
            if average == "macro":
                return np.mean(precisions)
            elif average == "micro":
                tp_total = np.sum([self.confusion_matrix[target]["TP"] for target in self.classes_])
                fp_total = np.sum([self.confusion_matrix[target]["FP"] for target in self.classes_])
                if tp_total + fp_total == 0:
                    return 0.0
                else:
                    return tp_total / (tp_total + fp_total)
            elif average == "weighted":
                class_counts = Counter(self.actuals)
                weighted_sum = sum(precisions[i] * class_counts[self.classes_[i]] for i in range(len(self.classes_)))
                total_count = len(self.actuals)
                if total_count == 0:
                    return 0.0
                else:
                    return weighted_sum / total_count
            else:
                raise ValueError("Invalid average parameter")

    def recall(self, target=None, average="macro"):
        # compute recall
        # target: target class (str). If not None, then return recall of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average recall
        # output: recall = float
        # note: be careful for divided by 0
        # write your own code below
        if self.confusion_matrix is None:
            self.confusion()
        
        if target is not None:
            tp = self.confusion_matrix[target]["TP"]
            fn = self.confusion_matrix[target]["FN"]
            if tp + fn == 0:
                return 0.0
            else:
                return tp / (tp + fn)
        else:
            recalls = []
            for target in self.classes_:
                tp = self.confusion_matrix[target]["TP"]
                fn = self.confusion_matrix[target]["FN"]
                if tp + fn == 0:
                    recalls.append(0.0)
                else:
                    recalls.append(tp / (tp + fn))
            
            if average == "macro":
                return np.mean(recalls)
            elif average == "micro":
                tp_total = np.sum([self.confusion_matrix[target]["TP"] for target in self.classes_])
                fn_total = np.sum([self.confusion_matrix[target]["FN"] for target in self.classes_])
                if tp_total + fn_total == 0:
                    return 0.0
                else:
                    return tp_total / (tp_total + fn_total)
            elif average == "weighted":
                class_counts = Counter(self.actuals)
                weighted_sum = sum(recalls[i] * class_counts[self.classes_[i]] for i in range(len(self.classes_)))
                total_count = len(self.actuals)
                if total_count == 0:
                    return 0.0
                else:
                    return weighted_sum / total_count
            else:
                raise ValueError("Invalid average parameter")

    def f1(self, target=None, average="macro"):
        # compute f1
        # target: target class (str). If not None, then return f1 of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average f1
        # output: f1 = float
        # note: be careful for divided by 0
        # write your own code below
        if self.confusion_matrix is None:
            self.confusion()

        if target is not None:
            if target not in self.classes_:
                raise ValueError("Target class not found.")
            precision = self.precision(target=target)
            recall = self.recall(target=target)
            return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        f1_scores = []
        for cls in self.classes_:
            precision = self.precision(target=cls)
            recall = self.recall(target=cls)
            if precision + recall > 0:
                f1_scores.append(2 * precision * recall / (precision + recall))
            else:
                f1_scores.append(0.0)

        if average == "macro":
            return np.mean(f1_scores)
        elif average == "micro":
            total_tp = sum(self.confusion_matrix[cls]["TP"] for cls in self.classes_)
            total_fn = sum(self.confusion_matrix[cls]["FN"] for cls in self.classes_)
            total_fp = sum(self.confusion_matrix[cls]["FP"] for cls in self.classes_)
            if total_tp + total_fn == 0:
                return 0.0
            return 2 * total_tp / (2 * total_tp + total_fp + total_fn)
        elif average == "weighted":
            class_counts = Counter(self.actuals)
            total = sum(class_counts.values())
            return sum(f1_scores[i] * class_counts[cls] / total for i, cls in enumerate(self.classes_))
        else:
            raise ValueError("Invalid average parameter")

    def auc(self, target):
        # compute AUC of ROC curve for the target class
        # return auc = float
        if type(self.pred_proba) == type(None):
            return None
        else:
            # write your own code below
            if target in self.classes_:
                order = np.argsort(self.pred_proba[target])[::-1]
                tp = 0
                fp = 0
                fn = Counter(self.actuals)[target]
                tn = len(self.actuals) - fn
                tpr = 0
                fpr = 0
                auc_target = 0
                for i in order:
                    if self.actuals[i] == target:
                        tp = tp + 1
                        fn = fn - 1
                        tpr = tp / (tp +fn)
                    else:
                        fp = fp + 1
                        tn = tn - 1
                        pre_fpr = fpr
                        fpr = fp / (fp+tn)
                        auc_target = auc_target + (tpr * (fpr - pre_fpr))
            else:
                raise Exception("Unknown target class.")
        return auc_target

