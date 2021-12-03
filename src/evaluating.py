import numpy as np
from sklearn.metrics import f1_score

class Evaluator(object):
    def __init__(self, label_dict):
        self.label_dict = label_dict

    def f1_score_func(self, preds, labels):
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return f1_score(labels_flat, preds_flat, average='weighted')

    def accuracy_per_class(self, preds, labels):
        label_dict_inverse = {v: k for k, v in self.label_dict.items()}

        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()

        for label in np.unique(labels_flat):
            y_preds_correct = preds_flat[
                labels_flat == label]  # the number of times our predictions were correct for a given label
            y_true = labels_flat[labels_flat == label]  # total number of times we should have predicted a given label
            print(f'Class: {label_dict_inverse[label]}')
            print(f'Accuracy: {len(y_preds_correct[y_preds_correct == label])}/{len(y_true)}\n')

