from torchmetrics.classification import MultilabelPrecision, MultilabelF1Score, MultilabelAccuracy, MultilabelAUROC,MultilabelRecall
from torchmetrics.classification import BinaryAccuracy,BinaryAUROC,BinaryRecall,BinaryPrecision,BinaryF1Score
import numpy as np
import torch

def classification_binary_metrics(predictions,labels):
    accuracy=BinaryAccuracy()
    f1=BinaryF1Score()
    precision=BinaryPrecision()
    recall=BinaryRecall()
    auc=BinaryAUROC()

    return accuracy,f1,precision,recall,auc


def classification_multilabel_metrics(predictions, labels):
    acc = MultilabelAccuracy(num_labels=12, average='weighted')
    f1 = MultilabelF1Score(num_labels=12, average='micro')
    precision = MultilabelPrecision(num_labels=12, average='micro')
    recall=MultilabelRecall(num_labels=12,average='micro')
    auc=MultilabelAUROC(num_labels=12,average='micro')

    label_accuracy = acc(predictions, labels)
    f1_micro = f1(predictions, labels)
    prec = precision(predictions, labels)
    rec=recall(predictions,labels)
    auc=recall(predictions,labels)
    
    return label_accuracy, f1_micro, prec,rec,auc