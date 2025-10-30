import torchmetrics

def get_binary_classification_metrics(num_classes=2, threshold=0.5):
    metrics = torchmetrics.MetricCollection({
        'accuracy': torchmetrics.Accuracy(task="binary", threshold=threshold),
        'precision': torchmetrics.Precision(task="binary", threshold=threshold),
        'recall': torchmetrics.Recall(task="binary", threshold=threshold),
        'f1_score': torchmetrics.F1Score(task="binary", threshold=threshold),
        'auroc': torchmetrics.AUROC(task="binary")
    })
    return metrics