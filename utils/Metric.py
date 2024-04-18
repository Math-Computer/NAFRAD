import numpy as np
from sklearn import metrics

def segment_iou_score(pred, label):
    pred = pred.reshape(1, -1).squeeze()
    label = label.reshape(1, -1).squeeze()
    cm = metrics.confusion_matrix(label, pred)
    intersection = np.diag(cm) # 交集
    union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - intersection # 并集
    IoU = intersection / union # 交并比，即IoU
    return IoU


if __name__ == '__main__':
    import torch
    predict = torch.ones(4, 256, 256)
    target = torch.zeros(4, 256, 256)
    iou = segment_iou_score(predict, target)
    print(iou)