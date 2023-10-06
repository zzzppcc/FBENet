import os
import cv2
import numpy as np
import evaluation
datasets = ["ECSSD","PASCAL-S","DUTS","HKU-IS","DUT-OMRON"]
for dataset in datasets:
    saliency_evaluation = evaluation.SaliencyEvaluation()
    saliency_evaluation.clear()
    with open(os.path.join("data",dataset,"test.txt")) as f:
        for line in f:
            gt_path = os.path.join("data",dataset,'mask',line.replace("\n","")+".png")
            pred_path = os.path.join("predict/b3net",dataset,line.replace("\n","")+".png")
            gt = cv2.imread(gt_path,cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_path,cv2.IMREAD_GRAYSCALE)
            if pred is not None and gt.shape==pred.shape:
                saliency_evaluation.add_one(
                        pred.astype(np.float),gt.astype(np.float)
                    )
    MAE, Precision, Recall, F_m, S_m, E_m = saliency_evaluation.get_evaluation()
    idx = np.argmax(F_m)
    best_F = F_m[idx]
    mean_F = np.mean(F_m)
    best_precison = Precision[idx]
    best_recall = Recall[idx]
    print('{} - MAE:{}, max F-Measure:{}, mean F-Measure:{}, Precision:{},'
            ' Recall:{}, S-Measure: {}, E-Measure: {}'
            .format(dataset, MAE, best_F, mean_F, best_precison, best_recall,
                    S_m, E_m))
