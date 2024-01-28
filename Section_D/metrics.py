import torch
import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt



device = 'cuda' if torch.cuda.is_available() else 'cpu'

def calculate_normalised_metrics(targets_1, y_pred_all_1, r=0.5):
    targets_1 = 1.-targets_1
    y_preds_1 = 1.-y_pred_all_1
    precision_1, recall_1, thresholds_1 = precision_recall_curve(targets_1, y_preds_1)
    
    for i in range(len(precision_1)):
        if precision_1[i] == 0 or recall_1[i] == 0:
            precision_1[i] += 0.01
            recall_1[i] += 0.01
    
    f_score_1_orig = (np.amax((1.+0.5**2) * ((precision_1 * recall_1) / (0.5**2 * precision_1 + recall_1))))
    targets_1, y_preds_1 = torch.tensor(targets_1, device = 'cpu'), torch.tensor(y_preds_1, device = 'cpu')
    p_1_list = precision_1**(-1) - 1
    n_1_list = recall_1**(-1) - 1
    
    # Add in the last threshold that the p-r curve doesn't output doesn't include
    thresholds_1 = np.concatenate([thresholds_1, [y_preds_1.max()]]) 
    f_scores_1 = []
    
    for threshold, p_1, n_1 in zip(thresholds_1, p_1_list, n_1_list):
        tp, fp, tn, fn = 0, 0, 0, 0
        for target, y_pred in zip(targets_1, y_preds_1):
            pred = (y_pred.item() >= threshold)
            if pred == 1 and target == 1: tp += 1
            elif pred == 1 and target == 0: fp += 1
            elif pred == 0 and target == 0: tn += 1
            elif pred == 0 and target == 1: fn += 1
        # Avoid divide by zero error
        if tn + fp == 0:
            fp += 1
            fn += 1 
        # h is calculated from the ground truth positive and negative classes
        h = (tp + fn) / (tn + fp)
        k_1 = (h * (r**(-1) - 1))
        f_scores_1.append((1.+0.5**2) / ((1.+0.5**2) + (0.5**2)* n_1 + p_1 * k_1))
        
    f_score_1 = max(f_scores_1)
    
    print("Normalised F0.5 scores are:", f_score_1)
    print("Original F0.5 scores are:", f_score_1_orig)
    

def calculate_metrics(targets, y_pred_all):
    """
        Calculates the precision and recall from the predictions and targets.
        Creates precision-recall graph and gives f-score
    """
    targets = 1.-targets
    y_pred = 1.-y_pred_all
    targets, y_pred = torch.tensor(targets, device = 'cpu'), torch.tensor(y_pred, device = 'cpu')
    precision, recall, _ = precision_recall_curve(targets, y_pred)
    
    print("Precision:", precision)
    print("Recall:", recall)
    
    # Create precision recall curve
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color='purple')

    # Add axis labels to plot
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    for i in range(len(precision)):
        if precision[i] == 0 or recall[i] == 0:
            precision[i] += 0.1
            recall[i] += 0.1
            
    f_score = np.amax( (1.+0.5**2) * ( (precision * recall) / (0.5**2 * precision + recall) ) )
    print("F0.5 score is:", f_score)

    # Display plot
    fig.show()
    print('/scratches/dialfs/alta/relevance/ns832/results' + '/LLaMA_' + str(f_score) +  '_plot.jpg')
    fig.savefig('/scratches/dialfs/alta/relevance/ns832/results' + '/LLaMA_' + str(f_score) +  '_plot.jpg', bbox_inches='tight', dpi=150)
