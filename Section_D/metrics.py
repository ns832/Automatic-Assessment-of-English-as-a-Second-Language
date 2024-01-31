import torch
import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def calculate_normalised_metrics(targets, y_pred_all, r=0.5):
    targets = 1.-targets
    y_preds = 1.-y_pred_all
    precision, recall, thresholds = precision_recall_curve(targets, y_preds)
    
    for i in range(len(precision)):
        if precision[i] == 0 or recall[i] == 0:
            precision[i] += 0.01
            recall[i] += 0.01
    
    f_score_orig = (np.amax((1.+0.5**2) * ((precision * recall) / (0.5**2 * precision + recall))))
    targets, y_preds = torch.tensor(targets, device = 'cpu'), torch.tensor(y_preds, device = 'cpu')
    p_list = precision**(-1) - 1
    n_list = recall**(-1) - 1
    
    # Add in the last threshold that the p-r curve doesn't output doesn't include
    thresholds = np.concatenate([thresholds, [y_preds.max()]]) 
    f_scores = []
    
    for threshold, p, n in zip(thresholds, p_list, n_list):
        tp, fp, tn, fn = 0, 0, 0, 0
        for target, y_pred in zip(targets, y_preds):
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
        k = (h * (r**(-1) - 1))
        f_scores.append((1.+0.5**2) / ((1.+0.5**2) + (0.5**2)* n + p * k))
        
    f_score = max(f_scores)
    
    print("Normalised F0.5 scores are:", f_score)
    print("Original F0.5 scores are:", f_score_orig)
    save_predictions_prompt(f_score, targets, y_pred_all)
    

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
    save_image_prompt(f_score, fig)
    save_predictions_prompt(f_score, targets, y_pred_all)

def save_image_prompt(f_score, fig):
    while True:
        user_input = input("Do you want to save the image? (y/n): ").lower()

        if user_input == 'y':
            print("Image will be saved.")
            file_path_model_predictions = ('/scratches/dialfs/alta/relevance/ns832/results' + '/' + str(f_score) +  '_plot.jpg')
            user_input = input("Default file name is: " + file_path_model_predictions + "\n If a different file path is desired specify here, else press enter.").lower()
            if user_input: file_path_model_predictions = user_input
            fig.savefig(file_path_model_predictions, bbox_inches='tight', dpi=150)
            print("Saved image at: ", file_path_model_predictions)
            break
        elif user_input == 'n':
            print("Image will not be saved.")
            break            
        else:
            print("Invalid response. Please enter 'y' or 'n'.")
            

def save_predictions_prompt(f_score, targets, y_pred_all):
    while True:
        user_input = input("Do you want to save the predictions and targets? (y/n): ").lower()

        if user_input == 'y':
            print("Predictions and targets will be saved.")
            file_path_model_predictions = ('/scratches/dialfs/alta/relevance/ns832/results' + '/model_predictions_'+ str(f_score) + '.jpg')
            file_path_targets = ('/scratches/dialfs/alta/relevance/ns832/results' + '/targets_'+ str(f_score) + '.jpg')
            user_input_predictions = input("Default file name for predictions is: " + file_path_model_predictions + "\n If a different file path is desired specify here, else press enter.").lower()
            user_input_targets = input("Default file name for targets is: " + file_path_targets + "\n If a different file path is desired specify here, else press enter.").lower()
            if user_input: file_path_model_predictions = user_input_predictions
            if user_input: file_path_targets= user_input_targets
            np.savetxt(file_path_model_predictions, y_pred_all)
            np.savetxt(file_path_targets, targets)
            print("Saved both at: \n", file_path_model_predictions, "\n", file_path_targets)
            break
        elif user_input == 'n':
            print("Files will not be saved.")
            break
        else:
            print("Invalid response. Please enter 'y' or 'n'.")

