import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
import matplotlib.pyplot as plt
import os


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def multiplots(predictions_list, targets_list, keys_list):
    """
        Plots multiple different precision-recall curves on the same graph.
        The predictions and targets are fed in from saved files.
    """
    fig, ax = plt.subplots()
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    for predictions, targets, key in zip(predictions_list, targets_list, keys_list):
        targets = 1.-targets
        predictions = 1.-predictions
        PrecisionRecallDisplay.from_predictions(targets, predictions, ax=ax, name=key, drawstyle="default", pos_label=0)
    ax.legend()
    save_image_prompt(" ", fig)
    

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
    
    # print("Precision:", precision)
    # print("Recall:", recall)
    
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
    # save_image_prompt(f_score, fig)
    # save_predictions_prompt(f_score, targets, y_pred_all)
    return f_score
    

def save_image_prompt(f_score, fig):
    while True:
        user_input = input("Do you want to save the image? (y/n): ").lower()

        if user_input == 'y':
            # Get file name
            file_name = str(f_score) + "_plot.jpg"
            user_input = input("Image will be saved as: " + file_name + "\n If a different name is desired specify it here, else press enter:")
            if user_input: file_name = user_input
            
            # Get file directory
            directory = ('/scratches/dialfs/alta/relevance/ns832/results/predictions/' )
            while True:
                user_input = input("Default directory is: " + directory + "\n If a different path is desired specify the directory here, else press enter.").lower()
                if user_input:
                    if os.path.exists(user_input) and os.path.isdir(user_input):
                        directory = user_input
                        break
                    else:
                        print("Directory does not exist, try again")
                else:break
            fig.savefig(directory + file_name, bbox_inches='tight', dpi=150)
            print("Saved image at: " + directory + file_name)
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
            # Get file name
            pred_file_name = str(f_score) + "_predictions.txt"
            user_input = input("Predictions will be saved as: " + pred_file_name + "\n If a different name is desired specify it here, else press enter:")
            if user_input: pred_file_name = user_input
            targets_file_name = str(f_score) + "_targets.txt"
            user_input = input("Targets will be saved as: " + targets_file_name + "\n If a different name is desired specify it here, else press enter:")
            if user_input: targets_file_name = user_input
            
            # Get file directory
            directory = ('/scratches/dialfs/alta/relevance/ns832/results/predictions/' )
            while True:
                user_input = input("Default directory is: " + directory + "\n If a different path is desired specify the directory here, else press enter.").lower()
                if user_input:
                    if os.path.exists(user_input) and os.path.isdir(user_input):
                        directory = user_input
                        break
                    else:
                        print("Directory does not exist, try again")
                else:break
            np.savetxt(directory + pred_file_name, y_pred_all)
            np.savetxt(directory + targets_file_name, targets)
            print("Saved both at: \n" + directory + "\n File Names:" + pred_file_name + "    " + targets_file_name)
            break
        elif user_input == 'n':
            print("Files will not be saved.")
            break
        else:
            print("Invalid response. Please enter 'y' or 'n'.")



def save_model(model, avg_train_loss):
    """
        Saves the model so it can be accessed for evaluationg
    """
    
    while True:
        user_input = input("Do you want to save the model? (y/n): ").lower()

        if user_input == 'y':
            # Get file name
            file_name = 'classification_model_' + str(avg_train_loss) + '.pt'
            user_input = input("Model will be saved as: " + file_name + "\n If a different name is desired specify it here, else press enter:")
            if user_input: file_name = user_input
            
            # Get file directory
            directory = ('/scratches/dialfs/alta/relevance/ns832/results/' )
            while True:
                user_input = input("Default directory is: " + directory + "\n If a different path is desired specify the directory here, else press enter.").lower()
                if user_input:
                    if os.path.exists(user_input) and os.path.isdir(user_input):
                        directory = user_input
                        break
                    else:
                        print("Directory does not exist, try again")
                else:break
            torch.save(model.state_dict(), directory + file_name)    
            print("Saved model at: " + directory + file_name)
            break
        elif user_input == 'n':
            print("Model will not be saved.")
            break            
        else:
            print("Invalid response. Please enter 'y' or 'n'.")
            
    