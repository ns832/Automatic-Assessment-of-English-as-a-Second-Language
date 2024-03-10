import eval_BERT
import argparse
import torch
import numpy as np
from sklearn.metrics import precision_recall_curve
from numpy import inf


parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size_1', type=int, default=100, help='Specify the test batch size')
parser.add_argument('--prompts_path_1', type=str, help='Load path to test prompts as text')
parser.add_argument('--responses_path_1', type=str, help='Load path to test responses as text')
parser.add_argument('--labels_path_1', type=str, help='Load path to labels')
parser.add_argument('--model_path_1', type=str, help='Load path to trained model')
parser.add_argument('--predictions_save_path_1', type=str, help="Where to save predicted values")
parser.add_argument('--batch_size_2', type=int, default=100, help='Specify the test batch size')
parser.add_argument('--prompts_path_2', type=str, help='Load path to test prompts as text')
parser.add_argument('--responses_path_2', type=str, help='Load path to test responses as text')
parser.add_argument('--labels_path_2', type=str, help='Load path to labels')
parser.add_argument('--model_path_2', type=str, help='Load path to trained model')
parser.add_argument('--predictions_save_path_2', type=str, help="Where to save predicted values")


class arguments:
    def __init__(self, batch_size, prompts_path, responses_path, labels_path, model_path, predictions_save_path):
        self.batch_size = batch_size
        self.prompts_path = prompts_path
        self.responses_path = responses_path
        self.labels_path = labels_path
        self.model_path = model_path
        self.predictions_save_path = predictions_save_path

def send_to_device(device, x):
    x = x.clone().detach()
    x = x.long()
    x = x.to(device)
    return x

def calculate_metrics(targets_1, y_pred_all_1, targets_2, y_pred_all_2):
    targets_1 = 1.-targets_1
    y_preds_1 = 1.-y_pred_all_1
    targets_2 = 1.-targets_2
    y_preds_2 = 1.-y_pred_all_2
    precision_1, recall_1, thresholds_1 = precision_recall_curve(targets_1, y_preds_1)
    precision_2, recall_2, thresholds_2 = precision_recall_curve(targets_2, y_preds_2)
    
    for i in range(len(precision_1)):
        if precision_1[i] == 0 or recall_1[i] == 0:
            precision_1[i] += 0.01
            recall_1[i] += 0.01
    
    for i in range(len(precision_2)):
        if precision_2[i] == 0 or recall_2[i] == 0:
            precision_2[i] += 0.01
            recall_2[i] += 0.01
    
    f_score_1_orig = (np.amax((1.+0.5**2) * ((precision_1 * recall_1) / (0.5**2 * precision_1 + recall_1))))
    f_score_2_orig = (np.amax((1.+0.5**2) * ((precision_2 * recall_2) / (0.5**2 * precision_2 + recall_2))))

    # r = (precision_1[0] + precision_2[0]) / 2
    r = 0.5
    
    targets_1, y_preds_1 = torch.tensor(targets_1, device = 'cpu'), torch.tensor(y_preds_1, device = 'cpu')
    targets_2, y_preds_2 = torch.tensor(targets_2, device = 'cpu'), torch.tensor(y_preds_2, device = 'cpu')
    
    p_1_list, p_2_list = precision_1**(-1) - 1, precision_2**(-1) - 1
    n_1_list, n_2_list = recall_1**(-1) - 1, recall_2**(-1) - 1
    
    
    # Add in the last threshold that the p-r curve doesn't output doesn't include
    thresholds_1 = np.concatenate([thresholds_1, [y_preds_1.max()]]) 
    thresholds_2 = np.concatenate([thresholds_2, [y_preds_2.max()]]) 
    
    f_scores_1, f_scores_2 = [], []
    
    for threshold, p_1, n_1 in zip(thresholds_1, p_1_list, n_1_list):
        tp, fp, tn, fn = 0, 0, 0, 0
        for target, y_pred in zip(targets_1, y_preds_1):
            pred = (y_pred.item() >= threshold)
            if pred == 1 and target == 1:
                tp += 1
            elif pred == 1 and target == 0:
                fp += 1
            elif pred == 0 and target == 0:
                tn += 1
            elif pred == 0 and target == 1:
                fn += 1
        # Avoid divide by zero error
        if tn + fp == 0:
            fp += 1
            fn += 1 
        # h is calculated from the ground truth positive and negative classes
        h = (tp + fn) / (tn + fp)
        k_1 = (h * (r**(-1) - 1))
        # f_scores_1.append( (1.+0.5**2) * tp / ((1.+0.5**2) * tp + (0.5**2)*k_1 * fn + fp))
        f_scores_1.append((1.+0.5**2) / ((1.+0.5**2) + (0.5**2)* n_1 + p_1 * k_1))
        
    for threshold, p_2, n_2 in zip(thresholds_2, p_2_list, n_2_list):
        tp, fp, tn, fn = 0, 0, 0, 0
        for target, y_pred in zip(targets_2, y_preds_2):
            pred = (y_pred.item() >= threshold)
            if pred == 1 and target == 1:
                tp += 1
            elif pred == 1 and target == 0:
                fp += 1
            elif pred == 0 and target == 0:
                tn += 1
            elif pred == 0 and target == 1:
                fn += 1
        # Avoid divide by zero error
        if tn + fp == 0:
            fp += 1
            fn += 1 
        # h is calculated from the ground truth positive and negative classes
        h = (tp + fn) / (tn + fp)
        k_2 = (h * (r**(-1) - 1))
        f_scores_2.append((1.+0.5**2) / ((1.+0.5**2) + (0.5**2)* n_2 + p_2 * k_2))
        # f_scores_2.append( (1.+0.5**2) * tp / ((1.+0.5**2) * tp + (0.5**2)*k_2 * fn + fp))
        
        
    f_score_1 = max(f_scores_1)
    f_score_2 = max(f_scores_2)
    
    print("Normalised F0.5 scores are:", f_score_1, f_score_2)
    print("Original F0.5 scores are:", f_score_1_orig, f_score_2_orig)


def main(args):
    args1 = arguments(args.batch_size_1, args.prompts_path_1, args.responses_path_1, args.labels_path_1, args.model_path_1, args.predictions_save_path_1)
    args2 = arguments(args.batch_size_2, args.prompts_path_2, args.responses_path_2, args.labels_path_2, args.model_path_2, args.predictions_save_path_2)

    device = eval_BERT.get_default_device()
    model_1, tokenizer_1 = eval_BERT.load_trained_model(args1, device)
    model_2, tokenizer_2 = eval_BERT.load_trained_model(args2, device)
    prompt_ids_1, resp_ids_1, att_mask_prompts_1, att_mask_responses_1, targets_1 = eval_BERT.load_dataset(tokenizer_1, args1)
    prompt_ids_2, resp_ids_2, att_mask_prompts_2, att_mask_responses_2, targets_2 = eval_BERT.load_dataset(tokenizer_2, args2)

    targets_1, predictions_1 = eval_BERT.eval_model(args1, model_1, device, prompt_ids_1, resp_ids_1, att_mask_prompts_1, att_mask_responses_1, targets_1)
    targets_2, predictions_2 = eval_BERT.eval_model(args2, model_2, device, prompt_ids_2, resp_ids_2, att_mask_prompts_2, att_mask_responses_2, targets_2)
    
    calculate_metrics(targets_1, predictions_1, targets_2, predictions_2)
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args) 