import argparse
import os

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import random
import time
import datetime
import re
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from keras_preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertForSequenceClassification
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=100, help='Specify the test batch size')
parser.add_argument('--prompts_path', type=str, help='Load path to test prompts as text')
parser.add_argument('--resps_path', type=str, help='Load path to test responses as text')
parser.add_argument('--labels_path', type=str, help='Load path to labels')
parser.add_argument('--model_path', type=str, help='Load path to trained model')
parser.add_argument('--predictions_save_path', type=str, help="Where to save predicted values")

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def send_to_device(device, x):
    x = x.clone().detach()
    x = x.long()
    x = x.to(device)
    return x

def get_default_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    return device

def load_trained_model(args, device):
    model = torch.load(args.model_path)
    model.eval().to(device)
    tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-small", do_lower_case=True)
    return model, tokenizer

def load_dataset(tokenizer, args):
    encoded_prompts, encoded_responses, att_mask_prompts, att_mask_resps = [], [], [], []
    targets = np.loadtxt(args.labels_path, dtype=int)
    targets = torch.tensor(targets)
    
    with open(args.resps_path) as f0, open(args.prompts_path) as f1:
        responses, prompts = f0.readlines(), f1.readlines()
        prompts = [x.strip().lower() for x in prompts]
        responses = [x.strip().lower() for x in responses]
        print("Dataset Loaded")
        
        max_prompt_length = 256
        max_resp_length = 256
        # max_prompt_length = max([len(sentence) for sentence in prompts])
        # max_resp_length = max([len(sentence) for sentence in responses])
        
        for (prompt, response) in zip(prompts, responses):
            encoded_prompt, att_mask_prompt = encode_data(tokenizer, prompt, max_prompt_length)
            encoded_response, att_mask_resp = encode_data(tokenizer, response, max_resp_length)
            encoded_prompts.append(encoded_prompt), att_mask_prompts.append(att_mask_prompt)
            encoded_responses.append(encoded_response), att_mask_resps.append(att_mask_resp)
        print("Data Encoded")
    
    encoded_prompts, encoded_responses = torch.tensor(encoded_prompts), torch.tensor(encoded_responses)
    att_mask_prompts, att_mask_resps = torch.tensor(att_mask_prompts), torch.tensor(att_mask_resps)
    return encoded_prompts, encoded_responses, att_mask_prompts, att_mask_resps, targets
    
def encode_data(tokenizer, inputs, MAX_LEN):
    input_ids, attention_mask = [], []
    encoding = tokenizer(inputs, padding="max_length", max_length = MAX_LEN, truncation=True)
    input_ids.append(encoding["input_ids"])
    attention_mask.append(encoding["attention_mask"])
    return input_ids, attention_mask

def eval_model(args, model, device, prompt_ids, resp_ids, att_mask_prompts, att_mask_resps, targets):
    prompt_ids = prompt_ids.squeeze(1)
    att_mask_prompts = att_mask_prompts.squeeze(1)
    resp_ids = resp_ids.squeeze(1)
    att_mask_resps = att_mask_resps.squeeze(1)
    eval_dataset = TensorDataset(prompt_ids, att_mask_prompts, resp_ids, att_mask_resps, targets)
    eval_dataloader = DataLoader(eval_dataset, batch_size = args.batch_size, shuffle = False)

    model.eval()    
    y_pred_all = []
    print("Started evaluation")
    
    #enumerate(train_dataloader)
    for prompt_id, att_mask_prompt, resp_id, att_mask_resp, target in eval_dataloader:
        pr_resp, pr_resp_msk = torch.cat((prompt_id, resp_id), 1), torch.cat((att_mask_prompt, att_mask_resp), 1)  
        pr_resp, pr_resp_msk = send_to_device(device, (pr_resp)), send_to_device(device, pr_resp_msk)        
        prompt_id, att_mask_prompt, resp_id, att_mask_resp, target = send_to_device(device, prompt_id), send_to_device(device, att_mask_prompt), send_to_device(device, resp_id) , send_to_device(device, att_mask_resp), send_to_device(device, target) 
        
        with torch.no_grad():
            outputs = model(pr_resp.squeeze(1), token_type_ids=None, attention_mask=pr_resp_msk.squeeze(1), labels=target)
        logits = outputs[1]
        logits = logits.cpu()
        logits = logits.detach().numpy()
        logits = np.squeeze(logits[:, 1])
        logits = logits.tolist()
        y_pred_all += logits  
        
    y_pred_all = np.array(y_pred_all)
    # np.savetxt(args.predictions_save_path, y_pred_all)
    # print("Predictions saved")
    return targets, y_pred_all

        
def calculate_metrics(targets, y_pred_all):
    targets = 1.-targets
    y_pred = 1.-y_pred_all
    targets, y_pred = torch.tensor(targets, device = 'cpu'), torch.tensor(y_pred, device = 'cpu')
    precision, recall, _ = precision_recall_curve(targets, y_pred)
    
    print("Precision:", precision)
    print("Recall:", recall)
    
    #create precision recall curve
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color='purple')

    #add axis labels to plot
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    for i in range(len(precision)):
        print(i)
        if precision[i] == 0 or recall[i] == 0:
            precision[i] += 0.1
            recall[i] += 0.1
            
    f_score = np.amax( (1.+0.5**2) * ( (precision * recall) / (0.5**2 * precision + recall) ) )
    print("F0.5 score is:", f_score)

    #display plot
    fig.show()
    fig.savefig('/scratches/dialfs/alta/relevance/ns832/results' + '/eval_' + str(f_score) +  '_plot.jpg', bbox_inches='tight', dpi=150)

    
def main(args):
    device = get_default_device()
    model, tokenizer = load_trained_model(args, device)
    prompt_ids, resp_ids, att_mask_prompts, att_mask_resps, targets = load_dataset(tokenizer, args)
    targets, y_pred_all = eval_model(args, model, device, prompt_ids, resp_ids, att_mask_prompts, att_mask_resps, targets)
    calculate_metrics(targets, y_pred_all)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args) 