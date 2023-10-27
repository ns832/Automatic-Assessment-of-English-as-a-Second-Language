import argparse
import os

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import random
import time
import datetime
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from keras_preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertForSequenceClassification


parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=100, help='Specify the test batch size')
parser.add_argument('--prompts_path', type=str, help='Load path to test prompts as text')
parser.add_argument('--responses_path', type=str, help='Load path to test responses as text')
parser.add_argument('--labels_path', type=str, help='Load path to labels')
parser.add_argument('--model_path', type=str, help='Load path to trained model')
parser.add_argument('--predictions_save_path', type=str, help="Where to save predicted values")
parser.add_argument('--reverse', type=bool, default=False, help='If true, then concatenate the response onto prompt instead of other way around')


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

def load_dataset(tokenizer, device, args):
    prompt_ids, resp_ids, att_mask_prompts, att_mask_resps = [], [], [], []
    targets = np.loadtxt(args.labels_path)
    targets = torch.tensor(targets)
    
    with open(args.prompts_path) as f0, open(args.responses_path) as f1:
        prompts, responses = f0.readlines(), f1.readlines()
        prompts = [x.strip().lower() for x in prompts[:10]]
        responses = [x.strip().lower() for x in responses[:10]]
        
        max_prompt_length = max([len(sentence) for sentence in prompts])
        max_resp_length = max([len(sentence) for sentence in responses])
        max_length = max(max_prompt_length, max_resp_length)
        
        for (prompt, response) in zip(prompts, responses):
            prompt_id, att_mask = encode_data(tokenizer, prompt, max_length)
            prompt_ids.append(prompt_id), att_mask_prompts.append(att_mask)
            resp_id, att_mask = encode_data(tokenizer, response, max_length)
            resp_ids.append(resp_id), att_mask_resps.append(att_mask)
    
    print("Dataset Loaded")
    prompt_ids, resp_ids = torch.tensor(prompt_ids), torch.tensor(resp_ids)
    att_mask_prompts, att_mask_resps = torch.tensor(att_mask_prompts), torch.tensor(att_mask_resps)
    return prompt_ids, resp_ids, att_mask_prompts, att_mask_resps, targets
    
def encode_data(tokenizer, inputs, MAX_LEN):
    input_ids, attention_mask = [], []
    encoding = tokenizer(inputs, padding="max_length", max_length = MAX_LEN, truncation=True)
    input_ids.append(encoding["input_ids"])
    attention_mask.append(encoding["attention_mask"])
    return input_ids, attention_mask

def eval_model(args, model, device, prompt_ids, resp_ids, att_mask_prompts, att_mask_resps, targets):

    eval_dataset = TensorDataset(prompt_ids, att_mask_prompts, resp_ids, att_mask_resps, targets[:10])
    eval_dataloader = DataLoader(eval_dataset, batch_size = args.batch_size, shuffle = False)

    model.eval()    
    y_pred_all = []
    print("Started evaluation")
    
    for prompt_id, att_mask_prompt, resp_id, att_mask_resp, target in zip(prompt_ids, att_mask_prompts, resp_ids, att_mask_resps, targets):
        pr_resp, pr_resp_msk = torch.cat((prompt_id, resp_id), 1), torch.cat((att_mask_prompt, att_mask_resp), 1)  
        pr_resp, pr_resp_msk = send_to_device(device, (pr_resp)), send_to_device(device, pr_resp_msk)        
        prompt_id, att_mask_prompt, resp_id, att_mask_resp, target = send_to_device(device, prompt_id), send_to_device(device, att_mask_prompt), send_to_device(device, resp_id) , send_to_device(device, att_mask_resp), send_to_device(device, target) 
        
        print(pr_resp.size())
        print(pr_resp_msk.size())
        print(target.size())
        
        with torch.no_grad():
            outputs = model(pr_resp, attention_mask=pr_resp_msk, labels=target)
        # logits = (outputs[1]).to('cpu')
        logits = outputs[1]
        logits = logits.detach().to(device)
        logits = np.squeeze(logits[:, 1])
        logits = logits.tolist()
        print(logits)
        break
        y_pred_all += logits
        
    y_pred_all = np.array(y_pred_all)
    # Save the predicted values so that they can be used for ensembling
    np.savetxt(args.predictions_save_path, y_pred_all)
    print("Predictions saved")
    calculate_metrics(targets, y_pred_all)
    return

        
def calculate_metrics(targets, y_pred_all):
    targets = 1.-targets
    y_pred = 1.-y_pred_all
    targets, y_pred = torch.tensor(targets, device = 'cpu'), torch.tensor(y_pred, device = 'cpu')
    precision, recall, _ = precision_recall_curve(targets, y_pred)
    print("Precision:", precision)
    print("Recall:", recall)
    f_score = np.amax( (1.+0.5**2) * ( (precision * recall) / (0.5**2 * precision + recall) ) )
    print("F0.5 score is:", f_score)

    
def main(args):
    device = get_default_device()
    model, tokenizer = load_trained_model(args, device)
    prompt_ids, resp_ids, att_mask_prompts, att_mask_resps, targets = load_dataset(tokenizer, device, args)
    eval_model(args, model, device, prompt_ids, resp_ids, att_mask_prompts, att_mask_resps, targets)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args) 