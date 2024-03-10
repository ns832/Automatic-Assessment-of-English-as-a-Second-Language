import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import datetime
from sklearn.metrics import precision_recall_curve
from transformers import BertTokenizer
import matplotlib.pyplot as plt
from Section_D import metrics


parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=100, help='Specify the test batch size')
parser.add_argument('--prompts_path', type=str, help='Load path to test prompts as text')
parser.add_argument('--responses_path', type=str, help='Load path to test responses as text')
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
    encoded_prompts, encoded_responses, att_mask_prompts, att_mask_responses = [], [], [], []
    targets = np.loadtxt(args.labels_path, dtype=int)
    targets = torch.tensor(targets)
    
    with open(args.responses_path) as f0, open(args.prompts_path) as f1:
        responses, prompts = f0.readlines(), f1.readlines()
        prompts = [x.strip().lower() for x in prompts]
        responses = [x.strip().lower() for x in responses]
        print("Dataset Loaded")
        
        max_prompt_length = 256
        max_resp_length = 256
        
        for (prompt, response) in zip(prompts, responses):
            encoded_prompt, att_mask_prompt = encode_data(tokenizer, prompt, max_prompt_length)
            encoded_response, att_mask_resp = encode_data(tokenizer, response, max_resp_length)
            encoded_prompts.append(encoded_prompt), att_mask_prompts.append(att_mask_prompt)
            encoded_responses.append(encoded_response), att_mask_responses.append(att_mask_resp)
        print("Data Encoded")
    
    encoded_prompts, encoded_responses = torch.tensor(encoded_prompts), torch.tensor(encoded_responses)
    att_mask_prompts, att_mask_responses = torch.tensor(att_mask_prompts), torch.tensor(att_mask_responses)
    return encoded_prompts, encoded_responses, att_mask_prompts, att_mask_responses, targets
    
def encode_data(tokenizer, inputs, MAX_LEN):
    input_ids, attention_mask = [], []
    encoding = tokenizer(inputs, padding="max_length", max_length = MAX_LEN, truncation=True)
    input_ids.append(encoding["input_ids"])
    attention_mask.append(encoding["attention_mask"])
    return input_ids, attention_mask

def eval_model(args, model, device, prompt_ids, resp_ids, att_mask_prompts, att_mask_responses, targets):
    prompt_ids = prompt_ids.squeeze(1)
    att_mask_prompts = att_mask_prompts.squeeze(1)
    resp_ids = resp_ids.squeeze(1)
    att_mask_responses = att_mask_responses.squeeze(1)
    eval_dataset = TensorDataset(prompt_ids, att_mask_prompts, resp_ids, att_mask_responses, targets)
    eval_dataloader = DataLoader(eval_dataset, batch_size = args.batch_size, shuffle = False)

    model.eval()    
    y_pred_all = []
    print("Started evaluation")
    
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
    return targets, y_pred_all


    
def main(args):
    device = get_default_device()
    model, tokenizer = load_trained_model(args, device)
    prompt_ids, resp_ids, att_mask_prompts, att_mask_responses, targets = load_dataset(tokenizer, args)
    targets, y_pred_all = eval_model(args, model, device, prompt_ids, resp_ids, att_mask_prompts, att_mask_responses, targets)
    metrics.calculate_metrics(targets, y_pred_all)
    metrics.calculate_normalised_metrics(targets, y_pred_all)
    return y_pred_all

if __name__ == '__main__':
    args = parser.parse_args()
    main(args) 