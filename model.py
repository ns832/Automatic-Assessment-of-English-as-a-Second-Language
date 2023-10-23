import argparse
# import os
# import sys
# import json

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import random
import time
import datetime
# from datasets import Dataset, Features, ClassLabel

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--question_path', type=str, help='Load path of question training data')
parser.add_argument('--answer_path', type=str, help='Load path of answer training data')
parser.add_argument('--batch_size', type=int, default=24, help='Specify the training batch size')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='Specify the initial learning rate')
parser.add_argument('--adam_epsilon', type=float, default=1e-6, help='Specify the AdamW loss epsilon')
parser.add_argument('--lr_decay', type=float, default=0.85, help='Specify the learning rate decay rate')
parser.add_argument('--dropout', type=float, default=0.1, help='Specify the dropout rate')
parser.add_argument('--n_epochs', type=int, default=10, help='Specify the number of epochs to train for')
parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
parser.add_argument('--save_path', type=str, help='Load path to which trained model will be saved')

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def get_default_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    return device

def load_dataset(args):
    
    questions, answers, train_data = [], [], []
    with open(args.question_path) as f0, open(args.answer_path) as f1:
        questions, answers = f0.readlines(), f1.readlines()
        
        # choose to permute half of the dataset
        val = int(len(questions)/2)
        
        permuted_questions, permuted_answers = questions[:val], answers[:val]
        non_permuted_questions, non_permuted_answers = questions[val + 1:], answers[val + 11:]
    
        for (question, answer) in zip(non_permuted_questions, non_permuted_answers):
            train_data.append(question + ' [SEP] ' + answer)
            
        for (question, answer) in zip(permuted_questions, random.sample(permuted_answers, len(permuted_answers))):
            train_data.append(question + ' [SEP] ' + answer)
            
    targets = torch.tensor([0] * val + [1] * val) # i.e. first half is on-topic, second half is off-topic
    print("Dataset Loaded")
    return train_data, targets


def encode_data(bert_base_uncased, train_data, device, targets):
    
    tokenizer = BertTokenizer.from_pretrained(bert_base_uncased, do_lower_case=True)
    
    for question in train_data:
        encoding = tokenizer(question, return_tensors='pt', max_length = 512, padding="max_length", truncation=True)
        input_ids = (encoding['input_ids']).clone().detach() # number ids for the words in the question
        attention_mask = (encoding['attention_mask']).clone().detach() # 1 for question, 0 for answer
        
    if device == 'cuda':
        input_ids.long().to(device)
        attention_mask.long().to(device)
        targets.long().to(device)
        
    print("Input_ids:", input_ids, "\n Attention_masks:", attention_mask)
    train_dataset = TensorDataset(input_ids, attention_mask, targets)
    return train_dataset


def set_seed(args):
    seed_val = args.seed
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    return

def configure_model(bert_base_uncased, device):
    model = BertForSequenceClassification.from_pretrained(
        bert_base_uncased, # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # Only two classes, on-topic and off-topic  
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
        )
    if device == 'cuda':
        model.to(device)
    return model

def configure_optimiser(model, args):
    optimizer = AdamW(model.base_model.parameters(),
                    lr = args.learning_rate,
                    eps = args.adam_epsilon
                    # weight_decay = 0.01
                    )
    return optimizer

def train_model(args, optimizer, model, device, train_dataset):
    
    # specify the sequence of indices/keys used in data loading,
    # want each epoch to have a different order to prevent over fitting
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, train_sampler, batch_size=args.batch_size, shuffle=True)
    
    total_steps = len(train_dataloader) * args.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 306,
                                                num_training_steps = total_steps)
    
    model.train()
    # labels = torch.tensor([0,1]).unsqueeze(0) # 0 being on-topic, 1 being off-topic
    
    for epoch in args.n_epoch:
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, args.n_epochs))
        print('Training...')
        t0 = time.time()
        total_loss = 0
        model.train()
        model.zero_grad() 
        
        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            model.zero_grad()
            
            b_input_ids = batch[0].to(device)
            b_att_msks = batch[1].to(device)
            b_targets = batch[2].to(device)
            
            # First compute the outputs given the current weights, and the current and total loss
            outputs = model(input_ids=b_input_ids, attention_mask=b_att_msks, labels=b_targets)
            loss = outputs.loss
            total_loss += loss.item()
            print("loss.item is", loss.item())
            
            # Then perform the backwards pass through the nn, updating the weights based on gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
    avg_train_loss = total_loss / len(b_input_ids)
    
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(format_time(time.time() - t0)))
    return

def save_model(args, model):
    file_path = args.save_path +' bert_seed ' + str(args.seed) + '.pt'
    torch.save(model, file_path)
    return

def main(args):
    bert_base_uncased = "bert-base-uncased"
    set_seed(args)
    
    device = get_default_device()
    train_data, targets = load_dataset(args)
    train_dataset = encode_data(bert_base_uncased, train_data, device, targets)
    model = configure_model(bert_base_uncased, device)
    optimizer = configure_optimiser(model, args)
    
    train_model(args, optimizer, model, device, train_dataset)
    save_model(args, model)
    return

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)