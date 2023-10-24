import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import random
import time
import datetime
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--prompts_path', type=str, help='Load path of question training data')
parser.add_argument('--responses_path', type=str, help='Load path of answer training data')
parser.add_argument('--prompt_ids_path', type=str, help='Load path of prompt ids')
parser.add_argument('--batch_size', type=int, default=12, help='Specify the training batch size')
parser.add_argument('--learning_rate', type=float, default=5e-5, help='Specify the initial learning rate')
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


def set_seed(args):
    seed_val = args.seed
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    return


def load_dataset(args, device):
    prompt_ids, response, train_data = [], [], []
    with open(args.prompt_ids_path) as f0, open(args.responses_path) as f1, open(args.prompts_path) as f2:
        prompt_ids, responses, prompts = f0.readlines(), f1.readlines(), f2.readlines()
        
        # choose to permute the dataset and concatenate it with the original dataset
        val = int(len(prompt_ids))
        prompt_ids = permute_data(prompt_ids, val, device)
        responses += responses # since we doubled the prompt size
    
        # need to prevent answers being assigned to their original question
        for (prompt_id, response) in zip(prompt_ids, responses):
            train_data.append(prompts[int(prompt_id)] + ' [SEP] ' + response)
            
    targets = torch.tensor([0] * val + [1] * val) # i.e. first half is on-topic, second half is off-topic
    print("Dataset Loaded")
    return train_data, targets


def permute_data(prompt_ids, val, device): 
    # Dynamic shuffling in order to generate off-topic samples, based on prompt probability dist.
    question_dist_path = '/home/alta/relevance/vr311/data_GKTS4_rnnlm/LINSKevl07/shuffled/'
    unique_prompts_distribution_path = "/scratches/dialfs/alta/ns832/data/train_seen/training/topics_dist.txt" 
    prompt_distribution = np.loadtxt(unique_prompts_distribution_path, dtype=np.int32)
    prompt_distribution = prompt_distribution / np.linalg.norm(prompt_distribution, 1)
    
    number_of_questions = len(prompt_distribution)
    new_prompt_ids = np.random.choice(number_of_questions, val, p=prompt_distribution)
    
    for i in range(val):
        while (new_prompt_ids[i] == prompt_ids[i]):
            new_prompt_ids[i] = np.random.choice(number_of_questions, 1, p=prompt_distribution)
    prompt_ids += list(new_prompt_ids)
    
    print("Data permuted")
    return prompt_ids


def encode_data(bert_base_uncased, train_data, device, targets):
    
    tokenizer = BertTokenizer.from_pretrained(bert_base_uncased, do_lower_case=True)
    input_ids, attention_mask = [], []
    
    for prompt_and_response in train_data:
        encoding = tokenizer(prompt_and_response, max_length = 512, padding="max_length", truncation=True)
        input_ids.append((encoding['input_ids'])) # number ids for the words in the question
        attention_mask.append((encoding['attention_mask'])) # mask any padding tokens

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    
    if device == 'cuda':
        input_ids.long().to(device)
        attention_mask.long().to(device)
        targets.long().to(device)
        
    print("Input_ids:", input_ids, "\n Attention_masks:", attention_mask)
    print(targets.size(), input_ids.size(), attention_mask.size())
    train_dataset = TensorDataset(input_ids, attention_mask, targets)
    return train_dataset


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
                    eps = args.adam_epsilon,
                    no_deprecation_warning=True
                    # weight_decay = 0.01
                    )
    return optimizer

def plot_loss(args, loss_values):
    fig = plt.figure(figsize=(8, 6))
    plt.plot(range(0, args.n_epochs), loss_values, marker='o', linestyle='-', color='b', label='Training Loss')
    plt.title('Training Loss vs Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    fig.savefig('/scratches/dialfs/alta/ns832/results/' + str(args.seed) + '_plot.jpg', bbox_inches='tight', dpi=150)
    plt.show()


def train_model(args, optimizer, model, device, train_dataset):
    
    # specify the sequence of indices/keys used in data loading,
    # want each epoch to have a different order to prevent over fitting
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
    
    total_steps = len(train_dataloader) * args.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 306,
                                                num_training_steps = total_steps)
    
    model.train()
    loss_values = []
    
    for epoch in range(args.n_epochs):
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
        
    avg_train_loss = total_loss / len(train_dataloader)
    loss_values.append(loss)
    
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(format_time(time.time() - t0)))
    
    plot_loss(args, loss_values)
    return


def save_model(args, model):
    file_path = str(args.save_path) + '/bert_seed_' + datetime.datetime.now() + str(args.seed) + '.pt'
    print(file_path)
    torch.save(model, file_path)
    return


def main(args):
    bert_base_uncased = "bert-base-uncased"
    set_seed(args)
    
    device = get_default_device()
    train_data, targets = load_dataset(args, device)
    train_dataset = encode_data(bert_base_uncased, train_data, device, targets)
    model = configure_model(bert_base_uncased, device)
    optimizer = configure_optimiser(model, args)
    
    train_model(args, optimizer, model, device, train_dataset)
    save_model(args, model)
    return


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)