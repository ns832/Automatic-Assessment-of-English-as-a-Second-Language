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
parser.add_argument('--topics_path', type=str, help='Load path of question training data')
parser.add_argument('--batch_size', type=int, default=12, help='Specify the training batch size')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Specify the initial learning rate')
parser.add_argument('--adam_epsilon', type=float, default=1e-6, help='Specify the AdamW loss epsilon')
parser.add_argument('--lr_decay', type=float, default=0.85, help='Specify the learning rate decay rate')
parser.add_argument('--dropout', type=float, default=0.1, help='Specify the dropout rate')
parser.add_argument('--n_epochs', type=int, default=1, help='Specify the number of epochs to train for')
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
    random.seed(seed_val), np.random.seed(seed_val)
    torch.manual_seed(seed_val), torch.cuda.manual_seed_all(seed_val)
    return


def load_dataset(args, device, bert_base_uncased):
    # First load the tokenizer and initialize empty arrays for your encoded inputs and masks
    tokenizer = BertTokenizer.from_pretrained(bert_base_uncased, do_lower_case=True)
    encoded_prompts, encoded_responses = [], []
    prompt_attention_masks, response_attention_masks = [], []
    
    with open(args.prompt_ids_path) as f0, open(args.responses_path) as f1, open(args.prompts_path) as f2,open(args.topics_path) as f3:
        prompt_ids, responses, prompts, topics = f0.readlines(), f1.readlines(), f2.readlines(), f3.readlines()
        prompts = [x.strip().lower() for x in prompts]
        prompt_ids = [x.strip().lower() for x in prompt_ids]
        responses = [x.strip().lower() for x in responses]
        topics = [x.strip().lower() for x in topics]
        print("Dataset Loaded")
        
        # choose to permute the dataset and concatenate it with the original dataset
        val = int(len(prompt_ids))
        prompts = permute_data(prompt_ids, val, prompts, topics)
        responses += responses # since we doubled the prompt size
        
        # max_prompt_length = max([len(sentence) for sentence in prompts]) max_resp_length = max([len(sentence) for sentence in responses])
        max_prompt_length, max_resp_length = 256, 256
        
        # Encode the prompts/responses and save the attention masks, padding applied to the end
        for (prompt, response) in zip(prompts, responses):
            encoded_prompt, prompt_attention_mask = encode_data(tokenizer, prompt, max_prompt_length)
            encoded_response, response_attention_mask = encode_data(tokenizer, response, max_resp_length)
            encoded_prompts.append(encoded_prompt), prompt_attention_masks.append(prompt_attention_mask)
            encoded_responses.append(encoded_response), response_attention_masks.append(response_attention_mask)
    
    # Targets i.e. first half is on-topic, second half is off-topic
    targets = torch.tensor([1] * val + [0] * val) 
    encoded_prompts, encoded_responses = torch.tensor(encoded_prompts), torch.tensor(encoded_responses)
    prompt_attention_masks, response_attention_masks = torch.tensor(prompt_attention_masks), torch.tensor(response_attention_masks)
    return encoded_prompts, encoded_responses, prompt_attention_masks, response_attention_masks, targets


def permute_data(prompt_ids, val, prompts, topics): 
    # Dynamic shuffling in order to generate off-topic samples, based on prompt probability dist.
    unique_prompts_distribution_path = "/scratches/dialfs/alta/relevance/ns832/data/train_seen/training/topics_dist.txt" 
    prompt_distribution = np.loadtxt(unique_prompts_distribution_path, dtype=np.int32)
    prompt_distribution = prompt_distribution / np.linalg.norm(prompt_distribution, 1)
    number_of_questions = len(prompt_distribution)
        
    # Cycle through the new batch, and reassign any responses that have been assigned their original prompts
    new_prompt_ids = np.random.choice(number_of_questions, val, p=prompt_distribution)
    for i in range(val):
        while (new_prompt_ids[i] == prompt_ids[i]):
            new_prompt_ids[i] = np.random.choice(number_of_questions, 1, p=prompt_distribution)
    prompt_ids += list(new_prompt_ids)
    
    # Assign the prompt, the topics.txt are one line out from the actual id due to how python works with indexing
    new_prompt_list = []
    for prompt_id in prompt_ids:
        new_prompt_list.append(topics[int(prompt_id)])
        
    print("Data permuted")
    return new_prompt_list


def encode_data(tokenizer, inputs, MAX_LEN):
    input_ids, attention_mask = [], []
    encoding = tokenizer(inputs, padding="max_length", max_length = MAX_LEN, add_special_tokens=True)
    if len(encoding)>256:
        encoding = encoding[:255]+[encoding[-1]]
    input_ids.append(encoding["input_ids"])
    attention_mask.append(encoding["attention_mask"])
    return input_ids, attention_mask


def create_dataset(encoded_prompts, encoded_responses, prompt_attention_masks, response_attention_masks, targets):
    encoded_prompts = encoded_prompts.squeeze(1)
    encoded_responses = encoded_responses.squeeze(1)
    prompt_attention_masks = prompt_attention_masks.squeeze(1)
    response_attention_masks = response_attention_masks.squeeze(1)
    
    prompt_and_response = torch.cat((encoded_prompts, encoded_responses),1)
    prompt_and_response_masks = torch.cat((prompt_attention_masks, response_attention_masks),1)
    prompt_and_response = prompt_and_response.squeeze(1)
    prompt_and_response_masks = prompt_and_response_masks.squeeze(1)
    print(prompt_and_response.size(), prompt_and_response_masks.size(), targets.size())
    
    train_data = TensorDataset(prompt_and_response, prompt_and_response_masks, targets)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
    return train_dataloader


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

def plot_loss(args, loss_values, avg_train_loss):
    
    fig = plt.figure(figsize=(8, 6))
    plt.plot(loss_values, marker='o', linestyle='-', color='b', label='Training Loss')
    plt.title('Training Loss vs Epoch')
    plt.xlabel('Epochs'), plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)  
    fig.savefig('/scratches/dialfs/alta/relevance/ns832/results' + '/bert_model_' + str(avg_train_loss) + '_plot.jpg', bbox_inches='tight', dpi=150)
    plt.show()


def train_model(args, optimizer, model, device, train_dataloader):
    
    # specify the sequence of indices/keys used in data loading,
    # want each epoch to have a different order to prevent over fitting
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
                loss_values.append(torch.tensor(loss, device = 'cpu'))
                
            model.zero_grad()
            prompt_response = (batch[0].to(device)).squeeze(1)
            prompt_response_mask = (batch[1].to(device)).squeeze(1)
            targets_batch = batch[2].to(device) 
            # print(prompt_response.size())
            # print(prompt_response_mask.size())
            # print(targets_batch.size())
            
            # First compute the outputs given the current weights, and the current and total loss
            outputs = model(input_ids=prompt_response, attention_mask=prompt_response_mask, labels=targets_batch)
            loss = outputs.loss
            total_loss += loss.item()
            print("loss.item is", loss.item())
            
            # Then perform the backwards pass through the nn, updating the weights based on gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
    avg_train_loss = total_loss / len(train_dataloader)
    
    
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(format_time(time.time() - t0)))
    
    plot_loss(args, loss_values, avg_train_loss)
    return avg_train_loss


def save_model(args, model, avg_train_loss):
    file_path = str(args.save_path) + '/bert_model_' + str(avg_train_loss) + '.pt'
    print(file_path)
    torch.save(model, file_path)
    return


def main(args):
    # bert_base_uncased = "bert-base-uncased"
    bert_base_uncased = "prajjwal1/bert-small"
    set_seed(args)
    
    device = get_default_device()
    encoded_prompts, encoded_responses, prompt_attention_masks, response_attention_mask, targets = load_dataset(args, device, bert_base_uncased)
    print("Dataset Encoded")
    train_dataloader = create_dataset(encoded_prompts, encoded_responses, prompt_attention_masks, response_attention_mask, targets)
    
    model = configure_model(bert_base_uncased, device)
    optimizer = configure_optimiser(model, args)
    
    avg_train_loss = train_model(args, optimizer, model, device, train_dataloader)
    save_model(args, model, avg_train_loss)
    return


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)