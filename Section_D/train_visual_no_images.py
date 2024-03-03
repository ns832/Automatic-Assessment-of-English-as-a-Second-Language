# import argparse
# import torch
# from torch.utils.data import DataLoader, RandomSampler, Dataset
# import numpy as np
# import preprocess_data, load_models
# from transformers import get_linear_schedule_with_warmup, BertModel, BertTokenizer, BertConfig, AdamW
# import torch.nn as nn
# import torch.optim as optim
# import metrics
# # from sentence_transformers import SentenceTransformer

# parser = argparse.ArgumentParser(description='Get all command line arguments.')
# parser.add_argument('--folder_path', type=str, default=None, help='Load path of the folder containing prompts, responses etc.')
# parser.add_argument('--prompts_path', type=str, default=None, help='Load path of question training data')
# parser.add_argument('--responses_path', type=str, default=None, help='Load path of answer training data')
# parser.add_argument('--prompt_ids_path', type=str, default=None, help='Load path of prompt ids')
# parser.add_argument('--topics_path', type=str, default=None, help='Load path of topics')
# parser.add_argument('--topic_dist_path', type=str, default=None, help='Load path of prompt distribution')
# parser.add_argument('--labels_path', type=str, default=None ,help='Load path to labels')

# # An optional image folder_path can be supplied if multiple image files are in the same directory - any file_path not given is then assumed to be in this folder
# parser.add_argument('--folder_path_images', type=str, default=None, help='Optional path for folder containing image data.')
# parser.add_argument('--images_path', type=str, help='Load path of image training data')
# parser.add_argument('--image_ids_path', type=str, help='Load path of image ids')
# parser.add_argument('--image_prompts_path', type=str, help='Load path of prompts corresponding to image ids')

# parser.add_argument('--model_path', type=str, default="/scratches/dialfs/alta/relevance/ns832/results/train_model_unseen/bert_model_0.14654680755471014.pt")

# parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
# parser.add_argument('--batch_size', type=int, default=12, help='Specify the test batch size')
# parser.add_argument('--learning_rate', type=float, default=1e-4, help='Specify the initial learning rate')
# parser.add_argument('--adam_epsilon', type=float, default=1e-6, help='Specify the AdamW loss epsilon')
# parser.add_argument('--n_epochs', type=int, default=4, help='Specify the number of epochs to train for')


# # Global Variables
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)
# args = parser.parse_args()

# class CustomDataset(Dataset):
#     def __init__(self, data):
#         self.data = data
        
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         item = self.data[index]
#         return {
#             'text': torch.Tensor(item.text[0]).to(torch.int64),
#             'mask': torch.Tensor(item.mask[0]).to(torch.int64),
#             'target': item.target
#         }
 
# class Model(nn.Module):
#     def __init__(self, config):
#         super(Model, self).__init__()
#         self.num_labels = 2 #config.num_labels

#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
#         self.loss_fct = nn.CrossEntropyLoss()
        
#     def forward(self, input_ids, attention_mask, labels):
        
#         outputs = self.bert(input_ids, attention_mask)
#         pooled_outputs = outputs[1]
#         pooled_outputs = self.dropout(pooled_outputs)

#         logits = self.classifier(pooled_outputs)
        
#         outputs = (logits,) + outputs[2:]
#         loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
#         outputs = (loss,) + outputs
#         return outputs   
        
        
# def encode_dataset(tokenizer, text_data, max_prompt_length = 256):
#     """
#         Encodes the entire text data by calling the function encode_data().
#         Adds the corresponding images to a list
#     """
    
#     for data in text_data:
#         text, mask = preprocess_data.encode_data(tokenizer, data, max_prompt_length)
#         data.add_encodings(text, mask)
        
#     return text_data


# def train_model(args, model, optimizer, train_dataloader, val_dataloader):
    
#     scheduler = get_linear_schedule_with_warmup(optimizer,
#                                                 num_warmup_steps = 0,
#                                                 num_training_steps = len(train_dataloader) * args.n_epochs)
    
#     model.train()
#     old_f_score = 0
        
#     for epoch in range(args.n_epochs):
#         print('\n ======== Epoch {:} / {:} ========'.format(epoch + 1, args.n_epochs))
#         total_loss = 0
        
#         for step , batch in enumerate(train_dataloader):          
#             model.zero_grad()
#             input_ids = batch['text'].to(device) 
#             attention_mask = batch['mask'].to(device) 
#             targets_batch = batch['target'].to(device).to(torch.int64)

#             output = model(input_ids, attention_mask, targets_batch)
#             loss = output[0]
#             total_loss += loss
#             optimizer.zero_grad()
#             if step%100 == 0: print(step, " : ", loss.item())
#             loss.backward()
#             optimizer.step()
#             scheduler.step()
#         avg_train_loss = total_loss / len(train_dataloader)
#         print("Avg loss: ", avg_train_loss.item())
#         f_score = eval_model(model, val_dataloader)
#         # if f_score < old_f_score: break
#         # else: old_f_score = f_score  
        
#     return model


# def eval_model(model, val_dataloader):
    
#     y_pred_list, targets = [], []  
#     total_loss = 0
    
#     for batch in val_dataloader:
#         input_ids = batch['text'].to(device) 
#         attention_mask = batch['mask'].to(device) 
#         targets_batch = batch['target'].to(device).to(int)

#         with torch.no_grad():
#             output = model(input_ids, attention_mask, targets_batch)
#         loss = output[0]
#         logits = output[1]
#         total_loss += loss.item()
        
#         y_pred_list += np.squeeze(logits.cpu().detach().numpy())[:,1].tolist()
#         targets += targets_batch.cpu().detach().numpy().tolist()
    
#     print(total_loss)
#     targets = np.array(targets)
#     y_pred_list = np.array(y_pred_list)
#     f_score = metrics.calculate_metrics(targets, y_pred_list)
    
#     return f_score


# def configure_model(args, bert_base_uncased):
#     config = BertConfig()    
#     model = Model(config)
#     tokenizer = BertTokenizer.from_pretrained(bert_base_uncased, do_lower_case=True)

#     # Freeze all parameters except for linear layer
#     # for param in classifier.image_embedder.parameters():
#     #     param.requires_grad = False
    
#     if device == 'cuda':
#         model.to(device)
        
#     optimizer = AdamW(model.parameters(),
#                     lr = args.learning_rate,
#                     eps = args.adam_epsilon,
#                     no_deprecation_warning=True
#                     # weight_decay = 0.01
#                     )
#     return model, tokenizer, optimizer


# def main():
#     preprocess_data.set_seed(args)
#     model, tokenizer, optimizer = configure_model(args, bert_base_uncased="prajjwal1/bert-small")
    
#     # Load data from files
#     text_data, image_data, topics = preprocess_data.load_dataset(args, images = False)
#     text_data = encode_dataset(tokenizer, text_data)

    
#     # Preprocess visual data
#     # image_data = preprocess_data.load_images(image_data, args)  
#     # text_data, image_list = encode_dataset( text_data, image_data)
#     # image_list = preprocess_data.apply_image_processor(image_list, image_processor)
#     # combined_data = preprocess_data.remove_mismatching_prompts(image_list, text_data)
    
    
#     combined_train_data = np.array([x for x in text_data])
#     train_data = CustomDataset(combined_train_data)
#     train_sampler = RandomSampler(train_data)
#     train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
#     val_dataloader = train_dataloader
#     model = train_model(args, model, optimizer, train_dataloader, val_dataloader)
#     metrics.save_model(model, "_")
    
    

# if __name__ == '__main__':
#     if args.folder_path:
#         if args.prompts_path == None: args.prompts_path = str(args.folder_path) + "prompts.txt"
#         if args.responses_path == None: args.responses_path = str(args.folder_path) + "responses.txt"
#         if args.topics_path == None: args.topics_path = str(args.folder_path) + "topics.txt"
#         if args.topic_dist_path == None: args.topic_dist_path = str(args.folder_path) + "topics_dist.txt"
#         if args.labels_path == None: args.labels_path = str(args.folder_path) + "targets.txt"
#     if args.folder_path_images:
#         if args.images_path == None: args.images_path = str(args.folder_path_images)
#         if args.image_ids_path == None: args.image_ids_path = str(args.folder_path_images) + "image_ids.txt"
#         if args.image_prompts_path == None: args.image_prompts_path = str(args.folder_path_images) + "image_questions.txt"
#     main() 


import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import numpy as np
import random
import time
import datetime
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, BertModel
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
import torch.nn as nn
import preprocess_data
import metrics

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--prompts_path', type=str, help='Load path of question training data')
parser.add_argument('--responses_path', type=str, help='Load path of answer training data')
parser.add_argument('--prompt_ids_path', type=str, help='Load path of prompt ids')
parser.add_argument('--topics_path', type=str, help='Load path of question training data')
parser.add_argument('--topic_dist_path', type=str, help='Load path of question training data')
parser.add_argument('--batch_size', type=int, default=12, help='Specify the training batch size')
parser.add_argument('--learning_rate', type=float, default=1e-6, help='Specify the initial learning rate')
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


def load_dataset(args, bert_base_uncased):
    # First load the tokenizer and initialize empty arrays for your encoded inputs and masks
    tokenizer = BertTokenizer.from_pretrained(bert_base_uncased, do_lower_case=True)
    encoded_prompts, encoded_responses = [], []
    prompt_attention_masks, response_attention_masks = [], []
    val_encoded_prompts, val_encoded_responses = [], []
    val_prompt_attention_masks, val_response_attention_masks = [], []
    
    with open(args.prompt_ids_path) as f0, open(args.responses_path) as f1, open(args.topics_path) as f2:
        prompt_ids_file, responses_file, topics = f0.readlines(), f1.readlines(), f2.readlines()
        prompt_ids = [x.strip().lower() for x in prompt_ids_file[:-100]]
        responses = [x.strip().lower() for x in responses_file[:-100]]
        val_prompt_ids = [x.strip().lower() for x in prompt_ids_file[-100:]]
        val_responses = [x.strip().lower() for x in responses_file[-100:]]
        topics = [x.strip().lower() for x in topics]
        
        print("Dataset Loaded")
        
        # choose to permute the dataset and concatenate it with the original dataset
        val = int(len(prompt_ids))
        val_2 = int(len(val_prompt_ids))
        prompts = permute_data(prompt_ids, val, topics, args)
        val_prompts = permute_data(val_prompt_ids, val_2, topics, args)
        responses += responses # since we doubled the prompt size
        val_responses += val_responses # since we doubled the prompt size
        
        # max_prompt_length = max([len(sentence) for sentence in prompts]) max_resp_length = max([len(sentence) for sentence in responses])
        max_prompt_length, max_resp_length = 256, 256
        
        # Encode the prompts/responses and save the attention masks, padding applied to the end
        for (prompt, response) in zip(prompts, responses):
            encoded_prompt, prompt_attention_mask = encode_data(tokenizer, prompt, max_prompt_length)
            encoded_response, response_attention_mask = encode_data(tokenizer, response, max_resp_length)
            encoded_prompts.append(encoded_prompt), prompt_attention_masks.append(prompt_attention_mask)
            encoded_responses.append(encoded_response), response_attention_masks.append(response_attention_mask)
        for (prompt, response) in zip(val_prompts, val_responses):
            encoded_prompt, prompt_attention_mask = encode_data(tokenizer, prompt, max_prompt_length)
            encoded_response, response_attention_mask = encode_data(tokenizer, response, max_resp_length)
            val_encoded_prompts.append(encoded_prompt), val_prompt_attention_masks.append(prompt_attention_mask)
            val_encoded_responses.append(encoded_response), val_response_attention_masks.append(response_attention_mask)

    # Targets i.e. first half is on-topic, second half is off-topic
    targets = torch.tensor([1] * val + [0] * val) 
    val_targets = torch.tensor([1] * val_2 + [0] * val_2) 
    encoded_prompts, encoded_responses = torch.tensor(encoded_prompts), torch.tensor(encoded_responses)
    prompt_attention_masks, response_attention_masks = torch.tensor(prompt_attention_masks), torch.tensor(response_attention_masks)
    val_encoded_prompts, val_encoded_responses = torch.tensor(val_encoded_prompts), torch.tensor(val_encoded_responses)
    val_prompt_attention_masks, val_response_attention_masks = torch.tensor(val_prompt_attention_masks), torch.tensor(val_response_attention_masks)
    
    train_dataloader = create_dataset(encoded_prompts, encoded_responses, prompt_attention_masks, response_attention_masks, targets)
    validation_dataloader = create_dataset(val_encoded_prompts, val_encoded_responses, val_prompt_attention_masks, val_response_attention_masks, val_targets)
    
    return train_dataloader, validation_dataloader


def permute_data(prompt_ids, val, topics, args): 
    # Dynamic shuffling in order to generate off-topic samples, based on prompt probability dist.
    unique_prompts_distribution_path = args.topic_dist_path 
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
        
    print("Data Permuted")
    return new_prompt_list


def encode_data(tokenizer, inputs, MAX_LEN):
    input_ids, attention_mask = [], []
    encoding = tokenizer(inputs, padding="max_length", max_length = MAX_LEN, add_special_tokens=True)
    # input_ids.append(encoding["input_ids"])
    if len(encoding["input_ids"]) > 256:
        input_ids.append(encoding["input_ids"][:256])
        attention_mask.append(encoding["attention_mask"][:256])
    else:
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

    train_data = TensorDataset(prompt_and_response, prompt_and_response_masks, targets)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
    return train_dataloader


def configure_model( device):
    class Model(nn.Module):
        def __init__(self, config):
            super(Model, self).__init__()
            self.num_labels = 2 #config.num_labels

            self.bert = BertModel(config)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
            
            self.loss_fct = nn.CrossEntropyLoss()
            
        def forward(self, input_ids, attention_mask, labels):
            outputs = self.bert(input_ids, attention_mask)
            pooled_outputs = outputs[1]
            pooled_outputs = self.dropout(pooled_outputs)

            logits = self.classifier(pooled_outputs)
            
            outputs = (logits,) + outputs[2:]
            
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
            outputs = (loss,) + outputs
            return outputs   
        
    config = BertConfig()
    model = Model(config)
    
    if device == 'cuda':
        model.to(device)
        
    return model


def configure_optimiser(model, args):
    optimizer = AdamW(model.parameters(),
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


def train_model(args, optimizer, model, device, train_dataloader, validation_dataloader):
    total_steps = len(train_dataloader) * args.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 306,
                                                num_training_steps = total_steps)
    
    model.train()
    old_f_score = 0
    
    for epoch in range(args.n_epochs):
        print('\n ======== Epoch {:} / {:} ========'.format(epoch + 1, args.n_epochs))
        total_loss = 0
        
        for step, batch in enumerate(train_dataloader):          
            model.zero_grad()
            prompt_response = (batch[0].to(device)).squeeze(1)
            prompt_response_mask = (batch[1].to(device)).squeeze(1)
            targets_batch = batch[2].to(device) 

            outputs = model(input_ids=prompt_response, attention_mask=prompt_response_mask, labels=targets_batch)
            loss = outputs[0]
            total_loss += loss.item()
            if step%100 == 0: print(step, ":", loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        avg_loss = total_loss / len(train_dataloader)
        print("avg_train_loss", avg_loss)
        f_score = eval_model(model, device, validation_dataloader)
        if f_score < old_f_score: break
        else: old_f_score = f_score
    return model, avg_loss


def send_to_device(device, x):
    x = x.clone().detach()
    x = x.long()
    x = x.to(device)
    return x


def eval_model(model, device, validation_dataloader):
    model.eval()    
    y_pred_all, targets = [], []
    total_loss = 0
    
    for batch in validation_dataloader:
        prompt_response = (batch[0].to(device)).squeeze(1)
        prompt_response_mask = (batch[1].to(device)).squeeze(1)
        targets_batch = batch[2].to(device) 

        with torch.no_grad():
            outputs = model(input_ids=prompt_response, attention_mask=prompt_response_mask, labels=targets_batch)
        loss = outputs[0]
        total_loss += loss.item()
        logits = outputs[1]
        logits = logits.cpu()
        logits = logits.detach().numpy()
        logits = np.squeeze(logits[:, 1])
        logits = logits.tolist()
        y_pred_all += logits  
        targets_batch = targets_batch.cpu().detach().numpy()
        targets_batch = np.squeeze(targets_batch).tolist()
        targets += targets_batch 
    print(total_loss)
    y_pred_all = np.array(y_pred_all)
    targets = np.array(targets)
    
    f_score = metrics.calculate_metrics(targets, y_pred_all)
    return f_score


def save_model(args, model, avg_train_loss):
    file_path = str(args.save_path) + '/bert_vit_' + str(avg_train_loss) + '.pt'
    print(file_path)
    torch.save(model, file_path)
    return
    

def main(args):
    # bert_base_uncased = "bert-base-uncased"
    bert_base_uncased = "prajjwal1/bert-small"
    set_seed(args)
    
    device = get_default_device()
    train_dataloader, validation_dataloader = load_dataset(args, bert_base_uncased)

    # tokenizer = BertTokenizer.from_pretrained(bert_base_uncased, do_lower_case=True)
    # text_data, _, _ = preprocess_data.load_dataset(args, images = False)
    # text_data = encode_dataset(tokenizer, text_data)
    
    
    model = configure_model(device)
    optimizer = configure_optimiser(model, args)
    
    model, avg_train_loss = train_model(args, optimizer, model, device, train_dataloader, validation_dataloader)
    # save_model(args, model, avg_train_loss)
    metrics.save_model(model, avg_train_loss)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)