# import argparse
# import torch
# from torch.utils.data import DataLoader, RandomSampler, Dataset
# import numpy as np
# import preprocess_data, load_models
# from transformers import get_linear_schedule_with_warmup
# import torch.nn as nn
# import torch.optim as optim
# import metrics
# from sentence_transformers import SentenceTransformer

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

# parser.add_argument('--classification_model_path', type=str)

# parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
# parser.add_argument('--batch_size', type=int, default=12, help='Specify the test batch size')
# parser.add_argument('--learning_rate', type=float, default=1e-3, help='Specify the initial learning rate')
# parser.add_argument('--adam_epsilon', type=float, default=1e-6, help='Specify the AdamW loss epsilon')
# parser.add_argument('--lr_decay', type=float, default=0.85, help='Specify the learning rate decay rate')
# parser.add_argument('--dropout', type=float, default=0.1, help='Specify the dropout rate')
# parser.add_argument('--n_epochs', type=int, default=1, help='Specify the number of epochs to train for')


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
#             'prompt': item.prompt,
#             'response': item.response,
#             'image': item.image,
#             'target': item.target
#         }
    
    
# class Combined_Model(nn.Module):
#     def __init__(self, text_embedder, image_embedder, linear_head):
#         super(Combined_Model, self).__init__()
#         self.text_embedder = text_embedder
#         self.image_embedder = image_embedder
#         self.linear_head = linear_head
        
#     def forward(self, prompt, response, image):
#         prompt_embedding = self.text_embedder.encode(prompt)
#         response_embedding = self.text_embedder.encode(response)
        
#         text_embedding = np.concatenate((prompt_embedding, response_embedding), axis=1)
#         text_embedding = torch.Tensor(text_embedding).to(device)
        
#         image_embedding = self.image_embedder(image.to(device))
#         image_CLS_token = image_embedding.last_hidden_state[:,0,:]
#         combined_embedding = torch.cat((text_embedding, image_CLS_token), dim=1)        
#         logits = self.linear_head(combined_embedding)
#         return logits


# class LinearHead(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(LinearHead, self).__init__()
#         self.dropout = nn.Dropout(p=0.1)
#         self.linear = nn.Linear(input_size, output_size)

#     def forward(self, x):
#         x = self.dropout(x)
#         x = self.linear(x)
#         return x
    
    
    
    
# def encode_dataset(text_data, image_data):
#     """
#         Adds the corresponding images to a list
#     """
    
#     # Encode the prompts/responses and save the attention masks, padding applied to the end
#     image_list = []
#     index_to_remove = []
    
#     # Find the corresponding image so that later they can be concatenated together
#     for data in text_data:
#         image = preprocess_data.find_corresponding_image_id(data.prompt, image_data)[1]
#         if image in image_data: image_list.append(image)
#         else: index_to_remove.append(text_data.index(data))
            
#     for index in list(reversed(index_to_remove)):
#         text_data.pop(index)
#     return text_data, image_list


# def eval_classifier(args, eval_dataloader, classifier):
#     y_pred_list, targets = [], []  
#     for epoch in range(args.n_epochs):
#         print("Epoch: ", epoch, " of ", args.n_epochs)
        
#         for i,batch in enumerate(eval_dataloader):
#             print(i, " of ", len(eval_dataloader))
#             classifier.zero_grad()
#             prompts_batch = batch['prompt']
#             responses_batch = batch['response']
#             image_batch = batch['image'].squeeze()
#             targets_batch = batch['target'].float().to(device)
            
#             logits = classifier(prompts_batch, responses_batch, image_batch).squeeze(dim=1)
            
#             y_pred_list += logits.cpu().detach().numpy().tolist()
#             targets += targets_batch.cpu().detach().numpy().tolist()

#     targets = np.array(targets)
#     y_pred_list = np.array(y_pred_list)
#     metrics.calculate_metrics(targets, y_pred_list)


# def load_classification_head(input_size, output_size, text_embedder, image_embedder):
#     linear_head = LinearHead(input_size=input_size, output_size=output_size)
#     classifier = Combined_Model(text_embedder, image_embedder, linear_head)
#     classifier.load_state_dict(torch.load(args.classification_model_path))
#     classifier.eval()
#     classifier.to(device)
#     return classifier


# def main(num_labels=1):
#     preprocess_data.set_seed(args)
    
#     # Load Models and Optimisers
#     text_embedder = SentenceTransformer('sentence-transformers/sentence-t5-base').to(device)
#     image_embedder, image_processor = load_models.load_vision_transformer_model()
#     text_input_size = getattr(text_embedder[2], 'out_features') * 2 # Since we are handling the prompts separately to the responses
#     image_input_size = getattr(image_embedder.pooler.dense, 'out_features') 
#     # image_input_size = 0

#     # Instantiate Model
#     classifier = load_classification_head(input_size=(text_input_size + image_input_size), output_size=num_labels, text_embedder=text_embedder, image_embedder=image_embedder)

#     # Load data from files
#     text_data, image_data, topics = preprocess_data.load_dataset(args)
#     text_data = preprocess_data.remove_incomplete_data(text_data, image_data)
          
#     if args.labels_path == None:
#         # Shuffling real data to create synthetic
#         text_data = [x for x in text_data if x.target == 1]
#         text_data = preprocess_data.permute_data(text_data, topics, args)
#         np.random.shuffle(text_data)

#     # Preprocess visual data
#     image_data = preprocess_data.load_images(image_data, args)  
#     text_data, image_list = encode_dataset( text_data, image_data)
#     image_list = preprocess_data.apply_image_processor(image_list, image_processor)
#     data_eval = preprocess_data.remove_mismatching_prompts(image_list, text_data)
    
#     # Create dataloader
#     eval_data = CustomDataset(data_eval)
#     eval_dataloader = DataLoader(eval_data, batch_size=args.batch_size)

#     eval_classifier(args, eval_dataloader, classifier)
    
    

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
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import datetime
from sklearn.metrics import precision_recall_curve
from transformers import BertTokenizer, BertModel, BertConfig
import matplotlib.pyplot as plt
import metrics


parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=100, help='Specify the test batch size')
parser.add_argument('--prompts_path', type=str, help='Load path to test prompts as text')
parser.add_argument('--responses_path', type=str, help='Load path to test responses as text')
parser.add_argument('--labels_path', type=str, help='Load path to labels')
parser.add_argument('--model_path', default="/scratches/dialfs/alta/relevance/ns832/results/classification_model_0.08220307986157885.pt", type=str, help='Load path to trained model')
parser.add_argument('--predictions_save_path', type=str, help="Where to save predicted values")


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
    config = BertConfig()
    model = Model(config)
    model.load_state_dict(torch.load(args.model_path))
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
            outputs = model(input_ids=pr_resp.squeeze(1), attention_mask=pr_resp_msk.squeeze(1), labels=target)
            # outputs = model(pr_resp.squeeze(1), token_type_ids=None, attention_mask=pr_resp_msk.squeeze(1), labels=target)
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