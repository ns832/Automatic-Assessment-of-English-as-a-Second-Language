import argparse
import torch
from torch.utils.data import DataLoader, RandomSampler, Dataset
import numpy as np
import preprocess_data, load_models
from transformers import get_linear_schedule_with_warmup, BertModel, BertTokenizer, BertConfig, AdamW
import torch.nn as nn
import torch.optim as optim
import metrics
# from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--folder_path', type=str, default=None, help='Load path of the folder containing prompts, responses etc.')
parser.add_argument('--prompts_path', type=str, default=None, help='Load path of question training data')
parser.add_argument('--responses_path', type=str, default=None, help='Load path of answer training data')
parser.add_argument('--prompt_ids_path', type=str, default=None, help='Load path of prompt ids')
parser.add_argument('--topics_path', type=str, default=None, help='Load path of topics')
parser.add_argument('--topic_dist_path', type=str, default=None, help='Load path of prompt distribution')
parser.add_argument('--labels_path', type=str, default=None ,help='Load path to labels')

# An optional image folder_path can be supplied if multiple image files are in the same directory - any file_path not given is then assumed to be in this folder
parser.add_argument('--folder_path_images', type=str, default=None, help='Optional path for folder containing image data.')
parser.add_argument('--images_path', type=str, help='Load path of image training data')
parser.add_argument('--image_ids_path', type=str, help='Load path of image ids')
parser.add_argument('--image_prompts_path', type=str, help='Load path of prompts corresponding to image ids')

# parser.add_argument('--model_path', type=str, default="/scratches/dialfs/alta/relevance/ns832/results/train_model_unseen/bert_model_0.14654680755471014.pt")

parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
parser.add_argument('--batch_size', type=int, default=12, help='Specify the test batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Specify the initial learning rate')
parser.add_argument('--adam_epsilon', type=float, default=1e-6, help='Specify the AdamW loss epsilon')
parser.add_argument('--n_epochs', type=int, default=4, help='Specify the number of epochs to train for')


# Global Variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
args = parser.parse_args()

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        return {
            'text': torch.Tensor(item.text[0]).to(torch.int64),
            'mask': torch.Tensor(item.mask[0]).to(torch.int64),
            'target': item.target
        }
 
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
        
        
def encode_dataset(tokenizer, text_data, max_prompt_length = 256):
    """
        Encodes the entire text data by calling the function encode_data().
        Adds the corresponding images to a list
    """
    
    for data in text_data:
        text, mask = preprocess_data.encode_data(tokenizer, data, max_prompt_length)
        data.add_encodings(text, mask)
        
    return text_data


def train_model(args, model, optimizer, train_dataloader, val_dataloader):
    
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0,
                                                num_training_steps = len(train_dataloader) * args.n_epochs)
    
    model.train()
    old_f_score = 0
        
    for epoch in range(args.n_epochs):
        print('\n ======== Epoch {:} / {:} ========'.format(epoch + 1, args.n_epochs))
        total_loss = 0
        
        for step , batch in enumerate(train_dataloader):          
            model.zero_grad()
            input_ids = batch['text'].to(device) 
            attention_mask = batch['mask'].to(device) 
            targets_batch = batch['target'].to(device).to(torch.int64)

            output = model(input_ids, attention_mask, targets_batch)
            loss = output[0]
            total_loss += loss
            optimizer.zero_grad()
            if step%100 == 0: print(step, " : ", loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_loss / len(train_dataloader)
        print("Avg loss: ", avg_train_loss.item())
        f_score = eval_model(model, val_dataloader)
        # if f_score < old_f_score: break
        # else: old_f_score = f_score  
        
    return model


def eval_model(model, val_dataloader):
    
    y_pred_list, targets = [], []  
    total_loss = 0
    
    for batch in val_dataloader:
        input_ids = batch['text'].to(device) 
        attention_mask = batch['mask'].to(device) 
        targets_batch = batch['target'].to(device).to(int)

        with torch.no_grad():
            output = model(input_ids, attention_mask, targets_batch)
        loss = output[0]
        logits = output[1]
        total_loss += loss.item()
        
        y_pred_list += np.squeeze(logits.cpu().detach().numpy())[:,1].tolist()
        targets += targets_batch.cpu().detach().numpy().tolist()
    
    print(total_loss)
    targets = np.array(targets)
    y_pred_list = np.array(y_pred_list)
    f_score = metrics.calculate_metrics(targets, y_pred_list)
    
    return f_score


def configure_model(args, bert_base_uncased):
    config = BertConfig()    
    model = Model(config)
    tokenizer = BertTokenizer.from_pretrained(bert_base_uncased, do_lower_case=True)

    # Freeze all parameters except for linear layer
    # for param in classifier.image_embedder.parameters():
    #     param.requires_grad = False
    
    if device == 'cuda':
        model.to(device)
        
    optimizer = AdamW(model.parameters(),
                    lr = args.learning_rate,
                    eps = args.adam_epsilon,
                    no_deprecation_warning=True
                    # weight_decay = 0.01
                    )
    return model, tokenizer, optimizer


def main():
    preprocess_data.set_seed(args)
    model, tokenizer, optimizer = configure_model(args, bert_base_uncased="prajjwal1/bert-small")
    
    # Load data from files
    text_data, image_data, topics = preprocess_data.load_dataset(args, images = False)
    text_data = encode_dataset(tokenizer, text_data)

    
    # Preprocess visual data
    # image_data = preprocess_data.load_images(image_data, args)  
    # text_data, image_list = encode_dataset( text_data, image_data)
    # image_list = preprocess_data.apply_image_processor(image_list, image_processor)
    # combined_data = preprocess_data.remove_mismatching_prompts(image_list, text_data)
    
    # Create dataloader
    train_ratio = 0.75
    validation_ratio = 0.25
    train_cutoff = int(train_ratio*len(text_data))
    val_cutoff = int(train_ratio*len(text_data) + validation_ratio*len(text_data))
    np.random.shuffle(text_data)

    combined_train_data = np.array([x for x in text_data[:train_cutoff]])
    train_data = CustomDataset(combined_train_data)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
    
    combined_val_data = np.array([x for x in text_data[train_cutoff:val_cutoff]])
    val_data = CustomDataset(combined_val_data)
    val_sampler = RandomSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=args.batch_size)
    
    # combined_eval_data = np.array([x for x in text_data[-eval_cutoff:]])
    # eval_data = CustomDataset(combined_eval_data)
    # eval_sampler = RandomSampler(eval_data)
    # eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)
    
    model = train_model(args, model, optimizer, train_dataloader, val_dataloader)
    metrics.save_model(model, "_")
    
    

if __name__ == '__main__':
    if args.folder_path:
        if args.prompts_path == None: args.prompts_path = str(args.folder_path) + "prompts.txt"
        if args.responses_path == None: args.responses_path = str(args.folder_path) + "responses.txt"
        if args.topics_path == None: args.topics_path = str(args.folder_path) + "topics.txt"
        if args.topic_dist_path == None: args.topic_dist_path = str(args.folder_path) + "topics_dist.txt"
        if args.labels_path == None: args.labels_path = str(args.folder_path) + "targets.txt"
    if args.folder_path_images:
        if args.images_path == None: args.images_path = str(args.folder_path_images)
        if args.image_ids_path == None: args.image_ids_path = str(args.folder_path_images) + "image_ids.txt"
        if args.image_prompts_path == None: args.image_prompts_path = str(args.folder_path_images) + "image_questions.txt"
    main() 

