import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import numpy as np
import preprocess_data, load_models
import torch.nn as nn
from transformers import BertTokenizer 
import metrics

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--folder_path', type=str, default=None, help='Load path of the folder containing prompts, responses etc.')
parser.add_argument('--prompts_path', type=str, default=None, help='Load path of question training data')
parser.add_argument('--resps_path', type=str, default=None, help='Load path of answer training data')
parser.add_argument('--prompt_ids_path', type=str, default=None, help='Load path of prompt ids')
parser.add_argument('--topics_path', type=str, default=None, help='Load path of topics')
parser.add_argument('--topic_dist_path', type=str, default=None, help='Load path of prompt distribution')
parser.add_argument('--labels_path', type=str, default=None ,help='Load path to labels')

# An optional image folder_path can be supplied if multiple image files are in the same directory - any file_path not given is then assumed to be in this folder
parser.add_argument('--folder_path_images', type=str, default=None, help='Optional path for folder containing image data.')
parser.add_argument('--images_path', type=str, help='Load path of image training data')
parser.add_argument('--image_ids_path', type=str, help='Load path of image ids')
parser.add_argument('--image_prompts_path', type=str, help='Load path of prompts corresponding to image ids')

parser.add_argument('--model_path', type=str, default="/scratches/dialfs/alta/relevance/ns832/results/train_model_unseen/bert_model_0.14654680755471014.pt")
parser.add_argument('--classification_model_path', type=str, help='Load path to trained classification head')

parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
parser.add_argument('--batch_size', type=int, default=100, help='Specify the test batch size')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Specify the initial learning rate')
parser.add_argument('--adam_epsilon', type=float, default=1e-6, help='Specify the AdamW loss epsilon')
parser.add_argument('--lr_decay', type=float, default=0.85, help='Specify the learning rate decay rate')
parser.add_argument('--dropout', type=float, default=0.1, help='Specify the dropout rate')
parser.add_argument('--n_epochs', type=int, default=1, help='Specify the number of epochs to train for')




# Global Variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
args = parser.parse_args()
       
    

def create_dataset(data_train):
    """
        Takes in data_train that is a class which include the prompts, responses, images etc.
        It extracts the different elements and turns them into torch tensors, concatenating the 
        encoded text with the encoded image, and concatenating their attention masks
        
        Then takes these along with the targets to create a dataset and then dataloader
    """
    # Extract the different elements and convert them to numpy array to make the conversion to torch tensors easier
    encoded_texts = np.array([x.text for x in data_train])
    mask = np.array([x.mask for x in data_train])
    targets = np.array([x.target for x in data_train])
    
    # Turn into torch tensor and send to cuda, squeezing the dimensions down to 2 dims for all
    encoded_texts = torch.tensor(encoded_texts).to(device).squeeze()
    mask = torch.tensor(mask).to(device).squeeze()
    targets = torch.tensor(targets).to(device)
    
    train_data = TensorDataset(encoded_texts, mask, targets)
    # train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size)
    
    return train_dataloader, targets


def get_hidden_state(model, device, dataloader):

    model.eval()    
    print("Started evaluation")
    CLS_tokens = torch.empty((0, 512), dtype=torch.float32)
    for batch in dataloader:
        
        input_batch = (batch[0].to(device)).squeeze(1)
        input_mask_batch = (batch[1].to(device)).squeeze(1)
        target_batch = batch[2].to(device) 

        with torch.no_grad():
            BERT_outputs = model(input_ids=input_batch, attention_mask=input_mask_batch, labels=target_batch, output_hidden_states = True)

        last_hidden_states = BERT_outputs.hidden_states[-1].detach().cpu()
        CLS_tokens_batch = last_hidden_states[:,0,:]
        CLS_tokens = torch.cat((CLS_tokens, CLS_tokens_batch), dim=0)
        
    return CLS_tokens


def get_hidden_state_image(data_train, visual_model):
    pixels = np.array([x.pixels for x in data_train])
    pixels = torch.tensor(pixels).to(device).squeeze()
    
    with torch.no_grad():
        outputs = visual_model(pixels) 
        last_hidden_state = outputs.hidden_states[-1].detach().cpu()
        CLS_tokens = last_hidden_state[:,0,:]
        
    return CLS_tokens

# def get_hidden_state(model, visual_model, device, eval_dataloader):

#     model.eval()    
#     print("Started evaluation")
    
#     concatenated_outputs = []
#     for batch in eval_dataloader:
        
#         input_batch = (batch[0].to(device)).squeeze(1)
#         input_mask_batch = (batch[1].to(device)).squeeze(1)
#         target_batch = batch[2].to(device) 
#         pixels = batch[3].to(device)

#         with torch.no_grad():
#             BERT_outputs = model(input_ids=input_batch, attention_mask=input_mask_batch, labels=target_batch)
#             VT_outputs = visual_model(pixels) 
        
#         BERT_hidden_state = torch.tensor(BERT_outputs.hidden_states[-1])
#         VT_hidden_state = torch.tensor((VT_outputs.hidden_states[-1])[:, :, :BERT_hidden_state.shape[2]])
#         concatenated_outputs.append(torch.cat((VT_hidden_state.cpu(), BERT_hidden_state.cpu()), dim=1))

#     concatenated_outputs = torch.cat(concatenated_outputs, dim=0).squeeze()
    # return concatenated_outputs


def load_classification_head(hidden_states, num_labels=1):
    classifier = nn.Linear(hidden_states.shape[1], num_labels)
    classifier.load_state_dict(torch.load(args.classification_model_path))
    classifier.eval()
    return classifier


def main():    
    bert_base_uncased = "prajjwal1/bert-small"
    preprocess_data.set_seed(args)
    
    # Load Models and Optimisers
    visual_model, feature_extractor = load_models.load_vision_transformer_model()
    BERT_model = load_models.load_trained_BERT_model(args)
    
    # Preprocess textual data
    text_data, image_data, topics = preprocess_data.load_dataset(args)
    text_data = preprocess_data.remove_incomplete_data(text_data, image_data)
    
    # print("Shuffling real data to create synthetic")
    text_data = [x for x in text_data if x.target == 1]
    text_data = preprocess_data.permute_data(text_data[:500], topics, args)
    np.random.shuffle(text_data)
    
    # Preprocess visual data
    image_data = preprocess_data.load_images(image_data, args)   
    tokenizer = BertTokenizer.from_pretrained(bert_base_uncased, do_lower_case=True)
    text_data, image_list = preprocess_data.encode_dataset(tokenizer, text_data, image_data)
    image_list = preprocess_data.apply_image_processor(image_list, feature_extractor)
    data_train = preprocess_data.remove_mismatching_prompts(image_list, text_data)
    
    
    # Create dataloader with the text data
    dataloader, targets = create_dataset(data_train)
    print("Dataset Size: ", len(data_train))
    
    # Obtain hidden states for VT and BERT
    BERT_CLS_tokens = get_hidden_state(BERT_model, device, dataloader)
    VT_CLS_tokens = get_hidden_state_image(data_train, visual_model)
    hidden_state = torch.cat((VT_CLS_tokens.cpu(), BERT_CLS_tokens.cpu()), dim=1)
    
    classification_head = load_classification_head(hidden_state)
    results = classification_head(hidden_state)
    y_pred_all = np.array(results.detach())
    targets = np.array([x.target for x in data_train])
    metrics.calculate_metrics(targets, y_pred_all)

if __name__ == '__main__':
    if args.folder_path:
        if args.prompts_path == None: args.prompts_path = str(args.folder_path) + "prompts.txt"
        if args.resps_path == None: args.resps_path = str(args.folder_path) + "responses.txt"
        if args.topics_path == None: args.topics_path = str(args.folder_path) + "topics.txt"
        if args.topic_dist_path == None: args.topic_dist_path = str(args.folder_path) + "topics_dist.txt"
        if args.labels_path == None: args.labels_path = str(args.folder_path) + "targets.txt"
    if args.folder_path_images:
        if args.images_path == None: args.images_path = str(args.folder_path_images)
        if args.image_ids_path == None: args.image_ids_path = str(args.folder_path_images) + "image_ids.txt"
        if args.image_prompts_path == None: args.image_prompts_path = str(args.folder_path_images) + "image_questions.txt"
    main() 