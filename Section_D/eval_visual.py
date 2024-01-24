import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import precision_recall_curve
import preprocess_data, load_models
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer 


parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--images_path', type=str, help='Load path of image training data')
parser.add_argument('--batch_size', type=int, default=100, help='Specify the test batch size')
parser.add_argument('--prompts_path', type=str, help='Load path to test prompts as text')
parser.add_argument('--resps_path', type=str, help='Load path of answer training data')
parser.add_argument('--prompt_ids_path', type=str, help='Load path of prompt ids')
parser.add_argument('--topics_path', type=str, help='Load path of question training data')
parser.add_argument('--topic_dist_path', type=str, help='Load path of question training data')
parser.add_argument('--image_ids_path', type=str, help='Load path of image ids')
parser.add_argument('--image_prompts_path', type=str, help='Load path of prompts corresponding to image ids')
parser.add_argument('--classification_model_path', type=str, help='Load path to trained classification head')
parser.add_argument('--model_path', type=str, help='Load path to trained model')
parser.add_argument('--predictions_save_path', type=str, help="Where to save predicted values")
parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Specify the initial learning rate')
parser.add_argument('--adam_epsilon', type=float, default=1e-6, help='Specify the AdamW loss epsilon')
parser.add_argument('--lr_decay', type=float, default=0.85, help='Specify the learning rate decay rate')
parser.add_argument('--dropout', type=float, default=0.1, help='Specify the dropout rate')
parser.add_argument('--n_epochs', type=int, default=1, help='Specify the number of epochs to train for')



# Global Variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
args = parser.parse_args()
        

class ClassificationHead(nn.Module):
    def __init__(self, head, head_2):
        super(ClassificationHead, self).__init__()
        self.head = head
        self.head_2 = head_2
    def forward(self, x):
        x = self.head(x).squeeze()
        x = self.head_2(x).squeeze()
        return F.softmax(x, dim=0)
    
    
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
    pixels = np.array([x.pixels for x in data_train])
    
    
    # Turn into torch tensor and send to cuda, squeezing the dimensions down to 2 dims for all
    pixels = torch.tensor(pixels).to(device).squeeze()
    mask = torch.tensor(mask).to(device).squeeze()
    targets = torch.tensor(targets).to(device)
    encoded_texts = torch.tensor(encoded_texts).to(device)
    
    # Create and return dataloader
    train_data = TensorDataset(encoded_texts, mask, targets, pixels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
    return train_dataloader


def get_hidden_state(model, visual_model, device, eval_dataloader):

    model.eval()    
    print("Started evaluation")
    
    concatenated_outputs = []
    for batch in eval_dataloader:
        
        input_batch = (batch[0].to(device)).squeeze(1)
        input_mask_batch = (batch[1].to(device)).squeeze(1)
        target_batch = batch[2].to(device) 
        pixels = batch[3].to(device)

        with torch.no_grad():
            BERT_outputs = model(input_ids=input_batch, attention_mask=input_mask_batch, labels=target_batch)
            VT_outputs = visual_model(pixels) 
        
        BERT_hidden_state = torch.tensor(BERT_outputs.hidden_states[-1])
        VT_hidden_state = torch.tensor((VT_outputs.hidden_states[-1])[:, :, :BERT_hidden_state.shape[2]])
        concatenated_outputs.append(torch.cat((VT_hidden_state.cpu(), BERT_hidden_state.cpu()), dim=1))

    concatenated_outputs = torch.cat(concatenated_outputs, dim=0).squeeze()
    return concatenated_outputs

        
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
        if precision[i] == 0 or recall[i] == 0:
            precision[i] += 0.1
            recall[i] += 0.1
            
    f_score = np.amax( (1.+0.5**2) * ( (precision * recall) / (0.5**2 * precision + recall) ) )
    print("F0.5 score is:", f_score)

    #display plot
    fig.show()
    fig.savefig('/scratches/dialfs/alta/relevance/ns832/results' + '/eval_visual_' + str(f_score) +  '_plot.jpg', bbox_inches='tight', dpi=150)
    print('/scratches/dialfs/alta/relevance/ns832/results' + '/eval_visual_' + str(f_score) +  '_plot.jpg')

def load_classification_head(hidden_state):
    # Load Classification Head
    head = nn.Sequential(
        nn.Linear(hidden_state.shape[2], 1),
        nn.ReLU()
    )
    head_2 = nn.Sequential(
        nn.Linear(hidden_state.shape[1], 1),
        nn.ReLU()
    )
    classification_head = ClassificationHead(head, head_2)
    classification_head.load_state_dict(torch.load(args.classification_model_path))
    # classification_head.to(device)
    classification_head.eval()
    # classification_head = torch.load(args.classification_model_path)
    return classification_head

def main():    
    bert_base_uncased = "prajjwal1/bert-small"
    preprocess_data.set_seed(args)
    
    # Load Models and Optimisers
    visual_model, feature_extractor = load_models.load_vision_transformer_model()
    model = load_models.load_trained_BERT_model(args)
    
    # Preprocess textual data
    text_data, image_data, topics = preprocess_data.load_dataset(args)
    text_data = preprocess_data.remove_incomplete_data(text_data, image_data)
    text_data = preprocess_data.permute_data(text_data, topics, args)
    
    # Preprocess visual data
    image_data = preprocess_data.load_images(image_data, args)   
    tokenizer = BertTokenizer.from_pretrained(bert_base_uncased, do_lower_case=True)
    text_data, image_list = preprocess_data.encode_dataset(tokenizer, text_data, image_data)
            
    image_list = preprocess_data.encode_images(image_list, feature_extractor)
    data_train = preprocess_data.remove_mismatching_prompts(image_list, text_data)
    
    # Obtain hidden states for VT and BERT
    train_dataloader = create_dataset(data_train)
    hidden_state = get_hidden_state(model, visual_model, device, train_dataloader)
    hidden_state.cpu()
    
    classification_head = load_classification_head(hidden_state)
    results = classification_head(hidden_state)
    y_pred_all = np.array(results.detach())
    targets = np.array([x.target for x in data_train])
    calculate_metrics(targets, y_pred_all)

if __name__ == '__main__':
    main() 