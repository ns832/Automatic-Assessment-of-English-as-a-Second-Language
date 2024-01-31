import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import numpy as np
import datetime
import time
import preprocess_data, load_models
from transformers import get_linear_schedule_with_warmup, BertTokenizer
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
parser.add_argument('--predictions_save_path', type=str, help="Where to save predicted values")
parser.add_argument('--model_path', type=str, default="/scratches/dialfs/alta/relevance/ns832/results/train_model_unseen/bert_model_0.14654680755471014.pt")
parser.add_argument('--save_path', type=str, help="Where to save predicted values")
parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Specify the initial learning rate')
parser.add_argument('--adam_epsilon', type=float, default=1e-6, help='Specify the AdamW loss epsilon')
parser.add_argument('--lr_decay', type=float, default=0.85, help='Specify the learning rate decay rate')
parser.add_argument('--dropout', type=float, default=0.1, help='Specify the dropout rate')
parser.add_argument('--n_epochs', type=int, default=1, help='Specify the number of epochs to train for')
parser.add_argument('--labels_path', type=str, help='Load path to labels')


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
    
    # Turn into torch tensor and send to cuda, squeezing the dimensions down to 2 dims for all
    encoded_texts = torch.tensor(encoded_texts).to(device).squeeze()
    mask = torch.tensor(mask).to(device).squeeze()
    targets = torch.tensor(targets).to(device)
    
    train_data = TensorDataset(encoded_texts, mask, targets)
    # train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size)
    
    return train_dataloader, targets


# def train_BERT_model(args, optimizer, model, device, train_dataloader):
#     """
#         Runs the training using the train_dataloader created in create_datasets().
#         Splits the data into batches specified by the args specified by user, runs the training, and returns the last outputs
#     """
#     # Specify the sequence of indices/keys used in data loading, want each epoch to have a different order to prevent over fitting
#     total_steps = len(train_dataloader) * args.n_epochs
#     scheduler = get_linear_schedule_with_warmup(optimizer,
#                                                 num_warmup_steps = 306,
#                                                 num_training_steps = total_steps)
    
#     model.train()
#     hidden_states = torch.empty((0, 256, 512), dtype=torch.float32)
    
#     for epoch in range(args.n_epochs):
#         print("")
#         print('======== Epoch {:} / {:} ========'.format(epoch + 1, args.n_epochs))
#         print('Training...')
#         t0 = time.time()
#         total_loss = 0
#         model.train()
#         model.zero_grad() 
        
#         for step, batch in enumerate(train_dataloader):
#             if step % 40 == 0:
#                 elapsed = str(datetime.timedelta(seconds=int(round(time.time() - t0))))
#                 print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
                
#             model.zero_grad()
#             input_batch = (batch[0].to(device)).squeeze(1)
#             input_mask_batch = (batch[1].to(device)).squeeze(1)
#             target_batch = batch[2].to(device) 
            
#             # First compute the outputs given the current weights, and the current and total loss
#             outputs = model(input_ids=input_batch, attention_mask=input_mask_batch, labels=target_batch)
#             loss = outputs.loss
#             total_loss += loss.item()
#             # print("loss.item is", loss.item())
            
#             # Then perform the backwards pass through the nn, updating the weights based on gradients
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             scheduler.step()
#             hidden_states = torch.cat((hidden_states, outputs.hidden_states[-1].detach().cpu()), dim=0)
        
#     avg_train_loss = total_loss / len(train_dataloader)
#     print("")
#     print("  Average training loss: {0:.2f}".format(avg_train_loss))
#     print("  Training epoch took: {:}".format(str(datetime.timedelta(seconds=int(round(time.time() - t0))))))
    
#     return hidden_states, avg_train_loss


def get_hidden_state(model, device, dataloader):

    model.eval()    
    print("Started evaluation")
    hidden_states = torch.empty((0, 256, 512), dtype=torch.float32)
    for batch in dataloader:
        
        input_batch = (batch[0].to(device)).squeeze(1)
        input_mask_batch = (batch[1].to(device)).squeeze(1)
        target_batch = batch[2].to(device) 

        with torch.no_grad():
            BERT_outputs = model(input_ids=input_batch, attention_mask=input_mask_batch, labels=target_batch, output_hidden_states = True)

        hidden_states = torch.cat((hidden_states, BERT_outputs.hidden_states[-1].detach().cpu()), dim=0)
    return hidden_states



def train_classification_head(args, optimizer, hidden_states, targets):
    head = nn.Sequential(
        nn.Linear(hidden_states.shape[2], 1),
        nn.ReLU()
    )
    head_2 = nn.Sequential(
        nn.Linear(hidden_states.shape[1], 1),
        nn.ReLU()
    )
    # Just take the first token cls token from the BERT
    targets = targets.float()
    model = ClassificationHead(head, head_2).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)  # Use the learning rate from args

    print("Training Classification Head")
    for epoch in range(1000):
        model.train()  # Ensure the model is in training mode
        optimizer.zero_grad()
        outputs = model(hidden_states.to(device))
        loss = criterion(outputs, targets)
        print("Epoch {}: Loss: {}".format(epoch, loss.item()))
        loss.backward()
        optimizer.step()

    file_path = str(args.save_path) + '/trial_classification_head.pt'
    torch.save(model.state_dict(), file_path)  # Save only the model state_dict
    print(file_path)
    return model


def save_model(args, model, avg_train_loss):
    """
        Saves the model so it can be accessed for evaluationg
    """
    file_path = str(args.save_path) + '/bert_model_' + str(avg_train_loss) + '.pt'
    print(file_path)
    torch.save(model, file_path)
    return


def main():    
    bert_base_uncased = "prajjwal1/bert-small"
    preprocess_data.set_seed(args)
    
    # Load Models and Optimisers
    visual_model, image_processor = load_models.load_vision_transformer_model()
    BERT_model = load_models.load_BERT_model(bert_base_uncased, device)
    optimizer = load_models.load_optimiser(BERT_model, args)
    
    # Preprocess textual data
    text_data, image_data, topics = preprocess_data.load_dataset(args)
    text_data = preprocess_data.remove_incomplete_data(text_data, image_data)
    
    # print("Shuffling real data to create synthetic")
    text_data = [x for x in text_data if x.target == 1]
    text_data = preprocess_data.permute_data(text_data[:500], topics, args)
    np.random.shuffle(text_data)
    text_data = text_data[:500]
    
    # Preprocess visual data
    image_data = preprocess_data.load_images(image_data, args)  
    tokenizer = BertTokenizer.from_pretrained(bert_base_uncased, do_lower_case=True) 
    text_data, image_list = preprocess_data.encode_dataset(tokenizer, text_data, image_data)
            
    image_list = preprocess_data.encode_images(image_list, image_processor)
    data_train = preprocess_data.remove_mismatching_prompts(image_list, text_data)
    
    pixels = np.array([x.pixels for x in data_train])
    pixels = torch.tensor(pixels).to(device).squeeze()
    
    # Obtain hidden states for VT and BERT
    with torch.no_grad():
        VT_outputs = visual_model(pixels) 
        VT_hidden_state = VT_outputs.hidden_states[-1]

    # Create dataloader with the text data
    dataloader, targets = create_dataset(data_train)
    print("Dataset Size: ", len(data_train))

    # BERT_hidden_state, avg_train_loss = train_BERT_model(args, optimizer, model, device, train_dataloader)  
    
    # Concatenate vision transformer and BERT hidden states
    BERT_model = load_models.load_trained_BERT_model(args)
    BERT_hidden_state = get_hidden_state(BERT_model, device, dataloader)
    VT_hidden_state = VT_hidden_state[:, :, :BERT_hidden_state.shape[2]] 
    VT_hidden_state = torch.stack([x.cpu() / 100 for x in VT_hidden_state])
    concatenated_outputs = torch.cat((VT_hidden_state.cpu(), BERT_hidden_state.cpu()), dim=1)
    
    # Train the classification head and save both models
    train_classification_head(args, optimizer, concatenated_outputs, targets)
    raise
    save_model(args, BERT_model, "test")

if __name__ == '__main__':
    main() 