import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import numpy as np
import matplotlib as plt
import datetime
import time
import preprocess_data, load_models
from transformers import get_linear_schedule_with_warmup


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
parser.add_argument('--save_path', type=str, help="Where to save predicted values")
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



def format_time(elapsed):
    """
        Used for displaying time when measuring time lengths
    """
    
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))
        
        

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
    # image = np.array([x.image for x in data_train])
    # image_mask = np.array([x.image_mask for x in data_train])
    
    # Turn into torch tensor and send to cuda, squeezing the dimensions down to 2 dims for all
    encoded_texts = torch.tensor(encoded_texts).to(device).squeeze()
    mask = torch.tensor(mask).to(device).squeeze()
    targets = torch.tensor(targets).to(device)
    # image = torch.tensor(image).to(device)
    # image_mask = torch.tensor(image_mask).to(device)

    # Concatenate the visual and textual elements together
    # input = torch.cat((encoded_texts, image),1)
    # input_mask = torch.cat((mask, image_mask),1)
    # input = input.squeeze(1)
    # input_mask = input_mask.squeeze(1)
    
    # Create and return dataloader
    # train_data = TensorDataset(input, input_mask, targets)
    train_data = TensorDataset(encoded_texts, mask, targets)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
    return train_dataloader



def train_BERT_model(args, optimizer, model, device, train_dataloader):
    """
        Runs the training using the train_dataloader created in create_datasets().
        Splits the data into batches specified by the args specified by user, runs the training, and returns the last outputs
    """
    # Specify the sequence of indices/keys used in data loading, want each epoch to have a different order to prevent over fitting
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
            input_batch = (batch[0].to(device)).squeeze(1)
            input_mask_batch = (batch[1].to(device)).squeeze(1)
            target_batch = batch[2].to(device) 
            
            # First compute the outputs given the current weights, and the current and total loss
            outputs = model(input_ids=input_batch, attention_mask=input_mask_batch, labels=target_batch)
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
    
    return outputs.hidden_states[-1], avg_train_loss



def apply_classification_head(BERT_hidden_state, VT_hidden_state):
    """
        Takes in the hidden states from the two systems, concatenates them and applies a linear classification head.
        The function returns logits
    """
    BERT_hidden_state = torch.tensor(BERT_hidden_state)
    VT_hidden_state = torch.tensor(VT_hidden_state)
    
    concatenated_states = torch.cat((BERT_hidden_state, VT_hidden_state))
    classification_head = torch.nn.Linear(concatenated_states.size(1), 2)
    logits = classification_head(concatenated_states)
    
    return logits



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
    model = load_models.load_BERT_model(bert_base_uncased, device)
    optimizer = load_models.load_optimiser(model, args)
    
    # Preprocess textual data
    text_data, image_data, topics = preprocess_data.load_dataset(args)
    text_data = preprocess_data.remove_incomplete_data(text_data, image_data)
    text_data = preprocess_data.permute_data(text_data, topics, args)
    
    # Preprocess visual data
    image_data = preprocess_data.load_images(image_data, args)   
    text_data, image_list = preprocess_data.encode_dataset(text_data, image_data, bert_base_uncased)
            
    image_list = preprocess_data.encode_images(image_list, image_processor)
    data_train = preprocess_data.remove_mismatching_prompts(image_list, text_data)
    
    pixel_values = [x.pixel_values for x in data_train]
    pixel_values = torch.tensor(pixel_values).to(device)
    
    # Obtain hidden states for VT and BERT
    with torch.no_grad():
        VT_outputs = visual_model(pixel_values) 
        VT_hidden_state = VT_outputs.hidden_states[-1]

    train_dataloader = create_dataset(data_train)
    BERT_hidden_state,avg_train_loss = train_BERT_model(args, optimizer, model, device, train_dataloader)
    print(len(BERT_hidden_state), len(VT_hidden_state))
    
    
    save_model(args, model, avg_train_loss)

if __name__ == '__main__':
    main() 