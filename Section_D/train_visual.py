import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import numpy as np
import preprocess_data, load_models
from transformers import get_linear_schedule_with_warmup, BertTokenizer
import torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
import metrics
from sentence_transformers import SentenceTransformer


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

parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
parser.add_argument('--batch_size', type=int, default=12, help='Specify the test batch size')
parser.add_argument('--learning_rate', type=float, default=5e-5, help='Specify the initial learning rate')
parser.add_argument('--adam_epsilon', type=float, default=1e-6, help='Specify the AdamW loss epsilon')
parser.add_argument('--lr_decay', type=float, default=0.85, help='Specify the learning rate decay rate')
parser.add_argument('--dropout', type=float, default=0.1, help='Specify the dropout rate')
parser.add_argument('--n_epochs', type=int, default=1, help='Specify the number of epochs to train for')


# Global Variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
args = parser.parse_args()


# def create_dataset(data_train):
#     """
#         Takes in data_train that is a class which include the prompts, responses, images etc.
#         It extracts the different elements and turns them into torch tensors, concatenating the 
#         encoded text with the encoded image, and concatenating their attention masks
        
#         Then takes these along with the targets to create a dataset and then dataloader
#     """
#     # Extract the different elements and convert them to numpy array to make the conversion to torch tensors easier
#     encoded_texts = np.array([x.text for x in data_train])
#     mask = np.array([x.mask for x in data_train])
#     targets = np.array([x.target for x in data_train])
#     pixels = np.array([x.pixels for x in data_train])
    
#     # Turn into torch tensor and send to cuda, squeezing the dimensions down to 2 dims for all
#     encoded_texts = torch.tensor(encoded_texts).to(device).squeeze()
#     mask = torch.tensor(mask).to(device).squeeze()
#     targets = torch.tensor(targets).to(device)
#     pixels = torch.tensor(pixels).to(device).squeeze()
    
#     text_data = TensorDataset(encoded_texts, mask, targets)
#     # train_sampler = RandomSampler(train_data)
#     train_dataloader = DataLoader(text_data, batch_size=args.batch_size)
#     image_dataloader = DataLoader(pixels, batch_size=args.batch_size)
    
#     return train_dataloader, image_dataloader, targets


def add_word_embeddings(model, data_train):
    
    model.eval()
    
    for data in data_train:
        # Get prompt and response and their encoding
        prompt, response = data.prompt, data.response
        prompt_encoding = model.encode(prompt)
        response_encoding = model.encode(response)
        
        combined_encoding = np.concatenate((prompt_encoding, response_encoding), axis=0)
        data.add_encodings(combined_encoding,"")
    
    return data_train
    

def get_image_CLS_tokens( visual_model, dataloader):
    
    visual_model.eval()
    CLS_tokens = torch.empty((0, 768), dtype=torch.float32)
    for batch in dataloader:
        pixels_batch = batch.to(device)
        
        with torch.no_grad():
            outputs = visual_model(pixels_batch) 
            
        last_hidden_state = outputs.hidden_states[-1].detach().cpu()
        CLS_tokens_batch = last_hidden_state[:,0,:]
        CLS_tokens = torch.cat((CLS_tokens, CLS_tokens_batch), dim=0)
        
    return CLS_tokens





def train_classification_head(args, train_dataloader, shape, num_labels=1):
    classifier = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(shape, num_labels)
    )
    classifier.to(device)    
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.SGD(
        classifier.parameters(),
        # eps = args.adam_epsilon,
        lr=args.learning_rate
        )
    # scheduler = get_linear_schedule_with_warmup(optimizer,
    #                                             num_warmup_steps = 20,
    #                                             num_training_steps = len(train_dataloader) * args.n_epochs)
    
    for epoch in range(args.n_epochs):
        print("Epoch: ", epoch, " of ", args.n_epochs)
        classifier.train()
        
        for batch in train_dataloader:
            classifier.zero_grad()
            # optimizer.zero_grad()
            word_embedding = (batch[0].float()).to(device)
            targets_batch = (batch[1].float()).to(device)
            logits = classifier(word_embedding).squeeze(dim=1)
            loss = criterion(logits, targets_batch)
            loss.backward()
            
            optimizer.step()
            # scheduler.step()
            print("Loss: ", loss.item())
    
    # train_BERT.plot_loss(args, loss_values, avg_train_loss)
    return classifier

    

def encode_dataset(text_data, image_data):
    """
        Encodes the entire text data by calling the function encode_data().
        Adds the corresponding images to a list
    """
    
    # Encode the prompts/responses and save the attention masks, padding applied to the end
    image_list = []
    index_to_remove = []
    for data in text_data:
        # Find the corresponding image so that later they can be concatenated together
        image = preprocess_data.find_corresponding_image_id(data.prompt, image_data)[1]
        if image in image_data: image_list.append(image)
        else: index_to_remove.append(text_data.index(data))
            
    for index in list(reversed(index_to_remove)):
        text_data.pop(index)
    return text_data, image_list

def main():    
    preprocess_data.set_seed(args)
    
    # Load Models and Optimisers
    text_embedder = SentenceTransformer('sentence-transformers/sentence-t5-base').to(device)
    visual_model, image_processor = load_models.load_vision_transformer_model()
    
    # Preprocess textual data
    text_data, image_data, topics = preprocess_data.load_dataset(args)
    # text_data = preprocess_data.remove_incomplete_data(text_data, image_data)
    
    # Shuffling real data to create synthetic
    text_data = [x for x in text_data if x.target == 1]
    text_data = preprocess_data.permute_data(text_data, topics, args)
    np.random.shuffle(text_data)
    
    # Preprocess visual data
    # image_data = preprocess_data.load_images(image_data, args)  
    # text_data, image_list = encode_dataset( text_data[:100], image_data)
    # image_list = preprocess_data.encode_images(image_list, image_processor)
    # data_train = preprocess_data.remove_mismatching_prompts(image_list, text_data)

    data_train = np.array([x for x in text_data[:100]])
    print("Dataset Size: ", len(data_train))

    # Concatenate vision transformer and BERT hidden states
    data_train = add_word_embeddings(text_embedder, data_train)
    # VT_CLS_tokens = get_hidden_state_image(visual_model, image_dataloader)
    
    targets = torch.Tensor([x.target for x in data_train])
    text_encodings = torch.Tensor([x.text for x in data_train])
    
    train_data = TensorDataset(text_encodings, targets)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size)
    classifier = train_classification_head(args, train_dataloader, shape=text_encodings.shape[1])

  
    # Eval on Test
    logits = classifier((text_encodings[500:]).to(device))
    y_pred_all = np.array(logits.detach().cpu())
    metrics.calculate_metrics(targets[500:], y_pred_all)
    metrics.save_model(classifier, "_")
    

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