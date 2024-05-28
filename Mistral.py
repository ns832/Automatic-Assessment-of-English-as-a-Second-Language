import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
import Section_D.preprocess_data as preprocess_data
import Section_D.metrics as metrics
import numpy.random as random

# An optional folder_path can be supplied if multiple files are in the same directory - any file_path not given is then assumed to be in this folder
parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--folder_path', type=str, default=None, help='Load path of the folder containing prompts, responses etc.')
parser.add_argument('--prompts_path', type=str, default=None, help='Load path of question training data')
parser.add_argument('--responses_path', type=str, default=None, help='Load path of answer training data')
parser.add_argument('--prompt_ids_path', type=str, default=None, help='Load path of prompt ids')
parser.add_argument('--topic_dist_path', type=str, default=None, help='Load path of prompt distribution')
parser.add_argument('--topics_path', type=str, default=None, help='Load path of topics')
parser.add_argument('--groups_path', type=str, default=None, help='Load path of groups')
parser.add_argument('--labels_path', type=str, default=None ,help='Load path to labels')
parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')


# Global Variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
args = parser.parse_args()

def set_seed(args):
    seed_val = args.seed
    random.seed(seed_val), np.random.seed(seed_val)
    torch.manual_seed(seed_val), torch.cuda.manual_seed_all(seed_val)
    return

def initialise_model(model_name = "mistralai/Mistral-7B-v0.1"):
    """
        Instantiates the tokeniser and model. Configuration for the two is set here.
    """
    # Load in the tokenizer and set pad token
    model_name = "HuggingFaceH4/zephyr-7b-beta"
    tokenizer = AutoTokenizer.from_pretrained(model_name,add_special_tokens=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Set configuration for model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    # Load pretrained model
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
    )  
    return model, tokenizer


def create_prompts(text_data):
    """
        Creates prompts to feed into the LLM with the prompt and response pairs.
        Prompts are formulated using the suggested [INST] [/INST] format suggested by Mistral.
    """    
    # Iterates through the text_data list and creates prompts to feed into the model
    for data in  text_data :
        Mistral_prompt = """<s>[INST] Respond to the prompts with only 'On' or 'Off' depending on if the answer is relevant to the question asked </s> [Question] """ + data.prompt.strip().lower().replace("</s>", "") + """. [Answer] """  + data.response.strip().lower().replace("</s>", "") + ". [/INST]"
        data.text = Mistral_prompt
    # Print a random prompt to check the format of the question is as desired
    print("Random prompt: ", text_data[np.random.randint(0, len(text_data))].text)
    return text_data


def create_dataloader(tokenizer, text_data, device):
    """
        Creates a dataloader with the encoded input ids and attention masks.
        Pad token is set to the eos token to allow for all encodings to be equal length.
    """
    # Encode the prompt
    for prompt in text_data:
        encoded_prompt = tokenizer(prompt.text, padding='max_length', truncation=True, max_length=512, add_special_tokens=False)
        prompt.add_encodings(encoded_prompt["input_ids"], encoded_prompt["attention_mask"])
    
    # Create torch tensors for the text, mask and target from the attributes stored in the Text class
    text = torch.tensor([x.text for x in text_data])
    mask = torch.tensor([x.mask for x in text_data])
    target = torch.tensor([x.target for x in text_data])
    group = torch.tensor([x.group for x in text_data])
    
    # Put the data into a DataLoader
    train_data = TensorDataset(text, mask, target, group)
    # train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, batch_size=1)
    
    return train_dataloader


def model_generate(model, tokenizer, device, train_dataloader):
    """
        Iterated through the data in batches, calling the LLM for each batch to extract the outputs.
        From there the outputs are passed through get_probabilities() to extract the probability of 
        an 'on topic' response.
    """
    # Initialise empty array and streamer to automatically show the decoded model output. 
    # Group refers to the set of 5 questions that make up a full question.
    on_probabilities = []
    groups, targets = [], []
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Iterate through the dataloder and feed the prompts into the model
    for step, batch in enumerate(train_dataloader):
        with torch.no_grad():
            b_input_ids = batch[0].to(device)
            b_att_msks = batch[1].to(device)
            output = model.generate(input_ids=b_input_ids, 
                                    attention_mask=b_att_msks, 
                                    pad_token_id=tokenizer.eos_token_id,
                                    do_sample = False,
                                    return_dict_in_generate=True,
                                    max_new_tokens=1,
                                    streamer=streamer,
                                    output_scores=True, 
                                    use_cache=False)
            # Use model scores to get the probability for 'On' via get_probabilities()
            print(step, " of ", len(train_dataloader), " Target: ", batch[2])
            targets += batch[2]
            groups += batch[3]
            scores = (output.scores)[0].cpu().numpy()
            for score in scores:
                logits = np.array(score)
                on_probabilities.append(get_probabilities(logits, tokenizer))
                
    # Get two arrays, one with the probabilities and the other with which group it corresponds to
    on_probabilities = np.array(on_probabilities)
    groups = np.array(groups, dtype=int)
    targets = np.array(targets, dtype=int)
    
    # Instantiate empty arrays which are used to store the cumulative probabilities & counts
    combined_predictions = np.zeros(len(set(groups)))
    combined_predictions_count = np.zeros(len(set(groups)), dtype=int)
    combined_targets = np.zeros(len(set(groups)), dtype=int)

    # Collate all probabilities corresponding to the same group, counting the instances in combined_predictions_count
    for prob, group, target in zip(on_probabilities, groups, targets):
        if combined_predictions_count[group] == 0:
            combined_targets[group] = target
        else:
            assert combined_targets[group] == target
        combined_predictions[group] = combined_predictions[group] + prob
        combined_predictions_count[group] = combined_predictions_count[group] + 1

    on_probabilities = [x / y for x,y in zip(combined_predictions, combined_predictions_count)]
    on_probabilities = np.array(on_probabilities)

    return on_probabilities, combined_targets


def get_probabilities(logits, tokenizer):
    """
        Takes in the output of the model and converts it into a probability for on topic.
        Softmax is applied to convert the logits into probabilities.
    """
    # Retrieve position of 'On' and 'Off' from the tokenizer dictionary
    on_token = tokenizer("On").input_ids
    off_token = tokenizer("Off").input_ids
    
    # Obtain scores fromm those positions and apply softmax to get overall score
    on_logits = logits[on_token[1]] 
    off_logits = logits[off_token[1]] 
    score = np.exp(on_logits) / (np.exp(on_logits) + np.exp(off_logits))
    print("Logits:", on_logits, off_logits, "Score:", score)
    
    return score


def main(args):
    # Instantiate model and tokenizer
    set_seed(args)
    model, tokenizer = initialise_model()
    
    # Load dataset and shuffle data before creating prompts
    text_data, __ , __ = preprocess_data.load_dataset(args, images = False)
    
    # Only Section E needs to worry about groups, hence if no groups are input each input is simply given a distinct group
    if args.groups_path: 
        groups = np.loadtxt(args.groups_path, dtype=int)
    else: groups = np.arange(0, len(text_data))
    for data, group in zip(text_data, groups):
        data.group =  group
        
    # Shuffling is done here (as well as during create_dataloader()) if only a subsection of the data is used for testing purposes
    np.random.shuffle(text_data)
    text_data = create_prompts(text_data)
    
    # Run the evaluation and obtain scores for 'On-topic'
    train_dataloader = create_dataloader(tokenizer, text_data, device)
    on_probabilities, targets = model_generate(model, tokenizer, device, train_dataloader)   
    prob_list = np.array(on_probabilities)
 
    # Create Precision-Recall Graph
    metrics.calculate_metrics(targets, prob_list)
    metrics.calculate_normalised_metrics(targets, prob_list)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.folder_path:
        if args.prompts_path == None: args.prompts_path = str(args.folder_path) + "prompts.txt"
        if args.responses_path == None: args.responses_path = str(args.folder_path) + "responses.txt"
        if args.topics_path == None: args.topics_path = str(args.folder_path) + "topics.txt"
        if args.topic_dist_path == None: args.topic_dist_path = str(args.folder_path) + "topics_dist.txt"
        if args.labels_path == None: args.labels_path = str(args.folder_path) + "targets.txt"
    main(args)