import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, TextStreamer
import Section_D.preprocess_data as preprocess_data
import Section_D.metrics as metrics

# An optional folder_path can be supplied if multiple files are in the same directory - any file_path not given is then assumed to be in this folder
parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--folder_path', type=str, default=None, help='Load path of the folder containing prompts, responses etc.')
parser.add_argument('--prompts_path', type=str, default=None, help='Load path of question training data')
parser.add_argument('--resps_path', type=str, default=None, help='Load path of answer training data')
parser.add_argument('--prompt_ids_path', type=str, default=None, help='Load path of prompt ids')
parser.add_argument('--topic_dist_path', type=str, default=None, help='Load path of prompt distribution')
parser.add_argument('--topics_path', type=str, default=None, help='Load path of topics')
parser.add_argument('--labels_path', type=str, default=None ,help='Load path to labels')


# Global Variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
args = parser.parse_args()



def initialise_model(model_name = "mistralai/Mistral-7B-v0.1"):
    """
        Instantiates the tokeniser and model. Configuration for the two is set here.
    """
    # Load in the tokenizer and set pad token
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
            trust_remote_code=True,
    )  
    return model, tokenizer


def create_prompts(text_data):
    """
        Creates prompts to feed into the LLM with the prompt and response pairs.
        Prompts are formulated using the suggested [INST] [/INST] format suggested by Mistral.
    """    
    # Iterates through the text_data list and creates prompts to feed into the model
    for data in  text_data :
        LLaMA_prompt = """<s>[INST] Respond to the prompts with only 'On' or 'Off' depending on if the answer is relevant to the question asked. For instance:
                        [Question] this chart shows the number of positive and negative responses given in a survey concerning customer satisfaction with a hotel look at the information and talk about the results of the customer survey.
                        [Answer] this charts present results of survey in hotel most negative response was given about the value for money it was about about nine hundred person and the best results get response for question about attitude of staff it was at about nine hundred and half percent and at about half of respondents talk that parking is positive and at about. 
                        Would be converted to:[/INST] On </s>
                        [INST] [Question] """ + data.prompt.strip().lower().replace("</s>", "") + """. 
                        [Answer] """  + data.response.strip().lower().replace("</s>", "") + ". [/INST]"
        data.text = LLaMA_prompt
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
    
    # Put the data into a DataLoader
    train_data = TensorDataset(text, mask, target)
    # train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, batch_size=1)
    
    return train_dataloader


def model_generate(model, tokenizer, device, train_dataloader):
    """
        Iterated through the data in batches, calling the LLM for each batch to extract the outputs.
        From there the outputs are passed through get_probabilities() to extract the probability of 
        an 'on topic' response.
    """
    # Initialise empty array and streamer to automatically show the decoded model output
    on_probabilities = []
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Iterate through the dataloder and feed the prompts into the model
    for step, batch in enumerate(train_dataloader):
        with torch.no_grad():
            b_input_ids = batch[0].to(device)
            b_att_msks = batch[1].to(device)
            output = model.generate(input_ids=b_input_ids, 
                                    attention_mask=b_att_msks, 
                                    pad_token_id=tokenizer.eos_token_id,
                                    do_sample = False, # temperature = 0
                                    return_dict_in_generate=True,
                                    max_new_tokens=1,
                                    streamer=streamer,
                                    output_scores=True, 
                                    use_cache=False)
            # Use lmodel scores to get the probability for 'On' via get_probabilities()
            print(step, " of ", len(train_dataloader), " Target: ", batch[2])
            scores = (output.scores)[0].cpu().numpy()
            for score in scores:
                logits = np.array(score)
                on_probabilities.append(get_probabilities(logits, tokenizer))
    # np.savetxt("/scratches/dialfs/alta/relevance/ns832/results/prediction_scores/LLaMA_predictions.txt", on_probabilities)         
    return on_probabilities


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
    model, tokenizer = initialise_model()
    
    # Load dataset and shuffle data before creating prompts
    text_data, __ , __ = preprocess_data.load_dataset(args, images = False)
    # Shuffling is done here (as well as during create_dataloader()) if only a subsection of the data is used for testing purposes
    np.random.shuffle(text_data)
    text_data = create_prompts(text_data[:4000])
    
    # Run the evaluation and obtain scores for 'On-topic'
    train_dataloader = create_dataloader(tokenizer, text_data, device)
    on_probabilities = model_generate(model, tokenizer, device, train_dataloader)   
    prob_list = np.array(on_probabilities)
 
    # Create Precision-Recall Graph
    targets = np.array([x.target for x in text_data])
    metrics.calculate_metrics(targets, prob_list)
    metrics.calculate_normalised_metrics(targets, prob_list)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.folder_path:
        if args.prompts_path == None: args.prompts_path = str(args.folder_path) + "prompts.txt"
        if args.resps_path == None: args.resps_path = str(args.folder_path) + "responses.txt"
        if args.topics_path == None: args.topics_path = str(args.folder_path) + "topics.txt"
        if args.topic_dist_path == None: args.topic_dist_path = str(args.folder_path) + "topics_dist.txt"
        if args.labels_path == None: args.labels_path = str(args.folder_path) + "targets.txt"
    main(args)