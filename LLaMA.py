import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, TextStreamer
import Section_D.preprocess_data as preprocess_data
import Section_D.metrics as metrics


parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--folder_path', type=str, help='Load path of the folder containing prompts, responses etc.')
parser.add_argument('--prompts_path', type=str, help='Load path of question training data')
parser.add_argument('--resps_path', type=str, help='Load path of answer training data')
parser.add_argument('--prompt_ids_path', type=str, help='Load path of prompt ids')
parser.add_argument('--topic_dist_path', type=str, help='Load path of prompt distribution')
parser.add_argument('--topics_path', type=str, help='Load path of topics')
parser.add_argument('--labels_path', type=str, default=None ,help='Load path to labels')


# Global Variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
args = parser.parse_args()




def initialise_model(model_name = "mistralai/Mistral-7B-v0.1"):
    """
        Instantiates the tokeniser and model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name,add_special_tokens=True)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
    )  
    
    # model.to(device)
    return model, tokenizer


def get_targets(args, topics, text_data):
    """
        Creates targets for the two different types of testing; synthetic and real.
        Real simply shuffles the dataset incase a smaller test is desired.
        If no labels are passed, assumed all are on-topic and treated as such.
        Synthetic isolates the on-topic answers, calls the permute_data() function and then shuffles.
    """
    if args.labels_path == None:
        print("No labels identified, assumed to be all on-topic")
        for data in text_data:
            data.target = 1
        text_data = preprocess_data.permute_data(text_data[:1000], topics, args)
    else:
        targets = np.loadtxt(args.labels_path, dtype=int)
        for data, target in zip(text_data, targets):
            data.target = target
        
    np.random.shuffle(text_data)
    text_data = text_data[:10]
    off_targets = [x for x in text_data if x.target == 0]
    print("Dataset size: ", len(text_data) , "Proportions: ", len(off_targets) / len(text_data))
    
    return text_data


def create_prompts(text_data):
    """
        Creates prompts to feed into the LLM with the prompt and response pairs.
    """    
    LLaMA_prompt_list = []
    prompts = [x.prompt.strip().lower().replace("</s>", "") for x in text_data]
    responses = [x.response.strip().lower().replace("</s>", "") for x in text_data]
    for prompt, response in zip(prompts, responses):
        LLaMA_prompt = """<s>[INST] Respond to the prompts with only 'On' or 'Off' depending on if the answer is relevant to the question asked. For instance:
                        [Question] this chart shows the number of positive and negative responses given in a survey concerning customer satisfaction with a hotel look at the information and talk about the results of the customer survey.
                        [Answer] this charts present results of survey in hotel most negative response was given about the value for money it was about about nine hundred person and the best results get response for question about attitude of staff it was at about nine hundred and half percent and at about half of respondents talk that parking is positive and at about. 
                        Would be converted to:[/INST] On </s>
                        [INST] [Question] """ + prompt + """. 
                        [Answer] """  + response + ". [/INST]"
        LLaMA_prompt_list.append(LLaMA_prompt)
    print("Random prompt: ", LLaMA_prompt_list[np.random.randint(0, len(LLaMA_prompt_list))])
    return LLaMA_prompt_list


def create_dataloader(tokenizer, prompt_list, device):
    """
        Creates a dataloader with the encoded input ids and attention masks.
        Pad token is set to the eos token to allow for all encodings to be equal length.
    """
    attention_mask, encoding = [], []
    for prompt in prompt_list:
        encoded_prompt = tokenizer(prompt, padding='max_length', truncation=True, max_length=512, add_special_tokens=False)
        # prompt.add_encodings(encoded_prompt["input_ids"], encoded_prompt["attention_mask"])
        encoding.append(encoded_prompt["input_ids"])
        attention_mask.append(encoded_prompt["attention_mask"])
    input_ids = torch.tensor(encoding).squeeze().to(device)
    attention_mask = torch.tensor(attention_mask).squeeze().to(device)
    print(input_ids.shape)
    
    train_data = TensorDataset(input_ids, attention_mask)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=1)
    
    return train_dataloader


def model_generate(model, tokenizer, device, train_dataloader, text_data):
    """
        Iterated through the data in batches, calling the LLM for each batch to extract the outputs.
        From there the outputs are passed through get_probabilities() to extract the probability of 
        an 'on topic' response.
    """
    on_probabilities = []
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    for step, batch in enumerate(train_dataloader):
        print(step, " of ", len(train_dataloader), " Target: ", text_data[step].target)
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
            
            scores = (output.scores)[0].cpu().numpy()
            for score in scores:
                logits = np.array(score)
                on_probabilities.append(get_probabilities(logits, tokenizer))
         
    return on_probabilities


def get_probabilities(logits, tokenizer):
    """
        Takes in the output of the model and converts it into a probability for on topic.
        Softmax is applied to convert the logits into probabilities.
    """
    on_token = tokenizer("On").input_ids
    off_token = tokenizer("Off").input_ids
    
    on_logits = logits[on_token[1]] 
    off_logits = logits[off_token[1]] 
    score = np.exp(on_logits) / (np.exp(on_logits) + np.exp(off_logits))
    print("Logits:", on_logits, off_logits, "Score:", score)
    
    return score


def main(args):
    # Instantiate model and tokenizer
    model, tokenizer = initialise_model()
    
    # Preprocess data
    text_data, __ , topics = preprocess_data.load_dataset(args, images = False)
    text_data = get_targets(args, topics, text_data)
    LLaMA_prompt_list = create_prompts(text_data)
    
    # Run the evaluation
    train_dataloader = create_dataloader(tokenizer, LLaMA_prompt_list, device)
    on_probabilities = model_generate(model, tokenizer, device, train_dataloader, text_data)   
    prob_list = np.array(on_probabilities)
 
    # Create Precision-Recall Graph
    targets = np.array([x.target for x in text_data])
    metrics.calculate_metrics(targets, prob_list)
    metrics.calculate_normalised_metrics(targets, prob_list)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.folder_path:
        args.prompts_path = str(args.folder_path) + "prompts.txt"
        args.resps_path = str(args.folder_path) + "responses.txt"
        args.prompt_ids_path = str(args.folder_path) + "prompt_ids.txt"
        args.topics_path = str(args.folder_path) + "topics.txt"
        args.topic_dist_path = str(args.folder_path) + "topics_dist.txt"
        args.labels_path = str(args.folder_path) + "targets.txt"
    main(args)