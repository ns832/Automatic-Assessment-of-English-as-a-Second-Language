import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import numpy as np
from sklearn.metrics import precision_recall_curve
from transformers import LlamaForCausalLM, LlamaTokenizer
import matplotlib.pyplot as plt
import Section_D.preprocess_data as preprocess_data


parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--prompts_path', type=str, help='Load path of question training data')
parser.add_argument('--resps_path', type=str, help='Load path of answer training data')
parser.add_argument('--prompt_ids_path', type=str, help='Load path of prompt ids')
parser.add_argument('--topic_dist_path', type=str, help='Load path of prompt distribution')
parser.add_argument('--topics_path', type=str, help='Load path of topics')
parser.add_argument('--labels_path', type=str, help='Load path to labels')
parser.add_argument('--real', type=bool, help='Is this real or shuffled data')


# Global Variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
args = parser.parse_args()


def initialise_model(device):
    model_id =  "enoch/llama-7b-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    model = LlamaForCausalLM.from_pretrained(model_id, load_in_8bit=True,)
    # model.to(device)
    return model, tokenizer


def create_prompts(text_data):
    """
        Creates prompts to feed into the LLM with the prompt and response pairs
    """
    LLaMA_prompt = 'Given this image and this question and answer pair, return either ”On” or ”Off” depending on if it is on-topic or off-topic. '
    LLaMA_prompt_list = []
    prompts = [x.prompt.strip().lower().replace("</s>", "") for x in text_data]
    responses = [x.response.strip().lower().replace("</s>", "") for x in text_data]
    for prompt, response in zip(prompts, responses):
        LLaMA_prompt_list.append(LLaMA_prompt + 'Question: ' + prompt + ' Answer: ' + response)
    return LLaMA_prompt_list


def create_dataloader(tokenizer, prompt_list, device):
    """
        Creates a dataloader with the targets, encodings, input ids, attention masks and labels.
    """
    # targets = tokenizer(targets, truncation=True, padding="longest")
    tokenizer.pad_token = tokenizer.eos_token
    encoding = tokenizer(prompt_list, return_tensors="pt", truncation=True, padding="longest", max_length = 256).to(device)
    input_ids = torch.tensor(encoding.input_ids).to(device)
    attention_mask = torch.tensor(encoding.attention_mask).to(device)
    # labels = torch.tensor(targets.input_ids).to(device)
    # train_data = TensorDataset(input_ids, attention_mask, labels)
    train_data = TensorDataset(input_ids, attention_mask)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=1)
    
    return train_dataloader


def get_output(model, tokenizer, device, train_dataloader):
    """
        Iterated through the data in batches, calling the LLM for each batch to extract the outputs.
        From there the outputs are passed through get_probabilities() to extract the probability of 
        an 'on topic' response.
    """
    on_probabilities = []
    for step, batch in enumerate(train_dataloader):
        with torch.no_grad():
            b_input_ids = batch[0].to(device)
            b_att_msks = batch[1].to(device)
            # b_output_ids = batch[2].to(device)
            output = model.generate(input_ids=b_input_ids, 
                                    attention_mask=b_att_msks, 
                                    return_dict_in_generate=True,
                                    # labels=b_output_ids,
                                    max_new_tokens=1,
                                    output_scores=True, 
                                    use_cache=False)
            logits = np.array(output.scores[0].cpu())
            on_probabilities.append(get_probabilities(logits, tokenizer))
         
    return on_probabilities


def get_probabilities(logits, tokenizer):
    """
        Takes in the output of the model and converts it into a probability for on topic
    """
    print("On token", tokenizer("On").input_ids)
    print("Off token", tokenizer("Off").input_ids)
    on_token = tokenizer("On").input_ids
    off_token = tokenizer("Off").input_ids
    
    on_logits = logits[0, on_token[1]]
    off_logits = logits[0, off_token[1]]
    score = on_logits / (on_logits + off_logits)
    # score = np.exp(on_logits) / (np.exp(on_logits) + np.exp(off_logits))
    print("Logits:", on_logits, off_logits, "Score:", score)
    
    return score


def get_answers(model, tokenizer, device, prompt_list, targets):
    """
        Calls the create_dataloader() and get_output() functions to pass the tokenised inputs through 
        the LLM and get the output probabilities for 'on topic'
    """
    train_dataloader = create_dataloader(tokenizer, prompt_list, device)
    on_probabilities = get_output(model, tokenizer, device, train_dataloader)   
    return on_probabilities


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
    print('/scratches/dialfs/alta/relevance/ns832/results' + '/LLaMA_' + str(f_score) +  '_plot.jpg')
    # fig.savefig('/scratches/dialfs/alta/relevance/ns832/results' + '/LLaMA_' + str(f_score) +  '_plot.jpg', bbox_inches='tight', dpi=150)


def get_targets(args, topics, text_data):
    """
        Creates targets for the two different types of testing; synthetic and real.
        Real simply shuffles the dataset incase a smaller test is desired.
        Synthetic isolates the on-topic answers, calls the permute_data() function and then shuffles.
    """
    # if args.real == True:
    #     targets = np.loadtxt(args.labels_path, dtype=int)
    # if args.real == False:
        # text_data = [x for (x, y) in zip(text_data, targets) if y == 1]
    text_data = preprocess_data.permute_data(text_data[:500], topics, args)
    targets = [1] * int(len(text_data) / 2) + [0] * int(len(text_data) / 2)
    assert (len(targets) == len(text_data))
    temp = list(zip(text_data, targets))
    np.random.shuffle(temp)
    text_data, targets = zip(*temp)
    targets = torch.tensor(targets)
    return text_data, targets

def main(args):
    model, tokenizer = initialise_model(device)
    
    # Preprocess data
    text_data, __ , topics = preprocess_data.load_dataset(args, images = False)
    text_data, targets = get_targets(args, topics, text_data)
    print(len(text_data))
    LLaMA_prompt_list = create_prompts(text_data)
    
    # Run the evaluation
    prob_list = get_answers(model, tokenizer, device, LLaMA_prompt_list, targets)
    prob_list = np.array(prob_list)
 
    calculate_metrics(targets, prob_list)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)