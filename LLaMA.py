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
parser.add_argument('--labels_path', type=str, default=None ,help='Load path to labels')
parser.add_argument('--real', type=bool, help='Is this real or shuffled data')


# Global Variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
args = parser.parse_args()


def initialise_model():
    """
        Instantiates the tokeniser and model
    """
    model_id =  "enoch/llama-7b-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    model = LlamaForCausalLM.from_pretrained(model_id, load_in_8bit=True,)
    return model, tokenizer


def get_targets(args, topics, text_data):
    """
        Creates targets for the two different types of testing; synthetic and real.
        Real simply shuffles the dataset incase a smaller test is desired.
        Synthetic isolates the on-topic answers, calls the permute_data() function and then shuffles.
    """
    # If no targets, assume all responses are on-topic
    if args.labels_path == None:
        text_data = preprocess_data.permute_data(text_data[:1000], topics, args)
        targets = [1] * int(len(text_data) / 2) + [0] * int(len(text_data) / 2)
    else:
        targets = np.loadtxt(args.labels_path, dtype=int)
        targets = targets[:len(text_data)]
        
    # Print prevelance of off-topic responses
    off_targets = [x for (x, y) in zip(text_data, targets) if y == 0]
    print("Dataset size: ", len(text_data) , "Proportions: ", len(off_targets) / len(targets))
    assert (len(targets) == len(text_data))

    # Shuffle the text_data and targets together
    temp = list(zip(text_data, targets))
    np.random.shuffle(temp)
    text_data, targets = zip(*temp)
    targets = torch.tensor(targets)
    
    return text_data, targets


def create_prompts(text_data):
    """
        Creates prompts to feed into the LLM with the prompt and response pairs.
    """
    LLaMA_prompt = 'Given this question and answer pair, return a single-word response of either ”On” or ”Off” depending on if it is on-topic or off-topic. '
    LLaMA_prompt_list = []
    prompts = [x.prompt.strip().lower().replace("</s>", "") for x in text_data]
    responses = [x.response.strip().lower().replace("</s>", "") for x in text_data]
    for prompt, response in zip(prompts, responses):
        LLaMA_prompt_list.append(LLaMA_prompt + 'Question: ' + prompt + '. Answer: ' + response)
    print("Random prompt: ", LLaMA_prompt_list[np.random.randint(0, 1000)])
    return LLaMA_prompt_list


def create_dataloader(tokenizer, prompt_list, device):
    """
        Creates a dataloader with the encoded input ids and attention masks.
        Pad token is set to the eos token to allow for all encodings to be equal length.
    """
    tokenizer.pad_token = tokenizer.eos_token
    encoding = tokenizer(prompt_list, return_tensors="pt", truncation=True, padding="longest", max_length = 256).to(device)
    input_ids = torch.tensor(encoding.input_ids).to(device)
    attention_mask = torch.tensor(encoding.attention_mask).to(device)
    
    train_data = TensorDataset(input_ids, attention_mask)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=10)
    
    return train_dataloader


def model_generate(model, tokenizer, device, train_dataloader):
    """
        Iterated through the data in batches, calling the LLM for each batch to extract the outputs.
        From there the outputs are passed through get_probabilities() to extract the probability of 
        an 'on topic' response.
    """
    on_probabilities = []
    for step, batch in enumerate(train_dataloader):
        print(step, " of ", len(train_dataloader))
        with torch.no_grad():
            b_input_ids = batch[0].to(device)
            b_att_msks = batch[1].to(device)
            output = model.generate(input_ids=b_input_ids, 
                                    attention_mask=b_att_msks, 
                                    return_dict_in_generate=True,
                                    max_new_tokens=1,
                                    output_scores=True, 
                                    use_cache=False)
            generated_ids = output.sequences[0]  # Extract the generated sequence

            # Filter out special tokens
            filtered_ids = [token_id.item() for token_id in generated_ids if token_id not in tokenizer.all_special_ids]

            # Convert numeric values to words using the tokenizer
            decoded_sequence = tokenizer.decode(filtered_ids, skip_special_tokens=True)
            print(decoded_sequence)
            raise
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


def calculate_normalised_metrics(targets_1, y_pred_all_1, r=0.5):
    targets_1 = 1.-targets_1
    y_preds_1 = 1.-y_pred_all_1
    precision_1, recall_1, thresholds_1 = precision_recall_curve(targets_1, y_preds_1)
    
    for i in range(len(precision_1)):
        if precision_1[i] == 0 or recall_1[i] == 0:
            precision_1[i] += 0.01
            recall_1[i] += 0.01
    
    f_score_1_orig = (np.amax((1.+0.5**2) * ((precision_1 * recall_1) / (0.5**2 * precision_1 + recall_1))))
    targets_1, y_preds_1 = torch.tensor(targets_1, device = 'cpu'), torch.tensor(y_preds_1, device = 'cpu')
    p_1_list = precision_1**(-1) - 1
    n_1_list = recall_1**(-1) - 1
    
    # Add in the last threshold that the p-r curve doesn't output doesn't include
    thresholds_1 = np.concatenate([thresholds_1, [y_preds_1.max()]]) 
    f_scores_1 = []
    
    for threshold, p_1, n_1 in zip(thresholds_1, p_1_list, n_1_list):
        tp, fp, tn, fn = 0, 0, 0, 0
        for target, y_pred in zip(targets_1, y_preds_1):
            pred = (y_pred.item() >= threshold)
            if pred == 1 and target == 1: tp += 1
            elif pred == 1 and target == 0: fp += 1
            elif pred == 0 and target == 0: tn += 1
            elif pred == 0 and target == 1: fn += 1
        # Avoid divide by zero error
        if tn + fp == 0:
            fp += 1
            fn += 1 
        # h is calculated from the ground truth positive and negative classes
        h = (tp + fn) / (tn + fp)
        k_1 = (h * (r**(-1) - 1))
        f_scores_1.append((1.+0.5**2) / ((1.+0.5**2) + (0.5**2)* n_1 + p_1 * k_1))
        
    f_score_1 = max(f_scores_1)
    
    print("Normalised F0.5 scores are:", f_score_1)
    print("Original F0.5 scores are:", f_score_1_orig)
    

def calculate_metrics(targets, y_pred_all):
    """
        Calculates the precision and recall from the predictions and targets.
        Creates precision-recall graph and gives f-score
    """
    targets = 1.-targets
    y_pred = 1.-y_pred_all
    targets, y_pred = torch.tensor(targets, device = 'cpu'), torch.tensor(y_pred, device = 'cpu')
    precision, recall, _ = precision_recall_curve(targets, y_pred)
    
    print("Precision:", precision)
    print("Recall:", recall)
    
    # Create precision recall curve
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color='purple')

    # Add axis labels to plot
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    for i in range(len(precision)):
        if precision[i] == 0 or recall[i] == 0:
            precision[i] += 0.1
            recall[i] += 0.1
            
    f_score = np.amax( (1.+0.5**2) * ( (precision * recall) / (0.5**2 * precision + recall) ) )
    print("F0.5 score is:", f_score)

    # Display plot
    fig.show()
    print('/scratches/dialfs/alta/relevance/ns832/results' + '/LLaMA_' + str(f_score) +  '_plot.jpg')
    fig.savefig('/scratches/dialfs/alta/relevance/ns832/results' + '/LLaMA_' + str(f_score) +  '_plot.jpg', bbox_inches='tight', dpi=150)


def main(args):
    # Instantiate model and tokenizer
    model, tokenizer = initialise_model()
    
    # Preprocess data
    text_data, __ , topics = preprocess_data.load_dataset(args, images = False)
    text_data, targets = get_targets(args, topics, text_data)
    LLaMA_prompt_list = create_prompts(text_data)
    
    # Run the evaluation
    train_dataloader = create_dataloader(tokenizer, LLaMA_prompt_list, device)
    on_probabilities = model_generate(model, tokenizer, device, train_dataloader)   
    prob_list = np.array(on_probabilities)
 
    # Create Precision-Recall Graph
    calculate_metrics(targets, prob_list)
    calculate_normalised_metrics(targets, prob_list)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)