import argparse
from sklearn.metrics import precision_recall_curve
import torch
import numpy as np
import matplotlib.pyplot as plt

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
from PIL import Image
from transformers import TextStreamer
import Section_D.preprocess_data as preprocess_data

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
parser.add_argument("--model-base", type=str, default=None)
parser.add_argument("--images_path", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--conv-mode", type=str, default=None)
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--max-new-tokens", type=int, default=512)
parser.add_argument("--load-8bit", action="store_true")
parser.add_argument("--load-4bit", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument('--prompts_path', type=str, help='Load path of question training data')
parser.add_argument('--resps_path', type=str, help='Load path of answer training data')
parser.add_argument('--topics_path', type=str, help='Load path of topics')
parser.add_argument('--topic_dist_path', type=str, help='Load path of topic distribution')
parser.add_argument('--prompt_ids_path', type=str, help='Load path of prompt ids')
parser.add_argument('--image_ids_path', type=str, help='Load path of image ids')
parser.add_argument('--image_prompts_path', type=str, help='Load path of prompts corresponding to image ids')
parser.add_argument('--labels_path', type=str, help='Load path to labels')
parser.add_argument('--real', type=bool, default=False, help='Is this real or shuffled data')

# Global Variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
args = parser.parse_args()


def get_targets(args, topics, text_data):
    """
        Creates targets for the two different types of testing; synthetic and real.
        Real simply shuffles the dataset incase a smaller test is desired, synthetic 
        isolates the on-topic answers, calls the permute_data() function and then shuffles.
    """
    
    targets = np.loadtxt(args.labels_path, dtype=int)
    if args.real == False:
        text_data = [x for (x, y) in zip(text_data, targets) if y == 1]
        text_data = preprocess_data.permute_data(text_data[:500], topics, args)
        targets = [1] * int(len(text_data) / 2) + [0] * int(len(text_data) / 2 )

    # Shuffle the lists together to keep the Q-A pairs and their corresponding targets together
    temp = list(zip(text_data, targets))
    np.random.shuffle(temp)
    text_data, targets = zip(*temp[:1000])
    targets = torch.tensor(targets)
    text_data = list(text_data)
    
    # Print prevelance of off-topic responses
    off_targets = [x for (x, y) in zip(text_data, targets) if y == 0]
    print("Dataset size: ", len(text_data) , "Proportions: ", len(off_targets) / len(targets))
    assert (len(targets) == len(text_data))
    
    return text_data, targets


def create_prompts(text_data):
    """
        Creates prompts to feed into the LLM with the prompt and response pairs
    """
    LLaMA_prompt = 'Given this image and this question and answer pair, return a single-word response of either ”On” or ”Off” depending on if it is on-topic or off-topic. '
    LLaMA_prompt_list = []
    prompts = [x.prompt.strip().lower().replace("</s>", "") for x in text_data]
    responses = [x.response.strip().lower().replace("</s>", "") for x in text_data]
    for prompt, response in zip(prompts, responses):
        LLaMA_prompt_list.append(LLaMA_prompt + 'Question: ' + prompt + ' Answer: ' + response)
    return LLaMA_prompt_list


def eval_model(model, LLaMA_prompt_list, image_paths, image_processor, tokenizer):
    # Initialise Variables
    disable_torch_init()
    on_token = tokenizer("On").input_ids
    off_token = tokenizer("Off").input_ids
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    prob_list = []
    
    i = 0
    for inp, image_path in zip(LLaMA_prompt_list, image_paths):
        print(i, " of ", len(LLaMA_prompt_list))
        i += 1
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        
        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
                   
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        with torch.inference_mode():
            output = model.generate( 
                input_ids,
                images=image_tensor, 
                return_dict_in_generate=True,
                max_new_tokens=1, 
                streamer=streamer,
                use_cache=False, 
                output_scores=True, 
                no_repeat_ngram_size = 1, 
                stopping_criteria=[stopping_criteria])
            
        logits = np.array(output.scores[0].cpu())
        on_prob = get_probabilities(logits, on_token, off_token)    
        prob_list.append(on_prob)    

        # If you don't reset conv then it will throw up issues as this template is designed for one image being fed in and then only text conversation after
        conv = conv_templates[conv_mode].copy()
    return prob_list


def get_probabilities(logits, on_token, off_token):
    """
        Takes in the output of the model and converts it into a probability for on topic
    """
    on_logits = logits[0, on_token[1]]
    off_logits = logits[0, off_token[1]]
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
    print('/scratches/dialfs/alta/relevance/ns832/results' + '/LLaVA_' + str(f_score) +  '_plot.jpg')
    fig.savefig('/scratches/dialfs/alta/relevance/ns832/results' + '/LLaVA_' + str(f_score) +  '_plot.jpg', bbox_inches='tight', dpi=150)


def main(args):
    
    # Preprocess textual data
    text_data, image_data, topics = preprocess_data.load_dataset(args)
    text_data = preprocess_data.remove_incomplete_data(text_data, image_data)
    text_data, targets = get_targets(args, topics, text_data)
    print(len(text_data))

    # Get model, tokenizer and image processor
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    
    # Preprocess visual data
    image_data = preprocess_data.load_images(image_data, args)   
    text_data, image_list = preprocess_data.encode_dataset(tokenizer, text_data, image_data)
    
    image_paths = [args.images_path + str(x.id.upper() + ".png") for x in image_list]
    LLaMA_prompt_list = create_prompts(text_data)
    assert (len(LLaMA_prompt_list) == len(image_paths) == len(text_data))
    
    # Model
    prob_list = eval_model(model, LLaMA_prompt_list, image_paths, image_processor, tokenizer)
    prob_list = np.array(prob_list)
 
    calculate_metrics(targets, prob_list)
    calculate_normalised_metrics(targets, prob_list)
        
        



if __name__ == "__main__":
    main(args)