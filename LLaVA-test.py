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

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import Section_D.preprocess_data as preprocess_data, Section_D.load_models as load_models

def get_default_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    return device


def permute_data(prompt_ids, val, device): 
    """
        Takes in the prompt ids and shuffles them to create additional off-topic prompt-response pairs
    """
    # Dynamic shuffling in order to generate off-topic samples, based on prompt probability dist.
    unique_prompts_distribution_path = args.topic_dist_path
    prompt_distribution = np.loadtxt(unique_prompts_distribution_path, dtype=np.int32)
    prompt_distribution = prompt_distribution / np.linalg.norm(prompt_distribution, 1)
    
    number_of_questions = len(prompt_distribution)
    new_prompt_ids = np.random.choice(number_of_questions, val, p=prompt_distribution)
    
    for i in range(val):
        while (new_prompt_ids[i] == prompt_ids[i]):
            new_prompt_ids[i] = np.random.choice(number_of_questions, 1, p=prompt_distribution)
    prompt_ids += list(new_prompt_ids)
    
    print("Data permuted")
    return prompt_ids


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


def get_probabilities(input_ids, output_ids, tokenizer):
    """
        Takes in the output of the model and converts it into a probability for on/off topic
    """
    
    # Score is a tuple consisting of two tensors. Each tensor has n number of rows, where n is the amount of sequences returned
    for seq, score_1, score_2 in zip(output_ids.sequences, output_ids.scores[0], output_ids.scores[1]):
        seq = tokenizer.decode(seq[input_ids.shape[1]:]).strip()
                
        probability = torch.nn.functional.softmax(score_1)
        values = torch.topk(probability, k=2).values
        score = values[0] / (values[0] + values[1])
        # probability = torch.nn.functional.softmax(score_2)
        # print(torch.topk(probability, k=2))
        print(seq)
    
    # On is 1 and off is 0
    if "off" in seq.lower():
        score = 1 - score
    
    return score.cpu()


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
    bert_base_uncased = "prajjwal1/bert-small"

    # Preprocess textual data
    text_data, image_data, topics = preprocess_data.load_dataset(args)
    text_data = preprocess_data.remove_incomplete_data(text_data, image_data)
    
    text_data = preprocess_data.permute_data(text_data[:100], topics, args)
    
    # Preprocess visual data
    image_data = preprocess_data.load_images(image_data, args)   
    text_data, image_list = preprocess_data.encode_dataset(text_data, image_data, bert_base_uncased)
    
    image_paths = [args.images_path + str(x.id.upper() + ".png") for x in image_list]
    LLaMA_prompt_list = create_prompts(text_data)
    assert (len(LLaMA_prompt_list) == len(image_paths) == len(text_data))
    print(len(LLaMA_prompt_list))
    
    # Model
    disable_torch_init()

    # Get model, tokenizer and image processor
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    
    
    prob_list = []
    for inp, image_path in zip(LLaMA_prompt_list, image_paths):

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
            output_ids = model.generate( 
                input_ids,
                images=image_tensor, # optional?, try removing to get text only classification
                num_return_sequences = 1, #
                return_dict_in_generate=True,
                do_sample=True if args.temperature > 0 else False,
                temperature=1,
                max_new_tokens=args.max_new_tokens, # Add max length?
                streamer=streamer,
                use_cache=False, # 
                output_hidden_states = True,
                output_scores=True, #
                no_repeat_ngram_size = 1, # 
                stopping_criteria=[stopping_criteria])
            
        probability = get_probabilities(input_ids, output_ids, tokenizer)
        prob_list.append(probability)

        # If you don't reset conv then it will throw up issues as this template is designed for one image being fed in and then only text conversation after
        conv = conv_templates[conv_mode].copy()
    
    if args.real == False:
        targets = [x.target for x in text_data]
        targets = np.array(targets)
    else:
        targets = np.loadtxt(args.labels_path, dtype=int)
        targets = torch.tensor(targets)
    prob_list = np.array(prob_list)
    for prob, target in zip(prob_list, targets):
        print(prob, target)
    calculate_metrics(targets, prob_list)
        
        



if __name__ == "__main__":
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
    parser.add_argument('--real', type=bool, help='Is this real or shuffled data')
    args = parser.parse_args()
    main(args)