import argparse
from sklearn.metrics import precision_recall_curve
import torch
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
from PIL import Image
from transformers import TextStreamer
import Section_D.preprocess_data as preprocess_data
import Section_D.metrics as metrics
import helper_scripts.create_composite_image

# An optional folder_path can be supplied if multiple files are in the same directory - any file_path not given is then assumed to be in this folder
parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', type=str, default=None, help='Load path of the folder containing prompts, responses etc.')
parser.add_argument('--prompts_path', type=str, default=None, help='Load path of question training data')
parser.add_argument('--resps_path', type=str, default=None, help='Load path of answer training data')
parser.add_argument('--prompt_ids_path', type=str, default=None, help='Load path of prompt ids')
parser.add_argument('--topic_dist_path', type=str, default=None, help='Load path of prompt distribution')
parser.add_argument('--topics_path', type=str, default=None, help='Load path of topics')
parser.add_argument('--labels_path', type=str, default=None ,help='Load path to labels')

# An optional image folder_path can be supplied if multiple image files are in the same directory - any file_path not given is then assumed to be in this folder
parser.add_argument('--folder_path_images', type=str, default=None, help='Optional path for folder containing image data.')
parser.add_argument("--images_path", type=str, default=None)
parser.add_argument('--image_ids_path', type=str, default=None, help='Load path of image ids')
parser.add_argument('--image_prompts_path', type=str, default=None, help='Load path of prompts corresponding to image ids')

parser.add_argument('--real', type=bool, default=None, help='Is this real or shuffled data')
parser.add_argument('--overlay', type=bool, default=None, help='Is the image to be used an amalgamation of all of them')

# parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
parser.add_argument("--model-base", type=str, default=None)
parser.add_argument("--conv-mode", type=str, default=None)
parser.add_argument("--load-8bit", action="store_true")
parser.add_argument("--load-4bit", action="store_true")

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
    
    # If the data is not real, all the targets are on-topic therefore the dataset needs to be permuted
    if not args.labels_path:
        print("No labels detected")
        text_data = preprocess_data.permute_data(text_data[:500], topics, args)
        
    np.random.shuffle(text_data)
    text_data = text_data[:1000]  
    off_targets = [x for x in text_data if x.target == 0]
    print("Dataset size: ", len(text_data) , "Proportions: ", len(off_targets) / len(text_data))
    print(len(text_data))  
    return text_data


def create_prompts(text_data):
    """
        Creates prompts to feed into the LLM with the prompt and response pairs.
        Prompts are formulated using the suggested [INST] [/INST] format suggested by Mistral.
    """    
    # Iterates through the text_data list and creates prompts to feed into the model
    for data in text_data:
        if args.images_path and args.overlay == False: LLaMA_prompt = 'Given this image and this question and answer pair, return a single-word response of either ”On” or ”Off” depending on if it is on-topic or off-topic. '
        else: LLaMA_prompt = 'Ignoring the image. Given this question and answer pair, return a single-word response of either ”On” or ”Off” depending on if it is on-topic or off-topic. '
        LLaMA_prompt = LLaMA_prompt + 'Question: ' + data.prompt + ' Answer: ' + data.response
        data.text = LLaMA_prompt
    # Print a random prompt to check the format of the question is as desired
    print("Random prompt: ", text_data[np.random.randint(0, len(text_data))].text)
    return text_data


def eval_model(model, text_data, image_paths, image_processor, tokenizer):
    """
        Evaluates the prompts-response-image pairs and outputs a probability list.
    """
    # Initialise Variables
    disable_torch_init()
    on_token = tokenizer("On").input_ids
    off_token = tokenizer("Off").input_ids
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    prob_list = []
    LLaMA_prompt_list = [x.text for x in text_data]
    
    i = 0
    for inp, image_path in zip(LLaMA_prompt_list, image_paths):
        print(i, " of ", len(LLaMA_prompt_list))
        i += 1
        if image_path.startswith('http://') or image_path.startswith('https://'):
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
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
                # do_sample = False, temperature = 0
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


def main(args):
    # Preprocess textual data
    if args.images_path: text_data, image_data, topics = preprocess_data.load_dataset(args)
    else: text_data, image_data, topics = preprocess_data.load_dataset(args, images=False)
    text_data = get_targets(args, topics, text_data[:1000])
    if args.images_path: text_data = preprocess_data.remove_incomplete_data(text_data, image_data)

    # Get model, tokenizer and image processor
    model_name = "liuhaotian/llava-v1.5-7b"
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_name, args.model_base, model_name, args.load_8bit, args.load_4bit, device=device)
    
    # Preprocess visual data
    if args.images_path:
        image_data = preprocess_data.load_images(image_data, args)             
        text_data, image_list = preprocess_data.encode_dataset(tokenizer, text_data, image_data)
        
        # If a composite of all images is to be used instead, set all to overlay.jpg 
        if args.overlay == True:
            print("Assigning all images to the composite")
            composite_path = "/scratches/dialfs/alta/relevance/ns832/data/visual_and_text/BLXXXgrp24_CDE.rnnlm/complete_data/overlay.jpg"
            image_paths = [composite_path for x in image_list]
        else: image_paths = [args.images_path + str(x.id.upper() + ".png") for x in image_list]
        assert (len(image_paths) == len(text_data))
    else:
        # If a composite of all images is to be used instead, call helper script  
        if args.overlay == True:
            print("Assigning all images to the composite")
            composite_path = "/scratches/dialfs/alta/relevance/ns832/data/visual_and_text/BLXXXgrp24_CDE.rnnlm/complete_data/overlay.jpg"
            image_paths = [composite_path for __ in image_list]
        else:
            print("Assigning all images to a blank one")
            file = 'https://upload.wikimedia.org/wikipedia/commons/a/a7/Blank_image.jpg'
            image_paths = [file for __ in text_data]
    text_data = create_prompts(text_data)

    # Model
    prob_list = eval_model(model, text_data, image_paths, image_processor, tokenizer)
    prob_list = np.array(prob_list)
 
    targets = np.array([x.target for x in text_data])
    metrics.calculate_metrics(targets, prob_list)
    metrics.calculate_normalised_metrics(targets, prob_list)
        

if __name__ == "__main__":
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
    if args.images_path == None: print("No images detected, text-only version chosen")
    main(args)
    