import torch
import numpy as np
from transformers import BertTokenizer
from torchvision import transforms
import os
from PIL import Image
import re
import random



device = 'cuda' if torch.cuda.is_available() else 'cpu'


class prompt_response():
    def __init__(self, prompt, prompt_id, response, target):
        self.prompt = prompt
        self.response = response
        self.id = prompt_id
        self.target = target
        self.text = None
        self.mask = None
    def add_encodings(self, encoded_text, text_attention_mask):
        self.text = encoded_text
        self.mask = text_attention_mask
        
        
class image():
    def __init__(self, image_id, image_prompt):
        self.prompt = image_prompt
        self.id = image_id
        self.image = None
        self.pixels = None
    def add_image(self, preprocessed_image):
        self.image = preprocessed_image
    def add_encodings(self, pixel_values):
        self.pixels = pixel_values
        

class complete_data():
    def __init__(self, image, prompt_response):
        self.text = prompt_response.text
        self.mask = prompt_response.mask
        self.target = prompt_response.target
        self.pixels = image.pixels
    def add_image_encodings(self, VT_outputs, VT_attention_mask):
        self.image = VT_outputs
        self.image_mask = VT_attention_mask



def set_seed(args):
    """
        Sets the seed to allow reproducibility of results
    """
    
    seed_val = args.seed
    random.seed(seed_val), np.random.seed(seed_val)
    torch.manual_seed(seed_val), torch.cuda.manual_seed_all(seed_val)
    
    return



def find_corresponding_image_id(prompt, image_data):
    
    """" 
    Due to how the prompts are processed certain phrases need to changed.
    
    Hyphens were either removed or replaced with 'to', so all variations are checked for. 
    If there are multiple hypens with a mixture of the two then the prompt is split to the 
    first hyphen and searched for in the hope that it still enough to uniquely identify the prompt.
    
    Phrases involving $ are dealt with depending on the context.
    
    """
    # Remove any apostraphes and change expressions involving dollars
    prompt = prompt.replace("\'", "")
    prompt = prompt.replace('millions of dollars', '$ million')
    prompt = prompt.replace('billions of dollars', '$ billion')
    prompt = prompt.replace("dollars", "$")
    
    # Iterate through the given lists to find a match, returning the id and a status check that can be accessed to see if a prompt was foung
    for image in image_data:
        im_prompt = (image.prompt).replace("dollars", "$")
        if im_prompt == "":
            pass
        elif im_prompt.replace(" - ", " ") == prompt or im_prompt.replace("-", "to") == prompt or im_prompt.replace("-", " ") == prompt:
            return True, image
        # If there are multiple hypens, it's harder. Attempt to split the expression by the hyphen and search
        elif re.split('-', im_prompt)[0] in prompt: 
            return True, image
    # print(prompt)
    return False, 0



def load_images(image_data, args):
    """
        Defines some preprocessing of the images, introducing some randomness to prevent overfitting.
        Then it takes an array of 'ids' (i.e. file names) that it uses to extract the images from and
        append to an output array 
    """
    
    # Define preprocessing
    image_size = (465, 770)
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5), ##
        transforms.RandomVerticalFlip(p=0.5), ##
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), ##
        transforms.RandomRotation(degrees=(30, 70)) ##
    ])
    
    # Iterate through the ids, find the file and preprocess them
    for image in image_data:
        # Check the image path exists (some have scripts but no images)
        image_path = os.path.join(args.images_path, (image.id).upper() + ".png")
        if os.path.isfile(image_path):
            im = Image.open(image_path).convert("RGB")
            
            # Clip the values to the range [0, 1]
            preprocessed_image = preprocess(im).unsqueeze(0) 
            preprocessed_image = torch.clamp(preprocessed_image, 0, 1)
            image.add_image(preprocessed_image)
        else:
            image_data.remove(image)
            
    return image_data



def load_dataset(args):
    """
        Takes in the different file paths and extracts the data from them.
        There are two separate data types and locations; images (with prompts and ids) and 
        the textual prompts, responses, ids and topics.
        
        The image ids and prompts match up, however with the textual data the prompt, reponses and ids match up, 
        but the topics are separate. Creates two new arrays, one for textual data and one for visual.
        
        All textual data is assigned a target of '1', i.e. on-topic, as these all match and will only be permuted later.
    """
     # First load the tokenizer and initialize empty arrays for your encoded inputs and masks    
    with open(args.resps_path) as f0, open(args.prompts_path) as f1, open(args.image_ids_path) as f2, open(args.image_prompts_path) as f3, open(args.prompt_ids_path) as f4, open(args.topics_path) as f5:
        responses, prompts = f0.readlines(), f1.readlines()
        image_ids, image_prompts = f2.readlines(), f3.readlines()
        prompt_ids, topics = f4.readlines(), f5.readlines()
        
        # Stripping the arrays to match the formating 
        # prompt_ids = [x.strip().lower() for x in prompt_ids]
        # prompts = [x.strip().lower() for x in prompts]
        # responses = [x.strip().lower() for x in responses]
        # topics = [x.strip().lower() for x in topics]
        # image_ids = [x.strip().lower() for x in image_ids]
        # image_prompts = [x.strip().lower() for x in image_prompts]
        prompt_ids = [x.strip().lower() for x in prompt_ids[:100]]
        prompts = [x.strip().lower() for x in prompts[:100]]
        responses = [x.strip().lower() for x in responses[:100]]
        topics = [x.strip().lower() for x in topics[:100]]
        image_ids = [x.strip().lower() for x in image_ids[:100]]
        image_prompts = [x.strip().lower() for x in image_prompts[:100]]
        
        # Create two datasets, using the custom classes defined
        print("Dataset Loaded")
        text_data, image_data = [], []
        
        for prompt, prompt_id, response in zip(prompts, prompt_ids, responses):
            text_data.append(prompt_response(prompt, prompt_id, response, 1))
            
        for image_id, image_prompt in zip(image_ids, image_prompts):
            image_data.append(image(image_id, image_prompt))
        
    return text_data, image_data, topics



def remove_incomplete_data(text_data, image_data):
    """
        Some prompts are missing their image counterparts, therefore we remove them here.
        Returns the text_data that has a corresponding image
    """
    for data in text_data:
        if (find_corresponding_image_id(data.prompt, image_data)[0]) == False:
            text_data.remove(data)
            
    return text_data



def permute_data(text_data, topics, args): 
    """
        Takes an existing topic distribution and creates a new list of topics where the probability of a given topic
        is similar to the original.
        Then the code checks that the topic is different to the original topic, and concatenates it to the existing data.
        Sets the new target to 0 instead of 1 to designate it ass off-topic
    """
    # Dynamic shuffling in order to generate off-topic samples, based on prompt probability dist.
    unique_prompts_distribution_path = args.topic_dist_path 
    prompt_distribution = np.loadtxt(unique_prompts_distribution_path, dtype=np.int32)
    prompt_distribution = prompt_distribution / np.linalg.norm(prompt_distribution, 1)
    number_of_questions = len(prompt_distribution)
        
    # Cycle through the new batch, and reassign any responses that have been assigned their original prompts
    val = int(len(text_data))
    new_prompt_ids = np.random.choice(number_of_questions, val, p=prompt_distribution)
    for i in range(val):
        while (new_prompt_ids[i] == text_data[i].id):
            new_prompt_ids[i] = np.random.choice(number_of_questions, 1, p=prompt_distribution)
    
    # Append to the list new prompt_response objects that have the same order of responses as the original text_data, but have shuffled prompts/ids
    for prompt_id, data in zip(new_prompt_ids, text_data):
        text_data.append(prompt_response(topics[int(prompt_id)], prompt_id, data.response, 0))
        
    print("Data Permuted")
    return text_data



def encode_images(image_data, feature_extractor):
    """
        Takes in an array of images and processes them using a pretrained image processor.
        Sets the corresponding class attributes to the new values
    """
    # Iterate through lists, processing the image and returning a pytorch tensor
    for image in image_data:
        if image.image != None:
            inputs = feature_extractor(image.image, return_tensors="pt", do_rescale=True).to(device)
            image.add_encodings((inputs.pixel_values).cpu())
            
    return image_data



def encode_dataset(tokenizer, text_data, image_data):
    """
        Encodes the entire text data by calling the function encode_data().
        Also create targets
    """
    max_prompt_length = 256
    
    # Encode the prompts/responses and save the attention masks, padding applied to the end
    image_list = []
    index_to_remove = []
    for data in text_data:
        # Find the corresponding image so that later they can be concatenated together
        image = find_corresponding_image_id(data.prompt, image_data)[1]
        if image in image_data:
            image_list.append(image)
            text, mask = encode_data(tokenizer, data, max_prompt_length)
            data.add_encodings(text, mask)
        else:
            index_to_remove.append(text_data.index(data))
            
    for index in list(reversed(index_to_remove)):
        text_data.pop(index)
    return text_data, image_list



def encode_data(tokenizer, data, MAX_LEN):
    input_ids, attention_mask = [], []
    input = data.prompt + data.response
    encoding = tokenizer(input, padding="max_length", max_length = MAX_LEN, add_special_tokens=True)
    # input_ids.append(encoding["input_ids"])
    if len(encoding["input_ids"]) > 256:
        input_ids.append(encoding["input_ids"][:256])
        attention_mask.append(encoding["attention_mask"][:256])
    else:
        input_ids.append(encoding["input_ids"])
        attention_mask.append(encoding["attention_mask"])
    return input_ids, attention_mask



def remove_mismatching_prompts(image_list, text_data):
    """
        Takes in an image list and data list and returns an object that combines text, image and targets.
        It checks to ensure that both image and text is complete
    """    
    data_train = []
    temp_im_list, temp_prompt_list, temp_response_list = [], [], []
    temp_targets_list = []
    for image, text in zip(image_list, text_data):
        if image.pixels != None and text.text != None:
            temp_targets_list.append(str(text.target))
            temp_im_list.append(image.id)
            temp_prompt_list.append(text.prompt)
            temp_response_list.append(text.response)
            data_train.append(complete_data(image, text))
            
    return data_train

