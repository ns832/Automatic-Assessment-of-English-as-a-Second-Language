import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import datetime
from transformers import AutoImageProcessor, ViTForImageClassification, ViTImageProcessor
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
from PIL import Image
from transformers import BertTokenizer
import eval_BERT, train_BERT

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--images_path', type=str, help='Load path of image training data')
parser.add_argument('--batch_size', type=int, default=100, help='Specify the test batch size')
parser.add_argument('--prompts_path', type=str, help='Load path to test prompts as text')
parser.add_argument('--responses_path', type=str, help='Load path of answer training data')
parser.add_argument('--prompt_ids_path', type=str, help='Load path of prompt ids')
parser.add_argument('--topics_path', type=str, help='Load path of question training data')
parser.add_argument('--topic_dist_path', type=str, help='Load path of question training data')
parser.add_argument('--image_ids_to_prompts_path', type=str, help='Load path of question training data')
parser.add_argument('--model_path', type=str, help='Load path to trained model')
parser.add_argument('--predictions_save_path', type=str, help="Where to save predicted values")
parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')



# Global Variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
args = parser.parse_args()


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def load_images(id_list):
    image_size = (465, 770)
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5), ##
        transforms.RandomVerticalFlip(p=0.5), ##
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), ##
        transforms.RandomRotation(degrees=(30, 70)) ##
    ])
    images = []
    for file in os.listdir(args.images_path):
        if file.endswith(".png"): 
            image_path = os.path.join(args.images_path, file)
            image = Image.open(image_path).convert("RGB")
            preprocessed_image = preprocess(image).unsqueeze(0)
            # Clip the values to the range [0, 1]
            preprocessed_image = torch.clamp(preprocessed_image, 0, 1)
            images.append(preprocessed_image)
    return images

def load_model():
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    model.to(device)
    return model, image_processor

def encode_images(images, image_processor):
    inputs_list = []
    pixel_values = []
    for image in images:
        inputs = image_processor(image, return_tensors="pt", do_rescale=True).to(device)
        inputs_list.append(inputs)
        pixel_values.append(inputs.pixel_values.squeeze().tolist())
    pixel_values = torch.tensor(pixel_values)
    pixel_values = pixel_values.to(device)
    return inputs_list, pixel_values

def create_id_dictionary(args):
    id_dict = dict()
    with open(args.image_ids_to_prompts_path) as f:
        image_ids_to_prompts = f.readlines()
        for line in image_ids_to_prompts:
            image_id = line.split()[0]
            prompt =  line.replace(image_id, "")
            prompt = prompt.replace("\t", "")
            prompt = prompt.replace("\n", "")
            if "/" in image_id: # Extract first id associated with the prompt
                image_id = image_id.split("/")[0]
            id_dict[prompt] = image_id
    id_list = []
    prompts = open(args.prompts_path).readlines()
    for prompt in prompts:
        prompt = prompt.replace("\n", "")
        if prompt in id_dict.keys():
            id_list.append(id_dict[prompt])
        else:
            print(prompt)
    raise
    return id_dict

def create_dataset(encoded_prompts, encoded_responses, VT_outputs, VT_attention_mask, prompt_attention_masks, response_attention_masks, targets):
    encoded_prompts = encoded_prompts.squeeze(1)
    encoded_responses = encoded_responses.squeeze(1)
    VT_outputs = VT_outputs.squeeze(1)
    VT_attention_mask = VT_attention_mask.squeeze(1)
    prompt_attention_masks = prompt_attention_masks.squeeze(1)
    response_attention_masks = response_attention_masks.squeeze(1)
    
    print(encoded_prompts.shape), print(encoded_responses.shape), print(VT_outputs.shape)
    input = torch.cat((encoded_prompts.cpu(), encoded_responses.cpu(), VT_outputs.cpu()),1)
    input_mask = torch.cat((prompt_attention_masks, response_attention_masks, VT_attention_mask),1)
    input = input.squeeze(1)
    input_mask = input_mask.squeeze(1)
    print(input.size(), input_mask.size(), targets.size())
    
    train_data = TensorDataset(input, input_mask, targets)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
    return train_dataloader

def main():
    # Match the images to their prompts
    id_dict = create_id_dictionary(args)
    
    # Obtain BERT outputs
    bert_base_uncased = "prajjwal1/bert-small"
    train_BERT.set_seed(args)
    encoded_prompts, encoded_responses, prompt_attention_masks, response_attention_mask, targets, id_list = train_BERT.load_dataset(args, bert_base_uncased, id_dict)
    
    # Obtain VT outputs (not used in training, only evaluation)
    images = load_images(id_list)
    model, image_processor = load_model()
    inputs_list, pixel_values = encode_images(images, image_processor)
    with torch.no_grad():
        VT_outputs = model(pixel_values) 
        VT_outputs = VT_outputs.logits    
    
    # outputs are then concatenated together, with attention mask set to zero for the visual elements
    VT_attention_mask = np.zeros_like(VT_outputs.cpu())
    VT_attention_mask = torch.tensor(VT_attention_mask)
    train_dataloader = create_dataset(encoded_prompts, encoded_responses, VT_outputs, VT_attention_mask, prompt_attention_masks, response_attention_mask, targets)


if __name__ == '__main__':
    main() 