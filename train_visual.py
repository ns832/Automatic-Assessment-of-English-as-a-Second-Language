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
import eval_BERT

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--images_path', type=str, help='Load path of image training data')
parser.add_argument('--batch_size', type=int, default=100, help='Specify the test batch size')
parser.add_argument('--prompts_path', type=str, help='Load path to test prompts as text')
parser.add_argument('--resps_path', type=str, help='Load path to test responses as text')
parser.add_argument('--labels_path', type=str, help='Load path to labels')
parser.add_argument('--model_path', type=str, help='Load path to trained model')
parser.add_argument('--predictions_save_path', type=str, help="Where to save predicted values")


# Global Variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
args = parser.parse_args()

class arguments:
    def __init__(self, batch_size, prompts_path, resps_path, labels_path, model_path, predictions_save_path):
        self.batch_size = batch_size
        self.prompts_path = prompts_path
        self.resps_path = resps_path
        self.labels_path = labels_path
        self.model_path = model_path
        self.predictions_save_path = predictions_save_path

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def create_dataset(train_dataset_images):
    train_dataloader_images = DataLoader(train_dataset_images, batch_size=args.batch_size)
    return train_dataloader_images


def load_images():
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
        # Folder contains not just .png but also .htm
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

def main():
    # Obtain VT outputs (not used in training, only evaluation)
    images = load_images()
    model, image_processor = load_model()
    inputs_list, pixel_values = encode_images(images, image_processor)
    with torch.no_grad():
        VT_outputs = model(pixel_values) 
        VT_outputs = VT_outputs.logits
        print(VT_outputs)
    # outputs are then concatenated with the outputs of the BERT system only during the evaluation
    args_model = arguments(args.batch_size, args.prompts_path, args.resps_path, args.labels_path, args.model_path, args.predictions_save_path)
    BERT_outputs = eval_BERT.main(args_model)
    BERT_outputs = torch.tensor(BERT_outputs)

    print(VT_outputs)
    print(BERT_outputs)
    concatenated_outputs = VT_outputs + BERT_outputs
    return VT_outputs

if __name__ == '__main__':
    main() 