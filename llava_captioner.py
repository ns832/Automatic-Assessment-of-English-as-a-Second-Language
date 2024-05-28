import argparse
import torch
import os

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
from PIL import Image
from transformers import TextStreamer

# An optional folder_path can be supplied if multiple files are in the same directory - any file_path not given is then assumed to be in this folder
parser = argparse.ArgumentParser()

parser.add_argument("--images_path", type=str, default=None)

# parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
parser.add_argument("--model-base", type=str, default=None)
parser.add_argument("--conv-mode", type=str, default=None)
parser.add_argument("--load-8bit", action="store_true")
parser.add_argument("--load-4bit", action="store_true")


# Global Variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
args = parser.parse_args()


class CustomStreamer():

    def __init__(self, tokenizer: "AutoTokenizer", skip_prompt: bool = False, **decode_kwargs):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.decode_kwargs = decode_kwargs

        # variables used in the streaming process
        self.token_cache = []
        self.print_len = 0
        self.next_tokens_are_prompt = True
        self.decoded_text = []

    def put(self, value):
        """
        Receives tokens, decodes them, and prints them to stdout as soon as they form entire words.
        """
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        # Add the new token to the cache and decodes the entire thing.
        self.token_cache.extend(value.tolist())
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
        
        # After the symbol for a new line, we flush the cache.
        if text.endswith("\n"):
            printable_text = text[self.print_len :]
            self.token_cache = self.token_cache[:-1]
            # self.token_cache = []
            self.print_len = 0
            
        # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
        # which may change with the subsequent token -- there are probably smarter ways to do this!)
        else:
            printable_text = text[self.print_len : text.rfind(" ") + 1]
            self.print_len += len(printable_text)

        self.on_finalized_text(printable_text)

    def end(self):
        """Flushes any remaining cache and prints a newline to stdout."""
        # Flush the cache, if it exists
        if len(self.token_cache) > 0:
            text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
            self.decoded_text.append(text)
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        else:
            printable_text = ""

        self.next_tokens_are_prompt = True
        self.on_finalized_text(printable_text, stream_end=True)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Prints the new text to stdout. If the stream is ending, also prints a newline."""
        # print(text, flush=True, end="" if not stream_end else None)
        pass

def eval_model(model, image_paths, image_processor, tokenizer):
    """
        Evaluates the prompts-response-image pairs and outputs a probability list.
    """
    # Initialise Variables
    disable_torch_init()
    
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    captions = []

    for image_path in image_paths:
        inp = "Give a moderately detailed caption for this image."

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
        streamer = CustomStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output = model.generate( 
                input_ids,
                images=image_tensor, 
                max_new_tokens=200, 
                streamer=streamer,
                use_cache=False, 
                stopping_criteria=[stopping_criteria])
            print(streamer.decoded_text)
            captions.append(streamer.decoded_text[0])
        # If you don't reset conv then it will throw up issues as this template is designed for one image being fed in and then only text conversation after
        conv = conv_templates[conv_mode].copy()
        
    return captions


def main(args):
        
    # Get model, tokenizer and image processor
    model_name = "liuhaotian/llava-v1.5-7b"
    tokenizer, model, image_processor, __ = load_pretrained_model(model_name, args.model_base, model_name, args.load_8bit, args.load_4bit, device=device)
    
    # Preprocess visual data
    image_paths = os.listdir(args.images_path)
    full_image_paths = [args.images_path + path for path in image_paths if '._' not in path]
    full_image_paths = [path for path in full_image_paths if path.endswith('.png')]
    print(full_image_paths)
    
    # Model
    captions = eval_model(model, full_image_paths, image_processor, tokenizer)
    print(captions)
    
if __name__ == "__main__":
    main(args)
    