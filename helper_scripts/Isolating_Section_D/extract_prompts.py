from bs4 import BeautifulSoup
import argparse
import os
from num2words import num2words
import re

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--search_path', type=str, help='Load path to test prompts as text')
parser.add_argument('--save_path', type=str, help='Path where the data will be saved')
args = parser.parse_args()

prompts, ids = [], []
for file in os.listdir(args.search_path):
    if file.endswith(".htm"): 
        full_path = os.path.join(args.search_path, file)
        html = open(full_path, encoding='latin-1')
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text()
        head, sep, tail = text.partition('Part Four')
        prompt, sep, tail = tail.partition('Part Five')
        if prompt != "":
            prompt = prompt.replace(prompt.split()[0],"")
            prompt = prompt.replace(prompt.split()[0],"")
            prompt = prompt.replace("\'", "")
            prompt = prompt.replace(".", "")
            prompt = prompt.replace("-", " - ")
            prompt = (prompt.replace('\n', '')).strip()
        result = []
        prompt = re.split(' |\, |\(|\)', prompt)
        for word in prompt:
            if any(chr.isdigit() for chr in word):
                if int(word) < 2000: # Assumes all dates are either 2000s or 1900s
                    string = "nineteen " + num2words(int(word) - 1900)
                    result.append(string)
                else:
                    result.append(num2words(word).replace(' and', ''))
            else:
                result.append(word)
        prompt = ' '.join(result)
        prompt = ' '.join(prompt.split())
        prompts.append(prompt.lower())
        ids.append(file.replace('.htm', ''))
        print(file, prompt.lower())
            
save_path_image_questions = args.save_path + "image_questions.txt"
save_path_image_id = args.save_path + "image_ids.txt"

out_prompts = open(save_path_image_questions, 'w')
for prompt in prompts:
    out_prompts.write(prompt + '\n')
out_prompts.close()

out_ids = open(save_path_image_id, 'w')
for id in ids:
    out_ids.write(id + '\n')
out_ids.close()