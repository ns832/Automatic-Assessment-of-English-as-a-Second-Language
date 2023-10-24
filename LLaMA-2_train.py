from transformers import LlamaForCausalLM, LlamaTokenizer
import argparse
import torch
import datetime
import numpy as np


parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--prompts_path', type=str, help='Load path of question training data')
parser.add_argument('--responses_path', type=str, help='Load path of answer training data')
parser.add_argument('--prompt_ids_path', type=str, help='Load path of prompt ids')



def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def get_default_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    return device

def initialise_model():
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_id, truncation_side='left', padding_side='right')
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map={"":0})
    return model, tokenizer


def load_dataset(args, device):
    prompt_ids, response, train_data = [], [], []
    with open(args.prompt_ids_path) as f0, open(args.responses_path) as f1, open(args.prompts_path) as f2:
        prompt_ids, responses, prompts = f0.readlines(), f1.readlines(), f2.readlines()
        
        # choose to permute the dataset and concatenate it with the original dataset
        val = int(len(prompt_ids))
        prompt_ids = permute_data(prompt_ids, val, device)
        responses += responses # since we doubled the prompt size
    
            
    targets = torch.tensor([0] * val + [1] * val) # i.e. first half is on-topic, second half is off-topic
    print("Dataset Loaded")
    return zip(prompts[int(prompt_ids)], responses), targets


def permute_data(prompt_ids, val, device): 
    # Dynamic shuffling in order to generate off-topic samples, based on prompt probability dist.
    question_dist_path = '/home/alta/relevance/vr311/data_GKTS4_rnnlm/LINSKevl07/shuffled/'
    unique_prompts_distribution_path = "/scratches/dialfs/alta/ns832/data/train_seen/training/topics_dist.txt" 
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


def create_prompts(zipped_question_answer):
    LLaMA_prompt = 'Given this question and answer pair, return either ”On-topic” if you think the answer matches the question, or ”Off-topic” if you think the answer does not match the question. Question:...... Answe'
    LLaMA_prompt_list = []
    for (question, answer) in zipped_question_answer:
        LLaMA_prompt_list.append(LLaMA_prompt + 'Question:' + question + 'Answer' + answer)
    print(LLaMA_prompt_list[0])
    print(LLaMA_prompt_list[-1])
    
    return LLaMA_prompt_list

def main(args):
    device = get_default_device()
    model, tokenizer = initialise_model()
    zipped_question_answer = load_dataset(args, device)
    LLaMA_prompt_list = create_prompts(zipped_question_answer)
    return


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)