import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--prompts_path', type=str, help='Load path to test prompts as text')
parser.add_argument('--responses_path', type=str, help='Load path to test responses as text')
parser.add_argument('--prompt_ids_path', type=str, help='Load path of prompt ids')
parser.add_argument('--sections_path', type=str, help='Load path to labels')
parser.add_argument('--targets_path', type=str, help='Load path to targets')
parser.add_argument('--save_path', type=str, help='Path where the data will be saved')
args = parser.parse_args()

print("Isolating section C")
with open(args.prompt_ids_path) as f0, open(args.responses_path) as f1, open(args.sections_path) as f2, open(args.prompts_path) as f3:
    responses, prompts, prompt_ids = [], [], []
    question_ids, responses, sections, questions = f0.readlines(), f1.readlines(), f2.readlines(), f3.readlines()
    for question_id, response, section, question in zip(question_ids, responses, sections, questions):
        if "C" in section:
            print("Section C found")
            print("Adding responses, prompts, prompt_ids and targets")
            responses.append(response)
            prompts.append(question)
            prompt_ids.append(question_id)
       
    save_path_prompts = args.save_path + "prompts.txt"
    save_path_prompt_ids = args.save_path + "prompt_ids.txt"
    save_path_responses = args.save_path + "responses.txt"
    
    out_prompts = open(save_path_prompts, 'w')
    for prompt in prompts:
        out_prompts.write(prompt)
    out_prompts.close()
    out_responses = open(save_path_responses, 'w')
    for response in responses:
        out_responses.write(response )
    out_responses.close()
    out_prompt_ids = open(save_path_prompt_ids, 'w')
    for prompt_id in prompt_ids:
        out_prompt_ids.write(prompt_id)
    out_prompt_ids.close()

with open(args.targets_path) as f0, open(args.sections_path) as f1:
    target_list = []
    targets, sections = f0.readlines(), f1.readlines()
    for target, section in zip(targets, sections):
        if "C" in section:
            print("Section C found")
            target_list.append(target)
            
    save_path_targets = args.save_path + "targets.txt"
    
    out_targets = open(save_path_targets, 'w')
    for target in target_list:
        out_targets.write(target)
    out_targets.close()