import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--prompts_path', type=str, help='Load path to test prompts as text')
parser.add_argument('--resps_path', type=str, help='Load path to test responses as text')
parser.add_argument('--prompt_ids_path', type=str, help='Load path of prompt ids')
parser.add_argument('--topics_path', type=str, help='Load path of question training data')
parser.add_argument('--topic_dist_path', type=str, help='Load path of question training data')
parser.add_argument('--save_path', type=str, help='Path where the data will be saved')
args = parser.parse_args()


def permute_data(prompt_ids, val, topics, args): 
    # Dynamic shuffling in order to generate off-topic samples, based on prompt probability dist.
    unique_prompts_distribution_path = args.topic_dist_path 
    prompt_distribution = np.loadtxt(unique_prompts_distribution_path, dtype=np.int32)
    prompt_distribution = prompt_distribution / np.linalg.norm(prompt_distribution, 1)
    number_of_questions = len(prompt_distribution)
        
    # Cycle through the new batch, and reassign any responses that have been assigned their original prompts
    new_prompt_ids = np.random.choice(number_of_questions, val, p=prompt_distribution)
    for i in range(val):
        while (new_prompt_ids[i] == prompt_ids[i]):
            new_prompt_ids[i] = np.random.choice(number_of_questions, 1, p=prompt_distribution)
    prompt_ids += list(new_prompt_ids)
    new_prompt_list = []
    for prompt_id in prompt_ids:
        new_prompt_list.append(topics[int(prompt_id)])
    return new_prompt_list

def main(args):
    with open(args.prompt_ids_path) as f0, open(args.resps_path) as f1, open(args.topics_path) as f3:
        question_ids, responses, topics = f0.readlines(), f1.readlines(), f3.readlines()
        val = int(len(question_ids))
        shuffled_prompts = permute_data(question_ids, val, topics, args)
        shuffled_responses = responses + responses # since we doubled the prompt size

    save_path_targets = args.save_path + "new_targets.txt"    
    save_path_prompts = args.save_path + "prompts_shuffled.txt"    
    save_path_responses = args.save_path + "responses_shuffled.txt"    

    targets = list([1] * val + [0] * val)
    
    out_targets = open(save_path_targets, 'w')
    for target in targets:
        out_targets.write(str(target) + "\n")
    out_targets.close()
    
    out_prompts = open(save_path_prompts, 'w')
    for prompt in shuffled_prompts:
        out_prompts.write(prompt)
    out_prompts.close()
    
    out_responses = open(save_path_responses, 'w')
    for response in shuffled_responses:
        out_responses.write(response)
    out_responses.close()
    


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
