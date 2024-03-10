import argparse
import os

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--prompts_path', type=str, help='Load path to test prompts as text')
parser.add_argument('--search_path', type=str, help='Path to search')
parser.add_argument('--responses_path', type=str, help='Path to responses')
parser.add_argument('--prompt_ids_path', type=str, help='Path to prompt_ids')
parser.add_argument('--save_path', type=str, help='Path to prompt_ids')
parser.add_argument('--targets_path', type=str, help='Path to targets')
args = parser.parse_args()


with open(args.prompt_ids_path) as f0, open(args.responses_path) as f1, open(args.prompts_path) as f2, open(args.targets_path) as f3:
    responses_list, prompts, prompt_ids, target_list = [], [], [], []
    question_ids, responses, questions, targets = f0.readlines(), f1.readlines(), f2.readlines(), f3.readlines()

    # For eval-real set
    for question, question_id, response, target in zip(questions, question_ids, responses, targets):
        # Extract first line with that prompt
        command = "grep -m 1 -o -n \"" + question.strip() +"\" " + args.search_path +  "prompts.txt | cut -d \":\" -f 1"
        line_number = os.popen(command).read()
        if line_number == "": print("Not found")
        # Get the line number from that section
        command =  "sed -n '" + line_number.strip() + "p' " + args.search_path + "sections.txt"
        section = os.popen(command).read()
        
        if section.strip() == "E":
            target_list.append(target)
            responses_list.append(response)
            prompts.append(question)
            prompt_ids.append(question_id)
    print(len(responses_list))

    save_path_prompts = args.save_path + "prompts.txt"
    # save_path_prompt_ids = args.save_path + "prompt_ids.txt"
    save_path_responses = args.save_path + "responses.txt"
    save_path_targets = args.save_path + "targets.txt"
    
    out_prompts = open(save_path_prompts, 'w')
    for prompt in prompts:
        out_prompts.write(prompt)
    out_prompts.close()
    out_responses = open(save_path_responses, 'w')
    for response in responses_list:
        out_responses.write(response)
    out_responses.close()
    # out_prompt_ids = open(save_path_prompt_ids, 'w')
    # for prompt_id in prompt_ids:
    #     out_prompt_ids.write(prompt_id)
    # out_prompt_ids.close()
    
    out_targets = open(save_path_targets, 'w')
    for target in target_list:
        out_targets.write(target)
    out_targets.close()
    
    # with open(args.targets_path) as f0, open(args.sections_path) as f1:
    #     target_list = []
    #     targets, sections = f0.readlines(), f1.readlines()
    #     for target, section in zip(targets, sections):
    #         if "D" in section:
    #             print("Section D found")
    #             target_list.append(target)
                
    