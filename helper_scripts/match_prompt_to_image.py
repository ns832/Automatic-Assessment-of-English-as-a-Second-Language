import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--image_folder_path', type=str, default="/scratches/dialfs/alta/relevance/imports/ques_images/", help='Load path of the folder containing prompts, responses etc.')
parser.add_argument('--main_folder_path', type=str, default="/scratches/dialfs/alta/relevance/data/relevance_v2/", help='Load path of the folder containing prompts, responses etc.')
parser.add_argument('--prompts_path', type=str, default="/scratches/dialfs/alta/relevance/data/relevance_v2/LIESTcal01/GKTS4/prompts.txt", help='Load path of question training data')
parser.add_argument('--prompt_ids_path', type=str, default="/scratches/dialfs/alta/relevance/data/relevance_v2/LIESTcal01/GKTS4/prompt_ids.txt", help='Load path of question training data')
parser.add_argument('--real_prompts_path', type=str, default="/scratches/dialfs/alta/relevance/ns832/data/eval_real/section_D_eval/prompts.txt", help='Load path of question training data')
parser.add_argument('--real_responses_path', type=str, default="/scratches/dialfs/alta/relevance/ns832/data/eval_real/section_D_eval/responses.txt", help='Load path of question training data')
parser.add_argument('--real_targets_path', type=str, default="/scratches/dialfs/alta/relevance/ns832/data/eval_real/section_D_eval/targets.txt", help='Load path of question training data')
parser.add_argument('--real_topics_path', type=str, default="/scratches/dialfs/alta/relevance/ns832/data/eval_real/section_D_eval/topics.txt", help='Load path of question training data')


def match_image_id_to_prompt(args):
    # Get image ids
    image_files = os.listdir(args.image_folder_path)
    image_files = [x.strip() for x in image_files]
    image_files = [x for x in image_files if x[0] != "." and x != "README.txt"]
    image_files = [((x.replace('.', '')).replace('png', '')) for x in image_files]
    prompt_dict = dict()
    
    
    for i, image in enumerate(image_files):
        # Extract line number with that prompt id
        print(i)
        search_path = "/scratches/dialfs/alta/relevance/data/relevance_v2/"
        command = "grep -m 1 -o -R -n \"" + image +"\" " + search_path +  " | cut -d \":\" -f 2 |  head -n 1"
        line_number = os.popen(command).read()
        
        # Get file name associated with the line
        command = "grep -m 1 -o -R -n \"" + image +"\" " + search_path +  " | cut -d \":\" -f 1 |  head -n 1"
        file = os.popen(command).read()
        file = file.replace("prompt_ids", "prompts")
        
        # Get the prompt
        command =  "sed -n '" + line_number.strip() + "p' " + file
        prompt = os.popen(command).read().replace("\n", "")
        if "_" in prompt:prompt = " ".join(prompt.split()[2:])
        if prompt in prompt_dict.keys():
            print("Duplicate prompt: ", prompt, "Ids: ", prompt_dict[prompt], image)
        prompt_dict[prompt.strip()] = image
        
        
    with open(args.real_prompts_path) as f1, open(args.real_responses_path) as f2, open(args.real_targets_path) as f3, open(args.real_topics_path) as f4: 
        prompts = f1.readlines()
        responses = f2.readlines()
        targets = f3.readlines()
        
        prompt_list, response_list, target_list, prompt_id_list = [],[],[],[]
        prompts = {x.strip() for x in prompts}
        # prompts = [x.strip() for x in prompts]
        
        no_id_list = []
        for prompt, response, target in zip(prompts, responses, targets):
            if prompt not in prompt_dict.keys():
                print("Prompt_id: None Prompt:", prompt)
                no_id_list.append(prompt)
            else:
                prompt_list.append(prompt)
                # response_list.append(response)
                # target_list.append(target)
                prompt_id_list.append(prompt_dict[prompt])
                
        save_path = "/scratches/dialfs/alta/relevance/ns832/data/visual_and_text/eval_real/"      
        prompt_file = open(save_path + "image_questions.txt", 'w')
        for prompt in prompt_list:
            prompt_file.write(prompt + "\n")
        prompt_file.close()                 
        # resp_file = open(save_path + "responses.txt", 'w')
        # for resp in response_list:
        #     resp_file.write(resp + "\n")
        # resp_file.close()             
        # target_file = open(save_path + "targets.txt", 'w')
        # for target in target_list:
        #     target_file.write(target + "\n")
        # target_file.close()         
        ids_file = open(save_path + "image_ids.txt", 'w')
        for id in prompt_id_list:
            ids_file.write(id + "\n")
        ids_file.close()           
        # topics_file = open(save_path + "topics.txt", 'w')
        # for topic in topic_list:
        #     topics_file.write(topic + "\n")
        # topics_file.close()          
                
    return

              
def match_prompt_to_image_id(path, new_file_name, search_path = "/scratches/dialfs/alta/relevance/data/relevance_v2/"):
    prompts = open(path).readlines()
    prompts = {x.upper() for x in prompts}
    prompt_ids = []
    
    for i,prompt in enumerate(prompts):
        prompt = prompt.replace('\n', '')
        command = "grep -m 1 -o -R -n \"" + prompt +"\" " + search_path +  " | cut -d \":\" -f 2 |  sed -n '1p'"
        line_number = os.popen(command).read()
        if line_number == "": 
            print("Not found: ", command)
            raise
        
        # Get file name associated with the line
        command = "grep -m 1 -o -R -n \"" + prompt +"\" " + search_path +  " | cut -d \":\" -f 1 |  sed -n '1p'"
        file = os.popen(command).read()
        if "/prompts.txt" in file: file = file.replace("prompts", "prompt_ids")
            
        # Get the prompt
        command =  "sed -n '" + line_number.strip() + "p' " + file

        prompt_id = os.popen(command).read().strip("\n")
        if "_" in prompt_id: prompt_id = prompt_id.split()[1]
        prompt_ids.append(prompt_id)
        print("Prompt id ", i, "of ", len(prompts), "added:", prompt_id)
    
    # out_file = open(new_file_name, 'w')
    # for id in prompt_ids:
    #     out_file.write(id + "\n")
    # out_file.close()
    
    
if __name__ == '__main__':
    args = parser.parse_args()
    # match_image_id_to_prompt(args)
    # missing_path = "/scratches/dialfs/alta/relevance/ns832/data/visual_and_text/eval_real/missing_prompts.txt"
    # new_file_name = "/scratches/dialfs/alta/relevance/ns832/data/visual_and_text/eval_real/missing_prompt_ids.txt"
    # match_prompt_to_image_id(missing_path, new_file_name)
    
    path = "/scratches/dialfs/alta/relevance/ns832/data/relevance_v2/LIESTgrp06/prompts.txt"
    new_file_name = "/scratches/dialfs/alta/relevance/ns832/data/visual_and_text/eval_real/image_ids"
    match_prompt_to_image_id(path, new_file_name)
