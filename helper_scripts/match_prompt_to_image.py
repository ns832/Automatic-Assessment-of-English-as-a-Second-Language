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
    subfolders = os.listdir(args.main_folder_path)

    # Get image ids
    image_files = os.listdir(args.image_folder_path)
    image_files = [x.strip() for x in image_files]
    image_files = [x for x in image_files if x[0] != "." and x != "README.txt"]
    image_files = [((x.replace('.', '')).replace('png', '')) for x in image_files]
    
    prompt_dict = dict()
    
    # Get prompt
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
        # print(prompt, image)
        
    print(len(prompt_dict.keys()))
    # save_path = "/scratches/dialfs/alta/relevance/ns832/data/prompt_database/"      
    # prompts_file = open(save_path + "prompts.txt", 'w')
    # prompt_ids_file = open(save_path + "prompts.txt", 'w')
    
    # for prompt, prompt_id in zip(prompt_dict.keys(), prompt_dict.values()):
    #     prompts_file.write(prompt)
    #     prompt_ids_file.write(prompt_id)
    # prompts_file.close()     
    # prompt_ids_file.close()     
    
    with open(args.real_prompts_path) as f1, open(args.real_responses_path) as f2, open(args.real_targets_path) as f3, open(args.real_topics_path) as f4: 
        prompts = f1.readlines()
        prompt_set = {x.strip() for x in prompts}
        
        no_id_list = []
        for prompt in prompt_set:
            if prompt not in prompt_dict.keys():
                print("Prompt_id: None Prompt:", prompt)
                no_id_list.append(prompt)
                
        # save_path = "/scratches/dialfs/alta/relevance/ns832/data/visual_and_text/eval_real/"      
        # no_id = open(save_path + "missing_prompts.txt", 'w')
        # for prompt in no_id_list:
        #     no_id.write(prompt)
        # no_id.close()              
                
    return

# First loop through the images

    # Within that loop, loop through the prompts and look at which maximises the probability/f_score
    
    # Within this loop the function should call LLaVA on 1000 trials and return the f_score
    
              
def match_prompt_to_id(missing_path = "/scratches/dialfs/alta/relevance/ns832/data/visual_and_text/eval_real/missing_prompts.txt"):
    missing_prompts = open(missing_path).readlines()
    missing_prompts = [x for x in missing_prompts]
    
    missing_prompt_ids = []
    
    for prompt in missing_prompts:
        prompt = prompt.replace('\n', '')
        search_path = "/scratches/dialfs/alta/relevance/data/relevance_v2/"
        
        n = 1
        while True:
            command = "grep -m 1 -o -R -n \"" + prompt +"\" " + search_path +  " | cut -d \":\" -f 2 |  sed -n '" + str(n) + "p'"
            line_number = os.popen(command).read()
            if line_number == "": 
                print("Not found: ", command)
                raise
            # Get file name associated with the line
            command = "grep -m 1 -o -R -n \"" + prompt +"\" " + search_path +  " | cut -d \":\" -f 1 |  sed -n '" + str(n) + "p'"
            file = os.popen(command).read()
            
            if "/prompts.txt" in file: file = file.replace("prompts", "prompt_ids")
                
            if file != "":break
            n += 15
            
        # Get the prompt
        command =  "sed -n '" + line_number.strip() + "p' " + file

        prompt_id = os.popen(command).read().strip("\n")
        if "_" in prompt_id: prompt_id = prompt_id.split()[1]
        missing_prompt_ids.append(prompt_id)
    
    
    # save_path = "/scratches/dialfs/alta/relevance/ns832/data/visual_and_text/eval_real/"
    # out_file = open(save_path + "missing_prompt_ids.txt", 'w')
    # for id in missing_prompt_ids:
    #     out_file.write(id + "\n")
    # out_file.close()
    
    
if __name__ == '__main__':
    args = parser.parse_args()
    match_image_id_to_prompt(args)
    # match_prompt_to_id()