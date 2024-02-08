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
    # subfolders =[x for x in subfolders if "README" not in x and "README~" not in x]
    print(subfolders)
    
    # Get all the prompts and their corresponding ids
    prompts, prompt_ids = [], []
    for subfolder in subfolders:
        path = args.main_folder_path + subfolder + "/GKTS4/"
        try:
            with open(path + "prompts.txt") as f1, open(path + "prompt_ids.txt") as f2:
                file_prompts, file_prompt_ids = f1.readlines(), f2.readlines()
                file_prompt_ids = [x.strip() for x in file_prompt_ids]
                file_prompts = [x.strip().lower() for x in file_prompts]
                prompts += file_prompts
                prompt_ids += file_prompt_ids
        except:
            try:
                path = args.main_folder_path + subfolder + "/" + subfolder + "/"
                with open(path + "prompts.txt") as f1, open(path + "prompt_ids.txt") as f2:
                    file_prompts, file_prompt_ids = f1.readlines(), f2.readlines()
                    file_prompt_ids = [x.strip() for x in file_prompt_ids]
                    file_prompts = [x.strip().lower() for x in file_prompts]
                    prompts += file_prompts
                    prompt_ids += file_prompt_ids
            except:
                print("File not found for: ", path)
            
            
    files = os.listdir(args.image_folder_path)
    files = [x.strip() for x in files]
    id_prompt_dict = dict()
    
    files = [((x.replace('.', '')).replace('png', '')) for x in files]
    not_found_files = files.copy()
    
    for file in files:
        for index, value in enumerate(prompt_ids):
            if value == file:
                prompt = prompts[index]
                id_prompt_dict[prompt] = file
                not_found_files.remove(file)
                break
            
    print(not_found_files)
    print(len(files), len(not_found_files))
    
    
    with open(args.real_prompts_path) as f1, open(args.real_responses_path) as f2, open(args.real_targets_path) as f3, open(args.real_topics_path) as f4: 
        prompts = f1.readlines()
        responses = f2.readlines()
        targets = f3.readlines()
        topics = f4.readlines()
        prompts = [x.strip().lower() for x in prompts]
        
        id_list, response_list, prompt_list, target_list, topic_list = [], [], [], [], []

        for prompt, response, target, topic in zip(prompts, responses, targets, topics):
            if prompt in id_prompt_dict.keys():
                prompt_list.append(prompt)
                id_list.append(id_prompt_dict[prompt])
                response_list.append(response)
                target_list.append(target)
                topic_list.append(topic)
        
        print(len(target_list))
        assert(len(prompt_list) == len(response_list) == len(id_list) == len(target_list) == len(topic_list))
        
        save_path = "/scratches/dialfs/alta/relevance/ns832/data/visual_and_text/eval_real/"
        out_responses = open(save_path + "responses.txt", 'w')
        for response in response_list:
            out_responses.write(response)
        out_responses.close()

        out_prompt = open(save_path + "prompts.txt", 'w')
        for prompt in prompt_list:
            out_prompt.write(prompt + '\n')
        out_prompt.close()
        
        out_targets = open(save_path + "targets.txt", 'w')
        for target in target_list:
            out_targets.write(target)
        out_targets.close()
        
        out_topics = open(save_path + "topics.txt", 'w')
        for topic in topic_list:
            out_topics.write(topic)
        out_topics.close()
        
        out_image_id = open(save_path + "image_ids.txt", 'w')
        for id in id_list:
            out_image_id.write(id + '\n')
        out_image_id.close()
        
        # Get real data from these prompts
        
        
        
        # # CEL-1BI-00001_P10003 b28ae556-0f27-4df4-8c43-d25cf8cf3925
        # split_prompts = prompts.split()
        # # CEL-1BI-00001 (section) P10003 (part)
        # section, part = split_prompts[0].split("_")
        # # b28ae556-0f27-4df4-8c43-d25cf8cf3925 (image id)
        # id = split_prompts[1]
    return

# First loop through the images

    # Within that loop, loop through the prompts and look at which maximises the probability/f_score
    
    # Within this loop the function should call LLaVA on 1000 trials and return the f_score
    
    
    
if __name__ == '__main__':
    args = parser.parse_args()
    match_image_id_to_prompt(args)