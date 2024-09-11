# Automatic-Assessment-of-English-as-a-Second-Language
4th year NLP project on detecting on and off-topic responses 

Within the project two main methods are explored: a **zero-shot generative model using LLaVA/Mistral** (LLaVA including the use of images whereas Mistral works purely based of text) and a **bespoke BERT-based model**. All files can be run from the command line with arguments used to locate the data and model parameters. The BERT is split into two with a *train_BERT.py* and *eval_BERT.py*, using different datasets for the training and evaluation stages.

*llava_captioner.py* is used to create captions for images which can be used to attempt to enhance the information fed into the text-only models. 

Folders **Section_D** and **helper_scripts** are filled with scripts that isolate out section D data (the only data which has a visual component and can be used with LLaVA) and scripts which pre-process the data respectively. These do not need to be called by the user providing the data is in the correct format already.
