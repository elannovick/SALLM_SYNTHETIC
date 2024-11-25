# SALLM_SYNTHETIC
Synthetic data creation

This is divided into 3 sections:
1.) Synthetic data via story prompting.
2.) Syntetic data via story translation.
3.) Synthetic data via code comment translation.

The below outlines how each of these is done.

1.)Synthetic data via story prompting.
The synthetic data via story prompting was done for both isiZulu and isiXhosa. The code is provided in write_stories.py and in this case is for isiXhosa (which can easily be changed by just adjusting the prompt and verbs/adjectives in the random prompt generator). The code uses the Afrollama model which can be found here https://huggingface.co/Jacaranda/AfroLlama_V1.  For the purposes of this project, I used a 4-bit quantization for faster inference.

2.) Synthetic data via story translation.
The Syntetic data via story translation was done for isiXhosa, SiSwati, Sepedi, Sestwana and Xitsonga. The model used for translation was https://github.com/wxjiao/WMT2022-Large-Scale-African and the stories translated were from https://huggingface.co/datasets/roneneldan/TinyStories/viewer/default/train. Due to the very large size of TinyStories (Around 2 million stories),  the stories were split into 106 groups of 20k stories each stored in a text file each which can easily be done. The code for each can be found in the respective folders.


