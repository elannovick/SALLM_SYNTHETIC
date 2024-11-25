from transformers import AutoTokenizer
import torch
from vllm import LLM, SamplingParams
import json
import random

class JHModel:
    def __init__(self):
        self.eval_model = "/scratch/nvcela001/AfroLlama/models--Jacaranda--AfroLlama_V1/snapshots/7007fd088ad29078bb0431531c19700054d7f296"
        self.tokenizer = AutoTokenizer.from_pretrained(self.eval_model)
        self.ft_model = LLM(
            model=self.eval_model,
            tokenizer=self.eval_model,
            tensor_parallel_size=1  # Adjust for multiple GPUs if needed
        )
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        self.DEFAULT_SYSTEM_PROMPT = ""
        self.system_format = '<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>'
        self.user_format = '<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
        self.assistant_format = '{content}<|eot_id|>'

    def generate_prompt_new(self, instruction):
        return self.system_format.format(content=self.DEFAULT_SYSTEM_PROMPT) + self.user_format.format(content=instruction)

    def generate(self, instruction):
        prompt = self.generate_prompt_new(instruction)
        generation_config = dict(
            temperature=1.2,
            top_k=50,
            top_p=0.95,
            max_tokens=400,
            presence_penalty=1.0
        )
        generation_config["stop_token_ids"] = self.terminators
        generation_config["stop"] = ["<|eot_id|>", "<|end_of_text|>"]

        with torch.no_grad():
            output = self.ft_model.generate([prompt], SamplingParams(**generation_config), use_tqdm=False)
            response = output[0].outputs[0].text.strip()
            return response

    def generate_batch(self, instructions):
        responses = []
        for instruction in instructions:
            response = self.generate(instruction)
            responses.append(response)
        return responses

def main():
    import itertools

    # Initialize the model
    model = JHModel()

    # Output JSONL file path
    output_jsonl_path = "/scratch/nvcela001/xhosa_short_stories10.jsonl"

    # Define the number of stories to generate
    num_stories = 7500

    # Create lists of story elements in isiXhosa
    subjects = ["inkawu", "intaka", "ibhokhwe", "inzwane", "umfana", "intombazana", "unonkala", "ibhubesi", "inkwenkwezi", "ilizwe"]
    adjectives = ["ehlakaniphileyo", "enkulu", "encinci", "enothando", "ebalaseleyo", "enobuganga", "enomlingo", "enomdla", "enamandla", "emangalisayo"]
    actions = ["efumanisa imfihlelo", "edlala egadini", "ehamba ehlathini", "efunda isifundo", "edlala nomhlobo", "ehamba kuhambo olude", "efumanisa umhlaba omtsha", "elwela ubulungisa", "efumana ubuhlobo obutsha", "edibana nesoyikiso"]
    settings = ["ehlathini", "edolophini", "elwandle", "endlwini", "entabeni", "elwandle", "ekhaya", "esikolweni", "emhlabeni wefantasy", "kwilizwe elikude"]

    # Generate a list of unique prompts by combining elements
    prompt_templates = []

    for subject in subjects:
        for adjective in adjectives:
            for action in actions:
                for setting in settings:
                    prompt = f"Bhala ibali elifutshane malunga ne{subject} {adjective} {action} {setting}."
                    prompt_templates.append(prompt)

    # Randomly select prompts for the desired number of stories
    if len(prompt_templates) < num_stories:
        # If not enough unique prompts, allow repeats (unlikely with a large prompt list)
        instructions = random.choices(prompt_templates, k=num_stories)
    else:
        instructions = random.sample(prompt_templates, num_stories)

    # Shuffle instructions to randomize order
    random.shuffle(instructions)

    # Generate the stories
    xhosa_stories = model.generate_batch(instructions)

    # Write the stories to a JSONL file
    try:
        with open(output_jsonl_path, 'w', encoding='utf-8') as f:
            for instruction, story in zip(instructions, xhosa_stories):
                data = {
                    "instruction": instruction,
                    "output": story
                }
                json_line = json.dumps(data, ensure_ascii=False)
                f.write(json_line + '\n')
        print(f"Xhosa short stories saved to {output_jsonl_path}")
    except Exception as e:
        print(f"Error saving output JSONL: {e}")

if __name__ == "__main__":
    main()
