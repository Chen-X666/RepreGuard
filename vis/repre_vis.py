import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from typing import List, Union, Optional
import torch
import numpy as np

class HiddenStateExtractor:
    def __init__(self, model, tokenizer):
        """
        Args:
        - model: loaded model
        - tokenizer: loaded tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer

    def get_hidden_states(self, outputs,
                          rep_token: Union[str, float] = -1,
                          hidden_layers: Union[List[int], int] = -1,
                          which_hidden_states: Optional[str] = None):
        if hasattr(outputs, 'encoder_hidden_states') and hasattr(outputs, 'decoder_hidden_states'):
            outputs['hidden_states'] = outputs[f'{which_hidden_states}_hidden_states']

        hidden_states_layers = {}
        for layer in hidden_layers:
            hidden_states = outputs['hidden_states'][layer]
            # 0 < rep_token <= 1 is the percentage of tokens to keep
            if 0 < rep_token <= 1:
                rep_token_num = int(rep_token * hidden_states.shape[1])
                hidden_states = torch.stack([hidden_states[:, i, :] for i in range(-1, -rep_token_num, -1)], dim=1)
                hidden_states = torch.mean(hidden_states, dim=1)
            # 0 is get all the tokens hidden states
            elif rep_token == 0:
                hidden_states = hidden_states
            # -1 is get the last token hidden states
            elif rep_token < 0:
                rep_token = int(rep_token)
                hidden_states = hidden_states[:, rep_token, :]

            hidden_states_layers[layer] = hidden_states.detach()

        return hidden_states_layers

    def forward(self, model_inputs, rep_token, hidden_layers, which_hidden_states=None):
        with torch.no_grad():
            if hasattr(self.model, "encoder") and hasattr(self.model, "decoder"):
                decoder_start_token = [self.tokenizer.pad_token] * model_inputs['input_ids'].size(0)
                decoder_input = self.tokenizer(decoder_start_token, return_tensors="pt").input_ids
                model_inputs['decoder_input_ids'] = decoder_input
                
            outputs = self.model(**model_inputs, output_hidden_states=True)

        return self.get_hidden_states(outputs, rep_token, hidden_layers, which_hidden_states)

    def batched_string_to_hiddens(self, train_inputs, rep_token, hidden_layers, batch_size, which_hidden_states, train_labels=None, **tokenizer_args):
        # Tokenize the inputs
        model_inputs = self.tokenizer(train_inputs, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        
        # Get the hidden states from the model
        hidden_states_outputs = self.forward(model_inputs, rep_token=rep_token,
                                             hidden_layers=hidden_layers, which_hidden_states=which_hidden_states)
        
        # Create a dictionary to store hidden states
        hidden_states = {layer: [] for layer in hidden_layers}
        
        # Store hidden states for each batch in the dictionary
        for layer, hidden_state in hidden_states_outputs.items():
            hidden_states[layer].append(hidden_state.cpu().numpy())
        
        # Convert lists to numpy arrays
        return {k: np.vstack(v) for k, v in hidden_states.items()}

def load_model_and_tokenizer(model_path):
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    use_fast = "LlamaForCausalLM" not in model.config.architectures
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        use_fast_tokenizer=use_fast,
        padding_side="left", 
        legacy=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0 
    
    extractor = HiddenStateExtractor(model=model, tokenizer=tokenizer)
    return model, extractor

def load_and_prep_data(data_path, sample_size):
    print(f"Loading data from {data_path}...")
    with open(data_path, "r") as f:
        train_data = json.load(f)
    
    print(f"Total data entries: {len(train_data)}")

    human_data = sorted([entry['human_text'] for entry in train_data], key=len, reverse=True)
    llm_data = sorted([entry['direct_prompt'] for entry in train_data], key=len, reverse=True)

    return human_data[:sample_size], llm_data[:sample_size]

def extract_features(data_list, extractor, hidden_layers, truncate_len, batch_size=1):
    all_hs_tensor_list = []
    
    print(f"Extracting features for {len(data_list)} samples...")
    for item in tqdm(data_list):
        # Extract hidden states
        hs_dict = extractor.batched_string_to_hiddens(
            train_inputs=[item],
            rep_token=0, 
            hidden_layers=hidden_layers, 
            batch_size=batch_size,               
            which_hidden_states=None 
        )
        
        processed_layers = []
        for v in hs_dict.values():
            tensor_v = torch.tensor(v) # [1, seq_len, hidden_dim]
            # 截断
            if tensor_v.shape[1] > truncate_len:
                tensor_v = tensor_v[:, :truncate_len, :]
            else:
                # Pad if necessary
                pass 
            processed_layers.append(tensor_v)
            
        layers_tensor = torch.stack(processed_layers) # [num_layers, 1, truncated_len, hidden_dim]
        all_hs_tensor_list.append(layers_tensor)

    if not all_hs_tensor_list:
        raise ValueError("No valid tensors extracted. Check if data length >= truncate_len.")

    all_hs_tensor = torch.stack(all_hs_tensor_list)
    all_hs_tensor = all_hs_tensor.squeeze(dim=2) 
    
    # 1. Take mean across the neuron dimension (hidden_dim) -> [sample_size, num_layers, truncated_len]
    mean_hs = torch.mean(all_hs_tensor, dim=-1)
    
    # 2. Take mean across the sample dimension (sample_size) -> [num_layers, truncated_len]
    final_mean_hs = torch.mean(mean_hs, dim=0)
    
    return final_mean_hs


def plot_lat_scans(llm_tensor, human_tensor, save_path):
    standardized_scores1 = np.array(llm_tensor.T.cpu().numpy())
    standardized_scores2 = np.array(human_tensor.T.cpu().numpy())

    # Calculate bounds
    bound1 = np.mean(standardized_scores1) + np.std(standardized_scores1)
    bound2 = np.mean(standardized_scores2) + np.std(standardized_scores2)

    standardized_scores1 = standardized_scores1.clip(-bound1, bound1)
    standardized_scores2 = standardized_scores2.clip(-bound2, bound2)

    cmap = 'coolwarm'

    fig, axs = plt.subplots(2, 1, figsize=(10, 4), dpi=300, sharex=True)

    cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])  # [left, bottom, width, height]

    sns.heatmap(np.flipud(standardized_scores1.T), cmap=cmap, linewidth=0.3, annot=False, fmt=".3f", ax=axs[0], cbar=False)
    axs[0].set_ylabel("Hidden Layer",fontsize=10)
    axs[0].set_xticks(np.arange(0, len(standardized_scores1), 10)[1:])
    axs[0].set_xticklabels(np.arange(0, len(standardized_scores1), 10)[1:])
    axs[0].tick_params(axis='x', rotation=0)
    axs[0].set_yticks(np.arange(0, len(standardized_scores1[0]), 5)[1:])
    axs[0].set_yticklabels(np.arange(1, len(standardized_scores1[0]) + 1, 5)[::-1][1:],fontsize=12)
    axs[0].set_title("Hidden Representation Patterns of LGT",fontsize=12)

    sns.heatmap(np.flipud(standardized_scores2.T), cmap=cmap, linewidth=0.3, annot=False, fmt=".3f", ax=axs[1], cbar_ax=cbar_ax)
    axs[1].set_xlabel("Token Position",fontsize=10)
    axs[1].set_ylabel("Hidden Layer",fontsize=10)
    axs[1].set_xticks(np.arange(0, len(standardized_scores2), 10)[1:])
    axs[1].set_xticklabels(np.arange(0, len(standardized_scores2), 10)[1:])
    axs[1].tick_params(axis='x', rotation=0)
    axs[1].set_yticks(np.arange(0, len(standardized_scores2[0]), 5)[1:])
    axs[1].set_yticklabels(np.arange(1, len(standardized_scores2[0]) + 1, 5)[::-1][1:],fontsize=12)
    axs[1].set_title("Hidden Representation Patterns of HWT",fontsize=12)

    cbar = axs[1].collections[0].colorbar
    cbar.set_label("Activation Level", rotation=270, labelpad=15, fontsize=12)

    plt.tight_layout(rect=[0, 0, 0.92, 1]) 

    print(f"Saving plot to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.show()


def main(args):
    print(f"Configuration: {vars(args)}")

    model, extractor = load_model_and_tokenizer(args.model_path)
    
    human_data, llm_data = load_and_prep_data(args.data_path, args.sample_size)
    
    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
    
    print("\nProcessing Human Data...")
    human_mean_hs = extract_features(human_data, extractor, hidden_layers, args.truncate_len, args.batch_size)
    
    print("\nProcessing LLM Data...")
    llm_mean_hs = extract_features(llm_data, extractor, hidden_layers, args.truncate_len, args.batch_size)
    
    plot_lat_scans(llm_mean_hs, human_mean_hs, args.save_plot_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Hidden State Extraction and Visualization")
    parser.add_argument("--model_path", type=str, 
                        default="meta-llama/Llama-2-7b-hf",
                        help="Path to the pretrained model")
    parser.add_argument("--data_path", type=str, 
                        default="datasets/detectrl_dataset/main_dataset/detectrl_train_dataset_llm_type_mix_llms_interleaved.json",
                        help="Path to the dataset JSON file")
    parser.add_argument("--sample_size", type=int, default=1000, 
                        help="Number of samples to use for each category (Human/LLM)")
    parser.add_argument("--truncate_len", type=int, default=210, 
                        help="Token length to truncate sequences to")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Batch size for hidden state extraction")
    parser.add_argument("--save_plot_name", type=str, default='Neural_Activity.pdf', 
                        help="Filename to save the resulting plot")
    args = parser.parse_args()
    main(args)
