import os
import json
import random
import numpy as np
import logging
import torch
from tqdm import tqdm
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, precision_score, recall_score,
    accuracy_score, f1_score, roc_auc_score
)
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, pipeline, set_seed
)
from baselines.Detectors.metrics import get_roc_by_threshold,get_roc_metrics
import argparse
from repe import repe_pipeline_registry
repe_pipeline_registry()

class AIHumanFunctionModel:
    def __init__(self, model_name_or_path, ntrain, rep_token, batch_size, random_seed=2025, ai_weight=1, human_weight=1, n_difference=1, direction_method='pca'):
        set_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

        self.model_name = os.path.basename(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
        use_fast_tokenizer = "LlamaForCausalLM" not in self.model.config.architectures
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast_tokenizer=use_fast_tokenizer, padding_side="left", legacy=False)
        self.tokenizer.pad_token_id = 0
        self.rep_reading_pipeline =  pipeline("rep-reading", model=self.model, tokenizer=self.tokenizer)
        self.ntrain = ntrain
        self.hidden_layers = list(range(-1, -self.model.config.num_hidden_layers, -1))
        self.rep_token = rep_token
        self.batch_size = batch_size
        self.n_difference = n_difference
        self.direction_method = direction_method
        self.ai_weight = ai_weight
        self.human_weight = human_weight
        self.rep_reader = None
        
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    def ai_human_function_dataset(self, train_dataset: str, tokenizer: PreTrainedTokenizer):
        pos_statements = []
        neg_statements = []
        # ai_datasets = [item for item in train_dataset if item.get("label") == "llm"]
        # human_datasets = [item for item in train_dataset if item.get("label") == "human"]
        ai_datasets = [item['direct_prompt'] for item in train_dataset]
        human_datasets = [item['human_text'] for item in train_dataset]

        for ai_data, human_data in zip(ai_datasets, human_datasets):
            # if ai_data['id'] == human_data['id'] and ai_data['domain'] == human_data['domain']:
                tokens_pos_statement = tokenizer.tokenize(ai_data)
                tokens_neg_statement = tokenizer.tokenize(human_data)

                string_tokens_pos_statement = tokenizer.convert_tokens_to_string(tokens_pos_statement)
                string_tokens_neg_statement = tokenizer.convert_tokens_to_string(tokens_neg_statement)
                pos_statements.append(string_tokens_pos_statement)
                neg_statements.append(string_tokens_neg_statement)

        combined_data = [[pos, neg] for pos, neg in zip(pos_statements, neg_statements)]
        train_data = combined_data
        train_labels = []
        for d in train_data:
            true_s = d[0]
            random.shuffle(d)
            train_labels.append([s == true_s for s in d])

        train_data = np.concatenate(train_data).tolist()

        return {
            'train': {'data': train_data, 'labels': train_labels}
        }

    def process_data(self, data, mode="train"):
        input_statements = []
        input_labels = []
        # ai_datasets = [item for item in data if item.get("label") == "llm"]
        # human_datasets = [item for item in data if item.get("label") == "human"]
        ai_datasets = [item['direct_prompt'] for item in data]
        human_datasets = [item['human_text'] for item in data]
    
        for ai_data, human_data in zip(ai_datasets, human_datasets):
            # if ai_data['id'] == human_data['id'] and ai_data['domain'] == human_data['domain']:
                input_statements.append(ai_data)
                input_labels.append(1)
                input_statements.append(human_data)
                input_labels.append(0)
        
        all_sentence_scores = []
        for statement in tqdm(input_statements):
            H_test_token = self.rep_reading_pipeline([statement],
                                    rep_reader=self.rep_reader,
                                    rep_token=0,
                                    hidden_layers=self.hidden_layers)
            all_token_scores = []
            
            num_tokens = len(H_test_token[0][-1][0])

            for token_idx in range(1,num_tokens,1):
                token_scores = []

                for layer in self.hidden_layers:
                    # 将当前 token 在当前层的分数添加到 token_scores 列表中
                    token_score_in_layer = H_test_token[0][layer][0][token_idx] * self.rep_reader.direction_signs[layer][0]
                    token_scores.append(token_score_in_layer)
                
                # 将当前 token 的所有层分数添加到 all_token_scores 中
                all_token_scores.append(token_scores)
            all_sentence_scores.append(all_token_scores)
    
        json_data = []
        for statement, sentence_score, label in zip(input_statements, all_sentence_scores, input_labels):
            data = {
                f"{mode}_input_statement": statement,
                "rep_reader_scores_dict": np.mean(sentence_score),
                f"{mode}_input_label": label
            }
            json_data.append(data)
    
        return json_data

    def save_json(self, data, file_path):
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

    def process_train_data(self,train_data):
        # logging.info(f"Train in {test_data_path}")
        # train_data = json.load(open(train_data_path, "r"))
        dataset = self.ai_human_function_dataset(train_data, self.tokenizer)
   
        self.rep_reader = self.rep_reading_pipeline.get_directions(
            dataset['train']['data'],
            rep_token=self.rep_token,
            hidden_layers=self.hidden_layers,
            n_difference=self.n_difference,
            train_labels=dataset['train']['labels'],
            direction_method=self.direction_method,
            batch_size=self.batch_size,
            ai_weight=self.ai_weight,
            human_weight=self.human_weight,
        )

        train_json_data = self.process_data(train_data, mode="train")
        # train_file_name = os.path.basename(f"{self.train_data_path.split('.json')[0]}_ntrain_{self.ntrain}_reptoken_{self.rep_token}")
        # self.save_json(train_json_data, f'results/{train_file_name}.json')

        return train_json_data

    def process_test_data(self,test_data):
        # logging.info(f"Test in {test_data_path}")
        # test_data = json.load(open(test_data_path, "r"))

        test_json_data = self.process_data(test_data, mode="test")

        # test_file_name = f"{os.path.basename(self.test_data_path.split('.json')[0])}_BY_{os.path.basename(self.train_data_path.split('.json')[0])}_ntrain_{self.ntrain}_reptoken_{self.rep_token}"
        # self.save_json(test_json_data, f'results/{test_file_name}.json')
        return test_json_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--train_data_path', type=str, required=True)
    parser.add_argument('--test_data_path', type=str, required=True)
    parser.add_argument('--ntrain', default=128, type=int)
    parser.add_argument('--rep_token', default=-1, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    # parser.add_argument('--mode',default='test',type=str)
    args = parser.parse_args()
    # entrance(args)
