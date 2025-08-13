open_models = ["llama-chat", "mistral-chat", "mistral", "mpt-chat", "mpt", "gpt2"]
decodings_types = ["sampling", "greedy"]
repetition_penaltys = ["yes", "no"]

for model in open_models:
    for decoding in decodings_types:
        for repetition in repetition_penaltys:
            !python3 repreGuard_evaluation.py \
                --model_name_or_path "meta-llama/Llama-3.1-8B" \
                --train_data_path "raid_datasets/decoding_dataset/raid_train_dataset_llm_type_{model}_decoding_type_{decoding}_repetition_penalty_{repetition}.json" \
                --test_data_paths "raid_datasets/decoding_dataset/raid_test_dataset_llm_type_{model}_decoding_type_{decoding}_repetition_penalty_{repetition}.json" \
                --ntrain 512 \
                --bootstrap_iter -1 \
                --batch_size 4 \
                --rep_token 0.1