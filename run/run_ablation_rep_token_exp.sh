llm_types = ["ChatGPT","Google-PaLM","Claude-instant","Llama-2-70b"]
rep_tokens = [-1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
for llm_type in llm_types:
    for rep_token in rep_tokens:
        !python3 repreGuard_evaluation.py \
            --model_name_or_path "meta-llama/Llama-3.1-8B" \
            --train_data_path "datasets/main_dataset/train_dataset_llm_type_{llm_type}.json" \
            --test_data_paths "detectrl_datasets/ablation_dataset/detectrl_test_dataset_llm_type_mix_llms.json" \
            --ntrain 512 \
            --batch_size 8 \
            --rep_token {rep_token} \
            --bootstrap_iter -1