llm_types = ["ChatGPT","Google-PaLM","Claude-instant","Llama-2-70b"]
for llm_type in llm_types:
    !python3 repreGuard_evaluation.py \
        --model_name_or_path "meta-llama/Llama-3.1-8B" \
        --train_data_path "
        datasets/detectrl_dataset/main_dataset/detectrl_train_dataset_llm_type_{llm_type}.json" \
        --test_data_paths "
        datasets/detectrl_dataset/length_dataset/detectrl_test_dataset_llm_type_mix_llms_data_distribution_64.json,
        datasets/detectrl_dataset/length_dataset/detectrl_test_dataset_llm_type_mix_llms_data_distribution_128.json,
        datasets/detectrl_dataset/length_dataset/detectrl_test_dataset_llm_type_mix_llms_data_distribution_192.json,
        datasets/detectrl_dataset/length_dataset/detectrl_test_dataset_llm_type_mix_llms_data_distribution_256.json" \
        --ntrain 512 \
        --batch_size 8 \
        --rep_token 0.1 \
        --bootstrap_iter -1