llm_types = ["ChatGPT","Google-PaLM","Claude-instant","Llama-2-70b"]
for llm_type in llm_types:
    !python3 repreGuard_evaluation.py \
        --model_name_or_path "meta-llama/Llama-3.1-8B" \
        --train_data_path "detectrl_dataset/main_dataset/train_dataset_llm_type_{llm_type}.json" \
        --test_data_paths "detectrl_dataset/main_dataset/test_dataset_llm_type_ChatGPT.json, detectrl_dataset/main_dataset/test_dataset_llm_type_Google-PaLM.json, detectrl_dataset/main_dataset/test_dataset_llm_type_Claude-instant.json, detectrl_dataset/main_dataset/test_dataset_llm_type_Llama-2-70b.json" \
        --ntrain 512 \
        --batch_size 8 \
        --rep_token 0.1 \
        --bootstrap_iter 5