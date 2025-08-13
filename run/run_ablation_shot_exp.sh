llm_types = ["ChatGPT","Google-PaLM","Claude-instant","Llama-2-70b"]
shots = [8,16,32,64,128,256,512,1024]
for llm_type in llm_types:
    for shot in shots:
        !python3 repreGuard_evaluation.py \
            --model_name_or_path "meta-llama/Llama-3.1-8B" \
            --train_data_path "dataset/detectrl_dataset/main_dataset/detectrl_train_dataset_llm_type_{llm_type}.json" \
            --test_data_paths "dataset/detectrl_dataset/ablation_dataset/detectrl_test_dataset_llm_type_mix_llms.json" \
            --ntrain {shot} \
            --batch_size 8 \
            --rep_token 0.1 \
            --bootstrap_iter -1