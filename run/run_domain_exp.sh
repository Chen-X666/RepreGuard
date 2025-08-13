!python3 repreGuard_evaluation.py \
        --model_name_or_path "meta-llama/Llama-3.1-8B" \
        --train_data_path "datasets/detectrl_dataset/domain_dataset/detectrl_train_dataset_domain_yelp_review_writing_prompt.json" \
        --test_data_paths "datasets/detectrl_dataset/domain_dataset/detectrl_test_dataset_domain_arxiv_xsum.json" \
        --ntrain 512 \
        --batch_size 8 \
        --rep_token 0.1 \
        --bootstrap_iter -1

!python3 repreGuard_evaluation.py \
        --model_name_or_path "meta-llama/Llama-3.1-8B" \
        --train_data_path "datasets/detectrl_dataset/domain_dataset/detectrl_train_dataset_domain_yelp_review_arxiv_xsum.json" \
        --test_data_paths "datasets/detectrl_dataset/domain_dataset/detectrl_test_dataset_domain_writing_prompt.json" \
        --ntrain 512 \
        --batch_size 8 \
        --rep_token 0.1 \
        --bootstrap_iter -1

!python3 repreGuard_evaluation.py \
        --model_name_or_path "meta-llama/Llama-3.1-8B" \
        --train_data_path "datasets/detectrl_dataset/domain_dataset/detectrl_train_dataset_domain_writing_prompt_yelp_review.json" \
        --test_data_paths "datasets/detectrl_dataset/domain_dataset/detectrl_test_dataset_domain_arxiv_xsum.json" \
        --ntrain 512 \
        --batch_size 8 \
        --rep_token 0.1 \
        --bootstrap_iter -1

!python3 repreGuard_evaluation.py \
        --model_name_or_path "meta-llama/Llama-3.1-8B" \
        --train_data_path "datasets/detectrl_dataset/domain_dataset/detectrl_train_dataset_domain_writing_prompt_arxiv_xsum.json" \
        --test_data_paths "datasets/detectrl_dataset/domain_dataset/detectrl_test_dataset_domain_yelp_review.json" \
        --ntrain 512 \
        --batch_size 8 \
        --rep_token 0.1 \
        --bootstrap_iter -1