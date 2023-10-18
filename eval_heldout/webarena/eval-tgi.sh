export SHOPPING="http://localhost:7770"
export SHOPPING_ADMIN="http://localhost:7780/admin"
export REDDIT="http://localhost:9999"
export GITLAB="http://localhost:8023"
export MAP="http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:3000/"
export WIKIPEDIA="http://localhost:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export HOMEPAGE="http://localhost:4399" # this is a placeholder
export CONTROLLER_ADDR=http://127.0.0.1:23333
export MODEL=agent-llama

export OPENAI_API_KEY=sk-your-openai-api-key

python run.py \
  --instruction_path agent/prompts/jsons/p_cot_id_actree_2s.json \
  --test_start_idx 0 \
  --test_end_idx 812 \
  --provider llama \
  --model $MODEL \
  --result_dir results/$MODEL