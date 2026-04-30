SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SAVE_DIR="$REPO_ROOT/save/tmp_transformer_tiny_polynomial"

echo "Saving to $SAVE_DIR"

mkdir -p "$SAVE_DIR"

export CUDA_VISIBLE_DEVICES=$1
RUN="${2:-1}"

python train.py \
	--output_dir "$SAVE_DIR" \
  	--expt_name Layer3Head2_totalseq_4_mod50_${RUN} \
	--logging_dir "$SAVE_DIR/logs"\
	--logging_steps 100 \
	--model_name_or_path transformer_tiny \
	--data_name polynomial \
	--seed 11 \
	--model_max_length 512 \
	--per_device_train_batch_size 256 \
  	--gradient_accumulation_steps 1 \
	--bf16 \
	--num_train_epochs 1000 \
	--learning_rate 3e-4 \
	--max_grad_norm 2.0 \
	--use_lora False \
	--lora_r 128 --lora_alpha 32 --lora_init \
	--save_strategy "no" \
	--save_safetensors False \
	--save_total_limit 1 \
	--weight_decay 0.1 \
	--warmup_ratio 0.03 \
	--lr_scheduler_type "cosine" \
	--do_train \
	--report_to tensorboard \
    --num_latent 6 \
    --logging_strategy "steps" \
	--use_prj True \
	--prj_dim 256 \
	--prj_dropout 0.0 \
	--distill_loss_div_std True \
	--exp_mode False \
	--exp_data_num 2000 \
	--remove_eos True \
	--print_ref_model_stats False \
	--print_loss False \
	--ce_loss_factor 1 \
	--distill_loss_factor 1 \
	--model_num_heads 2 \
	--model_num_layers 3 \
	--total_seqs 4 \
	--n_data_per_seq 5000 \
	--polynomial_mod 50 \
	--fix_attn_mask True \
	--extra_ans_token_in_ref_input True \
