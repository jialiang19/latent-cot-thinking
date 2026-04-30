# Modified from https://github.com/zhenyi4/codi/blob/main/src/model.py 
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GPTNeoXForCausalLM, GPT2Config
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from dataclasses import dataclass, field
from typing import Optional
from peft import (
    get_peft_model,
    PeftModel,
    PeftConfig
)
from torch.nn.functional import gelu
import math
from safetensors.torch import load_file
from transformers.modeling_outputs import ModelOutput
import random
import copy
from src.data.data_processing import TOKEN_DICT as POLYNOMIAL_TOKEN_DICT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="mistralai/Mistral-7B-Instruct-v0.2")
    separate_decoder_name: str = field(default="")
    lora_r: int = field(default=128, metadata={"help": "lora rank"})
    lora_dropout: float = field(default=0.05, metadata={"help": "lora dropout"})
    full_precision: bool = field(default=True, metadata={"help": "whether use int4 for the base model"})
    train: bool = field(
        default=True,
        metadata={
            "help": "if true, the model ckpt will be initialized for training; else, it's for inference"
        },
    )
    lora_init: bool = field(
        default=False,
        metadata={"help": "True: Use zero and gaussian initialization; False: Load adapters from LoftQ in HF hub."},
    )
    token: Optional[str] = field(
        default=None,
        metadata={"help": "HF token to access to private models, e.g., meta-llama"},
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the LoRA adapter. Used in evaluation or resuming from the checkpoint."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoftQ does not require this config. Used for QLoRA."},
    )
    ckpt_dir: Optional[str] = field(default=None, metadata={"help": "checkpoint dir for inference."})
    model_num_layers: int = field(default=2, metadata={"help": "The number of layers of the model for tiny transformer."})
    model_num_heads: int = field(default=1, metadata={"help": "The number of heads of the model for tiny transformers."})
    model_embd_dim: int = field(default=128, metadata={"help": "The embedding dimension of the model."})
    model_embd_dropout: float = field(default=0.1, metadata={"help": "The dropout rate of the model embedding."})



@dataclass
class DataArguments:
    data_name: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    debug_data: bool = field(
        default=False,
        metadata={
            "help": "Enable debug dataset to quickly verify the training process"
        },
    )
    batch_size: int = field(default=1, metadata={"help": "batch size during inference"})
    total_seqs: int = field(default=32, metadata={"help": "for polynomial data only, The total number of sequences."})
    n_data_per_seq: int = field(default=1000, metadata={"help": "for polynomial data only, The number of data per sequence."})
    polynomial_mod: int = field(default=50, metadata={"help": "for polynomial data only, The modulus of the polynomial."}) 
    extra_ans_token_in_ref_input: bool = field(default=False, metadata={"help": "Adding the extra answer token to the reference input or not. Note that regardless of this, the student inputs (such as encoder_input_ids and decoder_input_ids) will not be affected."})
    hyper_polynomial_mod: int = field(default=50, metadata={"help": "for hyper polynomial data only, The modulus of the hyper polynomial."})
    hyper_polynomial_addition_value: int = field(default=1, metadata={"help": "for hyper polynomial data only, The addition value of the hyper polynomial."})
    hyper_poly_data_seed: int = field(default=1, metadata={"help": "for data generation/loading only, currently only used for hyper polynomial data in train_ans_ans.py"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=28000,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    restore_from: str = field(
        default="",
        metadata={
            "help": "The checkpoint that should be restored from for fine-tuning"
        },
    )
    per_device_train_batch_size: int = field(
        default=1,
    )
    per_device_eval_batch_size: int = field(
        default=1,
    )
    expt_name: str = field(
        default="default",
        metadata={"help": "Experiment name"},
    )
    icot_train_path: str = field(default="/users/k24020023/efficient_cot/icae/code/coconut/icot_gsm8k/train.txt", metadata={"help":"The training data path"})
    num_latent: int = field(default=5, metadata={"help": "The number of latent for training or inference."})
    use_lora: bool = field(default=True, metadata={"help": "Use lora or not."})
    greedy: bool = field(default=False, metadata={"help": "Greedy decoding during inference."})
    exp_mode: bool = field(default=False, metadata={"help": "Use partial number of data. for debugging."})
    exp_data_num: int = field(default=10000, metadata={"help": "The number of data used in exp mode"}) 
    use_prj: bool = field(default=False, metadata={"help": "Use a prj module after the llm for latent generation."}) 
    prj_dim: int = field(default=2048, metadata={"help": "The hidden dim of the projection module."})
    prj_dropout: float = field(default=0.0, metadata={"help": "Dropout ratio of the projection module."})
    prj_no_ln: bool = field(default=False, metadata={"help": "Remove the Layer Norm layer for the projection module."})
    distill_loss_div_std: bool = field(default=False, metadata={"help": "Divide the distillation loss by a std for normallisation."})
    distill_loss_type: str = field(default="smooth_l1", metadata={"help": "Specify the distillation loss. Use smoothL1 by default."})
    distill_loss_factor: float = field(default=1.0, metadata={"help": "A multiplier of the distillation loss."})
    ref_loss_factor: float = field(default=1.0, metadata={"help": "A multiplier of the reference teacher loss."})
    ce_loss_factor: float = field(default=1.0, metadata={"help": "A multiplier of the student CE loss."}) 
    inf_latent_iterations: int = field(default=1, metadata={"help": ""})
    inf_num_iterations: int = field(default=5, metadata={"help": "Run multiple times during inference"})
    remove_eos: bool = field(default=False, metadata={"help": "Do not add <eos> as a delimiter to split QA."})
    print_ref_model_stats: bool = field(default=False, metadata={"help": "Print some stats for the teacher task."})
    include_last_cot: bool = field(default=False, metadata={"help": "Include the last CoT step in the training data."})
    fix_attn_mask: bool = field(default=False, metadata={"help": "Correct a bug about attention mask."})
    log_full: bool = field(default=False, metadata={"help": "Log all losses."})
    print_loss: bool = field(default=True)
    max_token_num: int = field(default=1000, metadata={"help": "Limit the longest data to avoid OOM."})
    save_every_n_epochs: int = field(default=-1, metadata={"help": "Save the model every N epochs. -1 means no saving."})


def print_trainable_parameters(model):
    trainable_parameters = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    print(
        f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}"
    )


def freeze_model(model):
    for _, param in model.named_parameters():
        param.requires_grad = False

# tiny gpt-style model: 2 layers, 1 attention head, and try to match configuration in "iteration head" paper 
transformer_tiny_config = GPT2Config(
    n_layer=2,          # <-- two transformer blocks
    n_head=1,           # <-- one attention head per block
    n_embd=128,         # model width (must be divisible by n_head)
    vocab_size= max(POLYNOMIAL_TOKEN_DICT.values()) + 1,      # small toy vocab
    n_positions=128,     # context length
    n_ctx=64            # (legacy alias used by GPT-2)
)

class CODI(torch.nn.Module):
    def __init__(self, model_args, training_args, lora_config):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path
        model_wrapper_class = AutoModelForCausalLM 
        if "transformer_tiny" in model_args.model_name_or_path:
            transformer_tiny_config.n_layer = model_args.model_num_layers
            transformer_tiny_config.n_head = model_args.model_num_heads
            transformer_tiny_config.n_embd = model_args.model_embd_dim 
            transformer_tiny_config.resid_pdrop = model_args.model_embd_dropout
            transformer_tiny_config.embd_pdrop = model_args.model_embd_dropout
            transformer_tiny_config.attn_pdrop = model_args.model_embd_dropout
            self.codi = AutoModelForCausalLM.from_config(transformer_tiny_config) 
        elif model_args.full_precision:
            self.codi = model_wrapper_class.from_pretrained(
                    self.model_name,
                    torch_dtype=(
                        torch.float16 if training_args.bf16 is False else torch.bfloat16
                    ),
                    resume_download=True,
                )
        else:
            self.codi = model_wrapper_class.from_pretrained(
                    self.model_name,
                    torch_dtype=(
                        torch.float16 if training_args.bf16 is False else torch.bfloat16
                    ),
                    resume_download=True,
                    quantization_config=transformers.BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=False,
                        bnb_4bit_quant_type='nf4',
                    )
                )

        if "transformer_tiny" in model_args.model_name_or_path:
            self.training = self.model_args.train
            self.pad_token_id = POLYNOMIAL_TOKEN_DICT['PAD']
            self.bot_id = POLYNOMIAL_TOKEN_DICT['BoT']
            self.eot_id = POLYNOMIAL_TOKEN_DICT['EoT']
            self.tokenizer = POLYNOMIAL_TOKEN_DICT
        else: 
            ori_vocab_size = self.codi.config.vocab_size
            self.training = self.model_args.train

            # special tokens to enclose the latent embeddings
            self.pad_token_id = ori_vocab_size
            self.bot_id = ori_vocab_size + 1
            self.eot_id = ori_vocab_size + 2

            self.codi.resize_token_embeddings(
                ori_vocab_size + 3
            )  # dummy values for mem tokens
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False) 

        self.dim = self.codi.config.hidden_size
        self.num_latent = training_args.num_latent
       
        # LoRA
        if training_args.use_lora:
            self.codi = get_peft_model(self.codi, lora_config)

        # Projection Layer
        self.use_prj = training_args.use_prj
        self.prj_no_ln = training_args.prj_no_ln
        if training_args.use_prj:
            self.prj = nn.Sequential(
                nn.Dropout(training_args.prj_dropout),
                nn.Linear(self.dim, training_args.prj_dim),
                nn.GELU(),
                nn.Linear(training_args.prj_dim, self.dim),
            )
            if not self.prj_no_ln:
                self.prj.add_module("ln", nn.LayerNorm(self.dim))
                
        # Losses
        self.print_loss = training_args.print_loss
        self.ref_loss_factor = training_args.ref_loss_factor
        self.ce_loss_factor = training_args.ce_loss_factor

        # Cross Entropy Loss
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100) 
        
        # Distillation Loss
        self.distill_loss_div_std = training_args.distill_loss_div_std
        self.distill_loss_type = training_args.distill_loss_type
        self.distill_loss_factor = training_args.distill_loss_factor
        if self.distill_loss_type == "smooth_l1":
            self.distill_loss_fct = nn.SmoothL1Loss()
        elif self.distill_loss_type == "l2":
            self.distill_loss_fct = nn.MSELoss()
        else:
            raise NotImplementedError

        # general 
        self.fix_attn_mask = training_args.fix_attn_mask

        if "transformer_tiny" not in model_args.model_name_or_path and self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token_id = self.pad_token_id

        if self.training:
            self.init()

    def get_embd(self, model, model_name):
        try:
            if "pythia" in model_name:
                return model.get_base_model().gpt_neox.embed_in
            elif "gpt2" or "transformer_tiny" in model_name:
                try:
                    return model.get_base_model().transformer.wte
                except Exception: # no lora
                    return model.transformer.wte
            else:
                try:
                    return model.get_base_model().model.embed_tokens
                except Exception: # no lora
                    return model.model.embed_tokens
        except AttributeError:
            if "pythia" in model_name:
                return model.gpt_neox.embed_in
            raise NotImplementedError

    def init(self):
        print_trainable_parameters(self)
        if (
            self.training_args.restore_from is not None
            and self.training_args.restore_from != ""
        ):
            print(
                f"Loading from the pretrained checkpoint: {self.training_args.restore_from}..."
            )
            state_dict = load_file(self.training_args.restore_from)
            self.load_state_dict(state_dict)
            print(f"Finished loading from {self.training_args.restore_from}")

    def forward(
        self,
        encoder_input_ids: torch.LongTensor = None,
        decoder_input_ids: torch.LongTensor = None,
        ref_input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        ref_answer_position: Optional[torch.LongTensor] = None,
        model_answer_position: Optional[torch.LongTensor] = None,
        ref_attention_mask: Optional[torch.LongTensor] = None,
        ref_labels: torch.LongTensor = None,
        step: int = None,
        step_ratio: float = None
    ):
        if not self.fix_attn_mask:
            ref_attention_mask = None
        
        # Encode the question
        past_key_values = None
        outputs = self.codi(input_ids=encoder_input_ids, 
                            use_cache=True, 
                            output_hidden_states=True, 
                            past_key_values=past_key_values, 
                            attention_mask=encoder_attention_mask, #Debug 
                            ) 
        past_key_values = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1) # as the next input
        
        if self.use_prj:
            latent_embd = self.prj(latent_embd)


        len_pred_loss = 0
        dynamic_mask = None
        if self.fix_attn_mask:
            dynamic_mask = torch.ones((encoder_attention_mask.size(0), self.num_latent), device=ref_labels.device)

        # Iterate over the latent embeddings
        distill_loss_total = 0
        ce_loss_total = 0

        with torch.no_grad():
            ref_outputs = self.codi(input_ids=ref_input_ids, output_hidden_states=True, attention_mask=ref_attention_mask)
        ref_outputs_with_grad = self.codi(input_ids=ref_input_ids, output_hidden_states=True, attention_mask=ref_attention_mask) 
        
        # Formatting for deprecated exps
        ref_outputs_list = [ref_outputs] 
        ref_input_ids = [ref_input_ids] 

        # Process the position tensor
        # Normalise the position definition 
        if "llama" in self.model_name.lower() or "qwen" in self.model_name.lower(): # there is one more token standing for " " 
            model_answer_position = model_answer_position + 1
            ref_answer_position = ref_answer_position + 1
       
        # For DEBUG: Print the probability of the teacher task to predict the correct answer
        if self.training_args.print_ref_model_stats:
            for i, (ref_inputs, ref_outputs) in enumerate(zip(ref_input_ids, ref_outputs_list)):
                # evalutae the reference model
                if len(ref_outputs_list) > 1:
                    pos = ref_answer_position[i]
                else:
                    pos = ref_answer_position
                ref_probs = torch.nn.functional.softmax(ref_outputs.logits, dim=-1)
                input_positions = (pos-1).unsqueeze(1).unsqueeze(1).expand(-1, -1, ref_probs.size(2))
                ref_probs_at_positions = ref_probs.gather(1, input_positions)
                probe_positions_positions = pos.unsqueeze(1)
                probe_positions = ref_inputs.gather(1, probe_positions_positions).unsqueeze(1)
                ref_probs_of_target = ref_probs_at_positions.gather(2, probe_positions)
                
                # Calculate the accuracy of the predicted tokens 
                logits_at_positions = ref_outputs.logits.gather(1, input_positions)  # Reuse input_positions from line 353
                predicted_tokens = logits_at_positions.squeeze(1).argmax(dim=-1) 
                target_tokens = ref_inputs.gather(1, probe_positions_positions).squeeze(1)
                correct = (predicted_tokens == target_tokens).float() 
                teacher_accuracy = correct.mean() 
                print(f'stage{i}: accuracy of the predicted tokens: {teacher_accuracy}')  
                #print(f'predicted_tokens:', torch.sum(predicted_tokens == 62) / predicted_tokens.numel()) 
                print(f'stage{i}: mean of the prob of the target token: {ref_probs_of_target.mean()}')
                
        
        # the model answer position is the position of the eot token to predict the first token of the response
        model_answer_position = model_answer_position - 1
        ref_answer_position = ref_answer_position -1
      
        num_latent = self.num_latent
        if self.num_latent != 0:
            for i in range(num_latent):
                #breakpoint() 
                # Implicit CoT generation
                # debug 
                if i == 0: 
                    current_mask = torch.cat([encoder_attention_mask, torch.ones((encoder_attention_mask.size(0), 1), dtype=torch.bool).to(encoder_attention_mask.device)], dim=1)
                else: 
                    current_mask = torch.cat([current_mask, torch.ones((encoder_attention_mask.size(0), 1), dtype=torch.bool).to(encoder_attention_mask.device)], dim=1)
                outputs = self.codi(inputs_embeds=latent_embd, use_cache=True, output_hidden_states=True, past_key_values=past_key_values, attention_mask=current_mask)
                # outputs = self.codi(inputs_embeds=latent_embd, use_cache=True, output_hidden_states=True, past_key_values=past_key_values)
                past_key_values = outputs.past_key_values
                
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                
                if self.use_prj:
                    latent_embd = self.prj(latent_embd)
                
                # Calculate the distillation loss
                if i == num_latent - 1: # the last latent embedding
                    # Decode the final answer in natural language
                    
                    embds = self.get_embd(self.codi, self.model_name)(decoder_input_ids)
                    
          
                    if dynamic_mask is not None: # Prevent attending the paddings
                        decoder_mask = torch.ones((embds.size(0), embds.size(1)), dtype=torch.bool).to(dynamic_mask)
                        dynamic_mask = torch.cat((encoder_attention_mask, dynamic_mask, decoder_mask), dim=1)
                        dynamic_mask = dynamic_mask.bool()
                    # Student task's output
                    
                    assert dynamic_mask is not None, 'dynamic_mask has to exist' 
                    outputs = self.codi(inputs_embeds=embds, use_cache=True, output_hidden_states=True, past_key_values=past_key_values, attention_mask=dynamic_mask) 
                    # Teacher task's output

                    ref_outputs = ref_outputs_list[0]
                    
                    distill_loss = 0
                    # Calculate distillation loss between the teacher's logits and the student's logits for every layer
                    for j, (out, ref_out) in enumerate(zip(outputs.hidden_states, ref_outputs.hidden_states)):
                        # print(f'ref_answer_position.shape={ref_answer_position.shape}, ref_out.shape={ref_out.shape}')
                        ref_selected = ref_out.gather(1, ref_answer_position.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, ref_out.size(-1)))
                        out_selected = out.gather(1, model_answer_position.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, out.size(-1)))

                        distill_loss_tmp = self.distill_loss_fct(out_selected, ref_selected.detach())
                        
                        if self.distill_loss_div_std:
                            if self.distill_loss_type == 'l2':
                                distill_loss_tmp /= ref_selected.std()
                            distill_loss_tmp /= ref_selected.std()
                        distill_loss += distill_loss_tmp
                    
                    
                    distill_loss /= len(outputs.hidden_states)
                    
                    if self.print_loss:
                        print(f'latent{i}: distill_loss={distill_loss}')

                    distill_loss_total += distill_loss

                    # Calculate the CE loss for the student task
                    if i == num_latent - 1:
                        logits = outputs.logits
                        effective_logits = logits[:, :-1, :]
                        effective_logits = effective_logits.reshape(-1, logits.size(-1))
                        target_ids = labels[:, 1:].reshape(-1)                        
                        ce_loss = self.loss_fct(effective_logits, target_ids)
                        ce_loss_total += ce_loss

                        # For DEBUG: Print the probability of the student task to predict the correct answer
                        if self.training_args.print_ref_model_stats:
                            # Calculate probabilities from student logits
                            student_probs = torch.nn.functional.softmax(logits, dim=-1)
                            
                            # Use model_answer_position for the student task
                            student_input_positions = (model_answer_position).unsqueeze(1).unsqueeze(1).expand(-1, -1, student_probs.size(2))
                            student_probs_at_positions = student_probs.gather(1, student_input_positions)
                            
                            # Get the target tokens from labels at the answer positions
                            # Note: labels corresponds to decoder_input_ids shifted by 1
                            student_probe_positions_positions = (model_answer_position + 1).unsqueeze(1)
                            student_probe_positions = labels.gather(1, student_probe_positions_positions).unsqueeze(1)
                            student_probs_of_target = student_probs_at_positions.gather(2, student_probe_positions)
                            
                            # Calculate the accuracy of the predicted tokens
                            student_logits_at_positions = logits.gather(1, student_input_positions)
                            student_predicted_tokens = student_logits_at_positions.squeeze(1).argmax(dim=-1)
                            student_target_tokens = labels.gather(1, student_probe_positions_positions).squeeze(1)
                            student_correct = (student_predicted_tokens == student_target_tokens).float()
                            student_accuracy = student_correct.mean()

                            student_accuracy_by_groundtruth = {}
                            for gt_value in student_target_tokens.unique():
                                mask = (student_target_tokens == gt_value)
                                accuracy_for_gt = student_correct[mask].mean().item()
                                student_accuracy_by_groundtruth[gt_value.item()] = accuracy_for_gt
                            
                            print(f'STUDENT: accuracy of the predicted tokens: {student_accuracy}')
                            print(f'STUDENT: mean of the prob of the target token: {student_probs_of_target.mean()}')

        # Calculate the CE loss for the teacher task
        ref_ce_loss = 0
        ref_logits = ref_outputs_with_grad.logits
        effective_ref_logits = ref_logits[:, :-1, :]
        effective_ref_logits = effective_ref_logits.reshape(-1, ref_logits.size(-1))
        ref_target_ids = ref_labels[:, 1:].reshape(-1)
        ref_ce_loss = self.loss_fct(effective_ref_logits, ref_target_ids)
        ref_ce_loss *= self.ref_loss_factor 

        # Weigh the distillation loss
        distill_loss *= self.distill_loss_factor
        distill_loss_total *= self.distill_loss_factor
        ce_loss_total = ce_loss_total * self.ce_loss_factor  

        if self.print_loss:
            print(f'loss={ce_loss+distill_loss}, ce_loss={ce_loss}, distill_loss={distill_loss}, ce_loss_total={ce_loss_total}, distill_loss_total={distill_loss_total}, ref_ce_loss={ref_ce_loss}')
        
        # print(f'ce_loss_total={ce_loss_total}, distill_loss_total={distill_loss_total}, ref_ce_loss={ref_ce_loss}')
        loss = ce_loss_total + distill_loss_total + ref_ce_loss
        
        # Keep as tensors for DataParallel compatibility - don't use .item()
        if ce_loss_total != 0:
            ce_loss_total = ce_loss_total.detach()
        if distill_loss_total != 0:
            distill_loss_total = distill_loss_total.detach()
        if ref_ce_loss != 0:
            ref_ce_loss = ref_ce_loss.detach()
        
        if self.training_args.print_ref_model_stats:
            return {"loss": loss, "logits": logits, "ce_loss": ce_loss_total, "distill_loss": distill_loss_total, "ref_ce_loss": ref_ce_loss, "teacher_accuracy": teacher_accuracy, "student_accuracy": student_accuracy, "student_accuracy_by_groundtruth": student_accuracy_by_groundtruth}
        else: 
            return {"loss": loss, "logits": logits, "ce_loss": ce_loss_total, "distill_loss": distill_loss_total, "ref_ce_loss": ref_ce_loss}

