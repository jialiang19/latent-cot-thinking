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
from transformer_lens import HookedTransformer, HookedTransformerConfig 
from transformer_lens.loading_from_pretrained import convert_gpt2_weights 
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache 

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
    extra_ans_token_in_ref_input: bool = field(default=False, metadata={"help": "Adding the extra answer token to the reference input or not."})


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
    ref_loss_factor: float = field(default=1.0, metadata={"help": "A multiplier of the distillation loss."})
    ce_loss_factor: float = field(default=1.0, metadata={"help": "A multiplier of the CE loss."}) 
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
            hooked_transformer_config = HookedTransformerConfig(
                n_layers=model_args.model_num_layers,
                n_heads=model_args.model_num_heads,
                d_model=128,
                d_head = 128 // model_args.model_num_heads,
                d_vocab=max(POLYNOMIAL_TOKEN_DICT.values()) + 1,
                n_ctx=128, 
                act_fn="gelu_new", 
                default_prepend_bos=False
            )
            self.codi = HookedTransformer(hooked_transformer_config) 
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

        if "transformer_tiny" in model_args.model_name_or_path:
            self.dim = self.codi.cfg.d_model
        else:
            self.dim = self.codi.config.hidden_size
        #self.dim = self.codi.config.hidden_size
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

            if "transformer_tiny" in self.model_name:
                # We need to convert the GPT-2 weights to the Transformer-lens weights for self.codi
                # and load the projection layer weights separately 

                state_dict = torch.load(self.training_args.restore_from)
                gpt2_state_dict = {} 
                prj_state_dict = {} 
                for key, value in state_dict.items(): 
                    if key.startswith("codi"): 
                        gpt2_state_dict[key[5:]] = value 
                    elif key.startswith("prj"): 
                        prj_state_dict[key] = value 

                transformer_tiny_config.n_layer = self.model_args.model_num_layers
                transformer_tiny_config.n_head = self.model_args.model_num_heads
                hf_codi = AutoModelForCausalLM.from_config(transformer_tiny_config) 
                hf_codi.load_state_dict(gpt2_state_dict)

                tl_gpt2_state_dict = convert_gpt2_weights(hf_codi, self.codi.cfg)
                tl_gpt2_state_dict = {f'codi.{key}': value for key, value in tl_gpt2_state_dict.items()} 
                tl_gpt2_state_dict.update(prj_state_dict) 
                missing_keys, unexpected_keys = self.load_state_dict(tl_gpt2_state_dict, strict=False)
                
                for key in missing_keys:
                    print(f"missing key: {key}")
                print(f"Unexpected keys: {unexpected_keys}")
                print(f"Finished loading from {self.training_args.restore_from}")

            else:
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
        step_ratio: float = None, 
        debug: bool = False, 
    ):
        
        interp_dict = {} 
        
        if not self.fix_attn_mask:
            ref_attention_mask = None
        
        # Encode the question 
        
        def make_fix_pos_embed_hook(attention_mask):
            """Factory function to create hook with access to attention_mask"""
            def fix_pos_embed_hook(pos_embed, hook):
                batch_size, seq_len, d_model = pos_embed.shape
                
                # Get absolute positional embeddings directly from W_pos
                # This matches HuggingFace GPT-2 behavior
                absolute_pos_embed = self.codi.pos_embed.W_pos[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
                
                return absolute_pos_embed
            return fix_pos_embed_hook
        
        
        past_key_values = HookedTransformerKeyValueCache.init_cache(
            self.codi.cfg, device=encoder_input_ids.device, batch_size=encoder_input_ids.size(0)) 
        
        with self.codi.hooks(fwd_hooks=[('hook_pos_embed', make_fix_pos_embed_hook(encoder_attention_mask))]):
            logits, cache = self.codi.run_with_cache(
                encoder_input_ids, 
                past_kv_cache=past_key_values, 
                attention_mask=encoder_attention_mask, 
                return_type="logits", 
                return_cache_object=True
            ) 

        outputs = {'logits': logits, 'hidden_states': cache}
        
        interp_dict['encoder_output'] = outputs
        
        if debug: 
            print('logits.shape:', logits.shape) 
            for key, value in cache.items():
                print(f'{key}.shape:', value.shape) 

        latent_embd = cache['ln_final.hook_normalized'][:, -1, :].unsqueeze(1) 

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
            #ref_outputs = self.codi(input_ids=ref_input_ids, output_hidden_states=True, attention_mask=ref_attention_mask)
            ref_outputs_logits, ref_outputs_cache = self.codi.run_with_cache(
                ref_input_ids, 
                attention_mask=ref_attention_mask, 
                return_type="logits", 
                return_cache_object=True) 
            ref_outputs = {'logits': ref_outputs_logits, 'hidden_states': ref_outputs_cache}

        #ref_outputs_with_grad = self.codi(input_ids=ref_input_ids, output_hidden_states=True, attention_mask=ref_attention_mask) 
        ref_outputs_with_grad_logits, ref_outputs_with_grad_cache = self.codi.run_with_cache(
            ref_input_ids, 
            attention_mask=ref_attention_mask, 
            return_type="logits", 
            return_cache_object=True) 
        ref_outputs_with_grad = {'logits': ref_outputs_with_grad_logits, 'hidden_states': ref_outputs_with_grad_cache}
        
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
                ref_probs = torch.nn.functional.softmax(ref_outputs['logits'], dim=-1)
                input_positions = (pos-1).unsqueeze(1).unsqueeze(1).expand(-1, -1, ref_probs.size(2))
                ref_probs_at_positions = ref_probs.gather(1, input_positions)
                probe_positions_positions = pos.unsqueeze(1)
                probe_positions = ref_inputs.gather(1, probe_positions_positions).unsqueeze(1)
                ref_probs_of_target = ref_probs_at_positions.gather(2, probe_positions)
                
                # Calculate the accuracy of the predicted tokens 
                logits_at_positions = ref_outputs['logits'].gather(1, input_positions)  # Reuse input_positions from line 353
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
                # Implicit CoT generation
                #outputs = self.codi(inputs_embeds=latent_embd, use_cache=True, output_hidden_states=True, past_key_values=past_key_values)
                if self.fix_attn_mask:
                    latent_token_mask = torch.ones((encoder_attention_mask.size(0), 1), 
                                          dtype=torch.bool, 
                                          device=encoder_attention_mask.device) 
                
                    if i == 0:
                        # First iteration: encoder + first latent
                        current_mask = torch.cat([encoder_attention_mask, latent_token_mask], dim=1)
                    else:
                        # Subsequent iterations: previous mask + new latent token
                        current_mask = torch.cat([current_mask, latent_token_mask], dim=1)
                    
                    full_attention_mask = past_key_values.append_attention_mask(latent_token_mask) 

                # debug for missed positional embeddings between HF and TL 
                pos_indice = past_key_values[0].past_keys.shape[1] 
                pos_embed = self.codi.pos_embed.W_pos[pos_indice].unsqueeze(0).unsqueeze(1).expand(latent_embd.size(0), -1, -1)
                latent_embd_with_pos = latent_embd + pos_embed 

                # IMPORTANT: Manually update the cache's attention mask
                # since start_at_layer=0 bypasses input_to_embed
                # past_key_values.previous_attention_mask = current_mask

                with self.codi.hooks(fwd_hooks=[('hook_pos_embed', make_fix_pos_embed_hook(current_mask))]):
                    logits, cache = self.codi.run_with_cache(
                        latent_embd_with_pos, 
                        start_at_layer=0, 
                        past_kv_cache=past_key_values, 
                        return_type="logits", 
                        return_cache_object=True, 
                        attention_mask=full_attention_mask) 
                
                outputs = {'logits': logits, 'hidden_states': cache}
                
                interp_dict[f'latent_output_{i}'] = outputs
                
                #past_key_values = outputs.past_key_values
                latent_embd = outputs['hidden_states']['ln_final.hook_normalized'][:, -1, :].unsqueeze(1)


                if self.use_prj:
                    latent_embd = self.prj(latent_embd)
                
                # Calculate the distillation loss
                if i == num_latent - 1: # the last latent embedding
                    # Decode the final answer in natural language
                    #embds = self.get_embd(self.codi, self.model_name)(decoder_input_ids) 

                    if self.fix_attn_mask:  # Changed from 'if dynamic_mask is not None'
                        # Create mask for decoder tokens
                        decoder_mask = torch.ones((decoder_input_ids.size(0), decoder_input_ids.size(1)), 
                                                dtype=torch.bool, 
                                                device=encoder_attention_mask.device)
                        # Concatenate: current_mask already has [encoder + all latent tokens]
                        final_mask = torch.cat([current_mask, decoder_mask], dim=1)
                    else:
                        final_mask = None
                    full_attention_mask = past_key_values.append_attention_mask(decoder_mask) 

                    # Student task's output
                    #outputs = self.codi(inputs_embeds=embds, use_cache=True, output_hidden_states=True, past_key_values=past_key_values, attention_mask=dynamic_mask) 
                    embds = self.codi.embed(decoder_input_ids) 

                                    # debug for missed positional embeddings between HF and TL 
                    pos_indice = past_key_values[0].past_keys.shape[1] 
                    pos_embed = self.codi.pos_embed.W_pos[pos_indice:pos_indice+decoder_input_ids.size(1)].unsqueeze(0).expand(latent_embd.size(0), -1, -1)
                    embds_with_pos = embds + pos_embed 

                    with self.codi.hooks(fwd_hooks=[('hook_pos_embed', make_fix_pos_embed_hook(final_mask))]):
                        logits, cache = self.codi.run_with_cache(
                            embds_with_pos, 
                            start_at_layer=0, 
                            past_kv_cache=past_key_values, 
                            return_type="logits", 
                            return_cache_object=True, 
                            attention_mask=full_attention_mask) 
                
                        outputs = {'logits': logits, 'hidden_states': cache}
                        
                        interp_dict[f'decoder_output'] = outputs

                    # Teacher task's output
                    ref_outputs = ref_outputs_list[0]
                    
                    distill_loss = 0
                    # Calculate distillation loss between the teacher's logits and the student's logits for every layer
                    for j, (out, ref_out) in enumerate(zip(outputs['hidden_states'], ref_outputs['hidden_states'])):
                        # print(f'ref_answer_position.shape={ref_answer_position.shape}, ref_out.shape={ref_out.shape}')
                        if out not in ['blocks.0.hook_resid_pre'] + [f'blocks.{i}.hook_resid_post' for i in range(self.codi.cfg.n_layers)]:
                            continue 
                        
                        out = outputs['hidden_states'][out] 
                        ref_out = ref_outputs['hidden_states'][ref_out] 
                        
                        try: 
                            ref_selected = ref_out.gather(1, ref_answer_position.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, ref_out.size(-1)))
                            out_selected = out.gather(1, model_answer_position.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, out.size(-1)))
                        except Exception as e:
                            print(f'Error: {e}')
                            breakpoint()
                            print(f'ref_answer_position.shape={ref_answer_position.shape}, ref_out.shape={ref_out.shape}')
                            print(f'model_answer_position.shape={model_answer_position.shape}, out.shape={out.shape}')
                            
                    #  I commented out the distill loss for now, because interpretability only needs forward pass. 
                    #  We are not using model_interp for training. 
                    #     distill_loss_tmp = self.distill_loss_fct(out_selected, ref_selected.detach())
                        
                    #     if self.distill_loss_div_std:
                    #         if self.distill_loss_type == 'l2':
                    #             distill_loss_tmp /= ref_selected.std()
                    #         distill_loss_tmp /= ref_selected.std()
                    #     distill_loss += distill_loss_tmp
                    
                    # distill_loss /= self.codi.cfg.n_layers
                    distill_loss = 0 

                    if self.print_loss:
                        print(f'latent{i}: distill_loss={distill_loss}')

                    distill_loss_total += distill_loss

                    # Calculate the CE loss for the student task
                    if i == num_latent - 1:
                        logits = outputs['logits']
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

                            interp_dict['student_correct'] = student_correct 

                            student_accuracy_by_groundtruth = {}
                            for gt_value in student_target_tokens.unique():
                                mask = (student_target_tokens == gt_value)
                                accuracy_for_gt = student_correct[mask].mean().item()
                                student_accuracy_by_groundtruth[gt_value.item()] = accuracy_for_gt
                            
                            print(f'STUDENT: accuracy of the predicted tokens: {student_accuracy}')
                            print(f'STUDENT: mean of the prob of the target token: {student_probs_of_target.mean()}')

        # Calculate the CE loss for the teacher task
        ref_ce_loss = 0
        ref_logits = ref_outputs_with_grad['logits']
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
            return {"loss": loss, "logits": logits, "ce_loss": ce_loss_total, "distill_loss": distill_loss_total, "ref_ce_loss": ref_ce_loss, "teacher_accuracy": teacher_accuracy, "student_accuracy": student_accuracy, "student_accuracy_by_groundtruth": student_accuracy_by_groundtruth, "interp_dict": interp_dict}
        else: 
            return {"loss": loss, "logits": logits, "ce_loss": ce_loss_total, "distill_loss": distill_loss_total, "ref_ce_loss": ref_ce_loss}

    
    def store_clean_run(self, encoder_input_ids, decoder_input_ids, encoder_attention_mask=None):
        """
        Run a complete clean forward pass through encoder, latents, and decoder.
        Store all intermediate activations and KV caches.
        
        Returns:
            dict with keys: 'encoder_cache', 'latent_caches', 'decoder_cache', 
                           'encoder_past_kv', 'latent_past_kvs', 'final_past_kv'
        """
        clean_run = {}
        
        # Encoder phase
        def make_fix_pos_embed_hook(attention_mask):
            def fix_pos_embed_hook(pos_embed, hook):
                batch_size, seq_len, d_model = pos_embed.shape
                absolute_pos_embed = self.codi.pos_embed.W_pos[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
                return absolute_pos_embed
            return fix_pos_embed_hook
        
        past_key_values = HookedTransformerKeyValueCache.init_cache(
            self.codi.cfg, device=encoder_input_ids.device, batch_size=encoder_input_ids.size(0))
        
        with torch.no_grad():
            with self.codi.hooks(fwd_hooks=[('hook_pos_embed', make_fix_pos_embed_hook(encoder_attention_mask))]):
                logits, cache = self.codi.run_with_cache(
                    encoder_input_ids,
                    past_kv_cache=past_key_values,
                    attention_mask=encoder_attention_mask,
                    return_type="logits",
                    return_cache_object=True
                )
        
        clean_run['encoder_cache'] = cache
        
        # Get latent embedding from encoder
        latent_embd = cache['ln_final.hook_normalized'][:, -1, :].unsqueeze(1)
        if self.use_prj:
            latent_embd = self.prj(latent_embd)
        
        # Latent iterations
        clean_run['latent_caches'] = []
        current_mask = encoder_attention_mask if self.fix_attn_mask else None
        
        for i in range(self.num_latent):
            if self.fix_attn_mask:
                latent_token_mask = torch.ones((encoder_attention_mask.size(0), 1),
                                              dtype=torch.bool,
                                              device=encoder_attention_mask.device)
                if i == 0:
                    current_mask = torch.cat([encoder_attention_mask, latent_token_mask], dim=1)
                else:
                    current_mask = torch.cat([current_mask, latent_token_mask], dim=1)
                full_attention_mask = past_key_values.append_attention_mask(latent_token_mask)
            
            pos_indice = past_key_values[0].past_keys.shape[1]
            pos_embed = self.codi.pos_embed.W_pos[pos_indice].unsqueeze(0).unsqueeze(1).expand(latent_embd.size(0), -1, -1)
            latent_embd_with_pos = latent_embd + pos_embed
            
            with torch.no_grad():
                with self.codi.hooks(fwd_hooks=[('hook_pos_embed', make_fix_pos_embed_hook(current_mask))]):
                    logits, cache = self.codi.run_with_cache(
                        latent_embd_with_pos,
                        start_at_layer=0,
                        past_kv_cache=past_key_values,
                        return_type="logits",
                        return_cache_object=True,
                        attention_mask=full_attention_mask if self.fix_attn_mask else None
                    )
            
            clean_run['latent_caches'].append(cache)
            
            latent_embd = cache['ln_final.hook_normalized'][:, -1, :].unsqueeze(1)
            if self.use_prj:
                latent_embd = self.prj(latent_embd)
        
        # Decoder phase
        if self.fix_attn_mask:
            decoder_mask = torch.ones((decoder_input_ids.size(0), decoder_input_ids.size(1)),
                                     dtype=torch.bool,
                                     device=encoder_input_ids.device)
            final_mask = torch.cat([current_mask, decoder_mask], dim=1)
            full_attention_mask = past_key_values.append_attention_mask(decoder_mask)
        else:
            final_mask = None
            full_attention_mask = None
        
        embds = self.codi.embed(decoder_input_ids)
        pos_indice = past_key_values[0].past_keys.shape[1]
        pos_embed = self.codi.pos_embed.W_pos[pos_indice:pos_indice+decoder_input_ids.size(1)].unsqueeze(0).expand(latent_embd.size(0), -1, -1)
        embds_with_pos = embds + pos_embed
        
        with torch.no_grad():
            with self.codi.hooks(fwd_hooks=[('hook_pos_embed', make_fix_pos_embed_hook(final_mask))]):
                logits, cache = self.codi.run_with_cache(
                    embds_with_pos,
                    start_at_layer=0,
                    past_kv_cache=past_key_values,
                    return_type="logits",
                    return_cache_object=True,
                    attention_mask=full_attention_mask
                )
        
        clean_run['decoder_cache'] = cache

        return logits, clean_run


    def activation_patch_forward(
        self,
        encoder_input_ids,
        decoder_input_ids,
        clean_run,
        patch_specs,  # List of dicts specifying what to patch
        encoder_attention_mask=None,
    ):
        """
        Run forward pass with activation patching at specified phases.
        
        Args:
            encoder_input_ids: Corrupted encoder input
            decoder_input_ids: Corrupted decoder input
            clean_run: Output from store_clean_run()
            patch_specs: List of dicts with keys:
                - 'phase': 'encoder', 'latent_0', 'latent_1', ..., or 'decoder'
                - 'component': e.g., 'blocks.0.attn.hook_z', 'blocks.1.hook_resid_post'
                - 'positions': Positions to patch (None = all)
                - 'heads': For attention components (None = all)
            encoder_attention_mask: Attention mask
            
        Returns:
            logits, full_interp_dict (with all caches from encoder, latents, decoder)
        """
        interp_dict = {}
        
        # Helper function for creating patching hooks
        def make_patching_hook(clean_cache, component, positions=None, heads=None):
            def patching_hook(activation, hook):
                clean_activation = clean_cache[component]
                if positions is not None:
                    if heads is not None and 'attn' in component:
                        activation[:, positions, heads, :] = clean_activation[:, positions, heads, :]
                    else:
                        activation[:, positions, :] = clean_activation[:, positions, :]
                else:
                    if heads is not None and 'attn' in component:
                        activation[:, :, heads, :] = clean_activation[:, :, heads, :]
                    else:
                        activation = clean_activation
                return activation
            return patching_hook
        
        # Organize patch specs by phase
        patch_by_phase = {
            'encoder': [],
            **{f'latent_{i}': [] for i in range(self.num_latent)},
            'decoder': []
        }
        for spec in patch_specs:
            if 'encoder' in spec['phase']: 
                patch_by_phase['encoder'].append(spec) 
            elif 'decoder_EoT' == spec['phase'] or 'decoder_ANS' == spec['phase']: 
                patch_by_phase['decoder'].append(spec)
            else: 
                patch_by_phase[spec['phase']].append(spec)
        
        # Helper for fix_pos_embed_hook
        def make_fix_pos_embed_hook(attention_mask):
            def fix_pos_embed_hook(pos_embed, hook):
                batch_size, seq_len, d_model = pos_embed.shape
                absolute_pos_embed = self.codi.pos_embed.W_pos[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
                return absolute_pos_embed
            return fix_pos_embed_hook
        
        # Encoder phase
        past_key_values = HookedTransformerKeyValueCache.init_cache(
            self.codi.cfg, device=encoder_input_ids.device, batch_size=encoder_input_ids.size(0))
        
        encoder_hooks = [(spec['component'], make_patching_hook(
            clean_run['encoder_cache'], 
            spec['component'],
            spec.get('positions'),
            spec.get('heads')
        )) for spec in patch_by_phase['encoder']]
        
        encoder_hooks.append(('hook_pos_embed', make_fix_pos_embed_hook(encoder_attention_mask)))
        
        with self.codi.hooks(fwd_hooks=encoder_hooks):
            logits, cache = self.codi.run_with_cache(
                encoder_input_ids,
                past_kv_cache=past_key_values,
                attention_mask=encoder_attention_mask,
                return_type="logits",
                return_cache_object=True
            )
        
        interp_dict['encoder_output'] = {'logits': logits, 'hidden_states': cache}
        
        # Get latent embedding
        latent_embd = cache['ln_final.hook_normalized'][:, -1, :].unsqueeze(1)
        if self.use_prj:
            latent_embd = self.prj(latent_embd)

        # Latent iterations
        current_mask = encoder_attention_mask if self.fix_attn_mask else None
        
        for i in range(self.num_latent):
            if self.fix_attn_mask:
                latent_token_mask = torch.ones((encoder_attention_mask.size(0), 1),
                                              dtype=torch.bool,
                                              device=encoder_attention_mask.device)
                if i == 0:
                    current_mask = torch.cat([encoder_attention_mask, latent_token_mask], dim=1)
                else:
                    current_mask = torch.cat([current_mask, latent_token_mask], dim=1)
                full_attention_mask = past_key_values.append_attention_mask(latent_token_mask)
            
            pos_indice = past_key_values[0].past_keys.shape[1]
            pos_embed = self.codi.pos_embed.W_pos[pos_indice].unsqueeze(0).unsqueeze(1).expand(latent_embd.size(0), -1, -1)
            latent_embd_with_pos = latent_embd + pos_embed
            
            # Build hooks for this latent iteration
            latent_hooks = [(spec['component'], make_patching_hook(
                clean_run['latent_caches'][i],
                spec['component'],
                spec.get('positions'),
                spec.get('heads')
            )) for spec in patch_by_phase[f'latent_{i}']]
            
            latent_hooks.append(('hook_pos_embed', make_fix_pos_embed_hook(current_mask)))
            
            with self.codi.hooks(fwd_hooks=latent_hooks):
                logits, cache = self.codi.run_with_cache(
                    latent_embd_with_pos,
                    start_at_layer=0,
                    past_kv_cache=past_key_values,
                    return_type="logits",
                    return_cache_object=True,
                    attention_mask=full_attention_mask if self.fix_attn_mask else None
                )
            
            interp_dict[f'latent_output_{i}'] = {'logits': logits, 'hidden_states': cache}
            
            latent_embd = cache['ln_final.hook_normalized'][:, -1, :].unsqueeze(1)
            if self.use_prj:
                latent_embd = self.prj(latent_embd)
        
        # Decoder phase
        if self.fix_attn_mask:
            decoder_mask = torch.ones((decoder_input_ids.size(0), decoder_input_ids.size(1)),
                                     dtype=torch.bool,
                                     device=encoder_input_ids.device)
            final_mask = torch.cat([current_mask, decoder_mask], dim=1)
            full_attention_mask = past_key_values.append_attention_mask(decoder_mask)
        else:
            final_mask = None
            full_attention_mask = None
        
        embds = self.codi.embed(decoder_input_ids)
        pos_indice = past_key_values[0].past_keys.shape[1]
        pos_embed = self.codi.pos_embed.W_pos[pos_indice:pos_indice+decoder_input_ids.size(1)].unsqueeze(0).expand(latent_embd.size(0), -1, -1)
        embds_with_pos = embds + pos_embed
        
        decoder_hooks = [(spec['component'], make_patching_hook(
            clean_run['decoder_cache'],
            spec['component'],
            spec.get('positions'),
            spec.get('heads')
        )) for spec in patch_by_phase['decoder']]
        
        decoder_hooks.append(('hook_pos_embed', make_fix_pos_embed_hook(final_mask)))
        
        with self.codi.hooks(fwd_hooks=decoder_hooks):
            logits, cache = self.codi.run_with_cache(
                embds_with_pos,
                start_at_layer=0,
                past_kv_cache=past_key_values,
                return_type="logits",
                return_cache_object=True,
                attention_mask=full_attention_mask
            )
        
        interp_dict['decoder_output'] = {'logits': logits, 'hidden_states': cache}
        

        return logits, interp_dict
    
    def systematic_latent_patching(
        self,
        clean_encoder_ids,
        clean_decoder_ids,
        corrupted_encoder_ids,
        corrupted_decoder_ids,
        metric_fn,
        encoder_attention_mask=None,
        components=None,
        input_seq_len=None, # the length of the input sequence without counting the padding tokens
    ):
        """
        Systematically patch each component in each phase (encoder, latents, decoder).
        
        Args:
            clean_encoder_ids, clean_decoder_ids: Clean inputs
            corrupted_encoder_ids, corrupted_decoder_ids: Corrupted inputs
            metric_fn: Function(logits, interp_dict) -> scalar
            components: List of components to test (None = all residual)
            
        Returns:
            Nested dict: {phase: {component: metric_difference}}
        """
        # Get clean run
        clean_logits,clean_run = self.store_clean_run(
            clean_encoder_ids, clean_decoder_ids, encoder_attention_mask
        )
        clean_metric = metric_fn(clean_logits, '')
        
        # Get corrupted baseline
        corrupted_logits, corrupted_dict = self.activation_patch_forward(
            corrupted_encoder_ids,
            corrupted_decoder_ids,
            clean_run,
            [],  # No patches
            encoder_attention_mask
        )
        baseline_metric = metric_fn(corrupted_logits, corrupted_dict)

        if torch.is_tensor(baseline_metric): 
            max_diff = (clean_metric - baseline_metric).item()
        else:
            max_diff = clean_metric - baseline_metric
        
        # Default components
        if components is None:
            components = ['blocks.0.hook_resid_pre'] + \
                        [f'blocks.{i}.hook_resid_post' for i in range(self.codi.cfg.n_layers)]
        
        results = {}
        
        # Test each phase
        if input_seq_len is None:
            phases = ['encoder'] + [f'latent_{i}' for i in range(self.num_latent)] + ['decoder_EoT', 'decoder_ANS']
        else: 
            phases = [] 
            for i in range(input_seq_len): 
                phases.append(f'encoder_{i}')
            phases += ['encoder_bot'] + [f'latent_{i}' for i in range(self.num_latent)] + ['decoder_EoT', 'decoder_ANS']
        
        if input_seq_len is not None:
            encoder_id_len = clean_encoder_ids.shape[1] 
            encoder_pos  = encoder_id_len - 1 - input_seq_len # starting position of the encoder tokens (first input rather than [POL] tokens)

        for phase in phases:
            results[phase] = {}
            for component in components:
                if 'encoder' in phase and input_seq_len is not None:
                    patch_spec = [{
                        'phase': phase,
                        'component': component,
                        'positions': encoder_pos,
                    }]
                elif phase == 'decoder_EoT':
                    patch_spec = [{
                        'phase': phase,
                        'component': component,
                        'positions': 0,  # EoT is at position 0
                    }]
                elif phase == 'decoder_ANS':
                    patch_spec = [{
                        'phase': phase,
                        'component': component,
                        'positions': 1,  # ANS is at position 1
                    }]    
                else:
                    patch_spec = [{
                        'phase': phase,
                        'component': component,
                    }]
                
                patched_logits, patched_dict = self.activation_patch_forward(
                    corrupted_encoder_ids,
                    corrupted_decoder_ids,
                    clean_run,
                    patch_spec,
                    encoder_attention_mask
                )
                
                patched_metric = metric_fn(patched_logits, patched_dict)
                results[phase][component] = {
                    'patched_metric': patched_metric.item() if torch.is_tensor(patched_metric) else patched_metric,
                    'baseline_metric': baseline_metric.item() if torch.is_tensor(baseline_metric) else baseline_metric,
                    'difference': (patched_metric - baseline_metric).item() if torch.is_tensor(patched_metric) else (patched_metric - baseline_metric),
                    'lift': (patched_metric - baseline_metric).item()/max_diff if torch.is_tensor(patched_metric) else (patched_metric - baseline_metric)/max_diff,
                }
            if 'encoder' in phase and input_seq_len is not None:
                encoder_pos += 1 

        return results
