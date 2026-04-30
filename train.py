# Modified from https://github.com/zhenyi4/codi/blob/main/train.py, which was initially modified from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
import copy
import logging
import os
import re
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import torch
import json
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback
from safetensors.torch import load_file
from tqdm import tqdm
from math import ceil
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from datasets import load_dataset
from functools import partial
import numpy as np 
from pathlib import Path
import csv
from datetime import datetime  

from src.model import (
    CODI,
    ModelArguments,
    DataArguments,
    TrainingArguments,
    freeze_model
)

from src.data.data_processing import (
    Polynomial_CODI, 
    Polynomial
)

from src.data.data_processing import TOKEN_DICT as POLYNOMIAL_TOKEN_DICT 

IGNORE_INDEX = -100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class EpochCheckpointCallback(TrainerCallback):
    """Callback to save model every N epochs with epoch number in the name"""
    def __init__(self, trainer, save_every_n_epochs=-1):
        self.trainer = trainer
        self.save_every_n_epochs = save_every_n_epochs
        self.last_saved_epoch = -1
    
    def on_epoch_end(self, args, state, control, **kwargs):
        # If save_every_n_epochs <= 0, disable this functionality
        if self.save_every_n_epochs <= 0:
            return control
            
        current_epoch = int(state.epoch)
        
        # Check if we should save (every N epochs)
        if current_epoch > 0 and current_epoch % self.save_every_n_epochs == 0:
            if current_epoch != self.last_saved_epoch:
                # Create checkpoint directory with epoch number
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{current_epoch}")
                
                print(f"\n{'='*80}")
                print(f"💾 Saving checkpoint at epoch {current_epoch}")
                print(f"📁 Location: {checkpoint_dir}")
                print(f"{'='*80}\n")
                
                # Directly call save_model with the custom directory
                self.trainer.save_model(output_dir=checkpoint_dir)
                
                self.last_saved_epoch = current_epoch
        
        return control


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize CSV logger
        self.csv_log_path = None
        if self.args.output_dir:
            os.makedirs(self.args.output_dir, exist_ok=True)
            self.csv_log_path = os.path.join(self.args.output_dir, "training_losses.csv")
            # Create CSV file with headers
            with open(self.csv_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'epoch', 'total_loss', 'ce_loss', 'distill_loss', 'ref_ce_loss', 'timestamp'])
            print(f"📝 Logging losses to: {self.csv_log_path}")
    
    def compute_loss(self, model, inputs, num_items_in_batch):
        # Extract the global step from the optimizer
        step = self.state.global_step

        # Get total training steps
        batch_size = self.args.per_device_train_batch_size
        gradient_accumulation_steps = self.args.gradient_accumulation_steps
        num_epochs = self.args.num_train_epochs
        dataset_size = len(self.train_dataset)

        effective_batch_size = batch_size * self.args.world_size * gradient_accumulation_steps
        total_steps = ceil(dataset_size / effective_batch_size) * num_epochs

        # Add the step information to the inputs dictionary
        inputs["step_ratio"] = step / total_steps
        inputs["step"] = step
        # Call the model's forward method
        outputs = model(**inputs)
        loss = outputs["loss"]
        #"ce_loss": ce_loss_total, "mse_loss": mse_loss_total, "ref_ce_loss": ref_ce_loss
        if step % self.args.logging_steps == 0:
            # Extract scalar values from tensors for logging
            # Handle both scalar tensors and vectors (from DataParallel gathering)
            def tensor_to_scalar(val):
                if isinstance(val, torch.Tensor):
                    return val.mean().item() if val.numel() > 1 else val.item()
                return val
            loss_val = tensor_to_scalar(outputs["loss"])
            ce_loss_val = tensor_to_scalar(outputs["ce_loss"])
            distill_loss_val = tensor_to_scalar(outputs["distill_loss"])
            ref_ce_loss_val = tensor_to_scalar(outputs["ref_ce_loss"])
            
            # Calculate current epoch
            current_epoch = step * effective_batch_size / dataset_size
            
            # Log to TensorBoard
            self.log({"loss": loss_val, "ce_loss": ce_loss_val, "distill_loss": distill_loss_val, "ref_ce_loss": ref_ce_loss_val})
            
            # Log to console in human-readable format
            print(f"\n{'='*80}")
            print(f"📊 Step {step:6d} | Epoch {current_epoch:6.2f} | Progress {step/total_steps*100:5.1f}%")
            print(f"{'='*80}")
            print(f"  Total Loss:       {loss_val:10.6f}")
            print(f"  CE Loss:          {ce_loss_val:10.6f}")
            print(f"  Distill Loss:     {distill_loss_val:10.6f}")
            print(f"  Ref CE Loss:      {ref_ce_loss_val:10.6f}")
            print(f"{'='*80}\n")
            
            # Log to CSV file
            if self.csv_log_path:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(self.csv_log_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([step, f"{current_epoch:.4f}", f"{loss_val:.6f}", 
                                   f"{ce_loss_val:.6f}", f"{distill_loss_val:.6f}", 
                                   f"{ref_ce_loss_val:.6f}", timestamp])
        
        return loss

    def log(self, logs, start_time=None):
        if self.state.global_step is not None:
            for k, v in logs.items():
                super().log({k: v})
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Override save_model to handle dict tokenizer for polynomial dataset"""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the model state_dict
        if self.args.should_save:
            # Get the model to save (unwrap if wrapped in DataParallel/DistributedDataParallel)
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            
            # Save the entire CODI model state dict
            state_dict = model_to_save.state_dict()
            save_path = os.path.join(output_dir, "pytorch_model.bin")
            torch.save(state_dict, save_path)
            print(f"✅ Saved model state dict to {save_path}")
            
            # Also save the underlying transformer model separately for easier loading
            if hasattr(model_to_save, 'codi'):
                codi_dir = os.path.join(output_dir, "codi_model")
                os.makedirs(codi_dir, exist_ok=True)
                
                # Save the config if available
                if hasattr(model_to_save.codi, 'config'):
                    model_to_save.codi.config.save_pretrained(codi_dir)
                
                # Save the codi model state
                codi_state = model_to_save.codi.state_dict()
                torch.save(codi_state, os.path.join(codi_dir, "pytorch_model.bin"))
                print(f"✅ Saved codi model to {codi_dir}")
            
        # Handle tokenizer saving - only if it has save_pretrained method
        # For polynomial dataset (dict tokenizer), save as JSON instead
        if hasattr(self.processing_class, 'save_pretrained'):
            self.processing_class.save_pretrained(output_dir)
            print(f"✅ Saved tokenizer to {output_dir}")
        elif isinstance(self.processing_class, dict):
            # For polynomial dataset, save the tokenizer dict as JSON
            tokenizer_path = os.path.join(output_dir, "tokenizer_dict.json")
            with open(tokenizer_path, 'w') as f:
                json.dump(self.processing_class, f, indent=2)
            print(f"✅ Saved tokenizer dictionary to {tokenizer_path}")
        
        # Save training arguments
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        print(f"✅ Saved training arguments to {output_dir}")

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=256,#training_args.model_max_length,
            truncation=True,
            return_attention_mask=False
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    segment = [sentence]
    if len(segment) > 1:
        pred_answer = segment[1]
        pred_answer = [s for s in re.findall(r'-?\d+\.?\d*', pred_answer)]
        if len(pred_answer) > 0:
            pred_answer = pred_answer[0]
        else:
            pred_answer = float(pred[-1])
    else:
        # use the last number as the answer
        pred_answer = float(pred[-1])

    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    ##########################
    #       Peft Model       #
    ##########################
    if model_args.lora_init:
        task_type = TaskType.CAUSAL_LM
        if any(name in model_args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon", "qwen"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["phi"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["gpt2"]):
            target_modules = ["c_attn", "c_proj", 'c_fc']
        elif any(name in model_args.model_name_or_path.lower() for name in ["transformer_tiny"]):
            target_modules = ["c_attn", "c_proj", 'c_fc'] # This is just a placeholder, not used in the transformer_tiny model
        else:
            raise ValueError(f"Only support LLAMA, Mistral, Falcon, Phi-2, Transformer-Tiny, but got {model_args.model_name_or_path}.")
        
        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
            init_lora_weights=True,
        )


    model = CODI(model_args, training_args, lora_config)


    if "transformer_tiny" in model_args.model_name_or_path.lower():
        tokenizer = POLYNOMIAL_TOKEN_DICT
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                token=model_args.token,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=False,
            )

        if tokenizer.pad_token_id is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.pad_token_id = model.pad_token_id
            if tokenizer.pad_token_id is None: # error handling
                tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    def get_answer_token_position(tokens, answer_prompts, tokenizer):
        #answer_prompt = torch.tensor([464, 3280, 318, 25])
        try:
            match_indices = (tokens.unfold(0, len(answer_prompts[0]), 1) == answer_prompts[0]).all(dim=1).nonzero(as_tuple=True)[0].item()
            answer_token_id = match_indices + len(answer_prompts[0])
            return answer_token_id
        except Exception:
            breakpoint()

    def preprocess_polynomial(
        poly_codi: np.ndarray,
        Debug: bool = False, 
    ) -> Dict:
        
        source_ids = [] 
        target_ids = [] 
        answer_ids = [] 
        ref_input_ids = [] 
        ref_labels = []
        ref_answer_positions = []
        model_answer_positions = []
        ref_eos_positions = []
        answer_eos_positions = []
        masked_answer_ids = []

        for i in range(len(poly_codi)):
            
            _EoI_pos = np.where(poly_codi[i] == POLYNOMIAL_TOKEN_DICT['EoI'])[0][0]
            _Ans_pos = np.where(poly_codi[i] == POLYNOMIAL_TOKEN_DICT['EoS'])[0][0] - 1
            source_id = torch.tensor(poly_codi[i][: _EoI_pos], dtype=torch.long)
            target_id = torch.tensor(poly_codi[i][_EoI_pos : _Ans_pos], dtype=torch.long)
            answer_id = torch.tensor(poly_codi[i][_Ans_pos :], dtype=torch.long) 
            ref_input_id = torch.tensor(poly_codi[i], dtype=torch.long) 
            ref_label = ref_input_id.clone() 
            ref_label[: _EoI_pos + 1] = -100 
            ref_label[_Ans_pos + 1:] = -100 
            source_id = torch.tensor(source_id.numpy().tolist() + [POLYNOMIAL_TOKEN_DICT['BoT']], dtype=torch.long)
            answer_id = torch.tensor([POLYNOMIAL_TOKEN_DICT['EoT']] + [POLYNOMIAL_TOKEN_DICT['ANS']] + answer_id.numpy().tolist(), dtype=torch.long)
            ref_answer_position = torch.where(ref_input_id == POLYNOMIAL_TOKEN_DICT['EoS'])[0][0] - 1  
            model_answer_position = torch.where(answer_id == POLYNOMIAL_TOKEN_DICT['EoS'])[0][0] - 1 
            ref_eos_position = torch.where(ref_input_id == POLYNOMIAL_TOKEN_DICT['EoS'])[0][0] 
            answer_eos_position = torch.where(answer_id == POLYNOMIAL_TOKEN_DICT['EoS'])[0][0] 

            #print('answer_id before:', answer_id)
            #print('answer_eos_position:', answer_eos_position)
            masked_answer_id = answer_id.clone()
            masked_answer_id[answer_eos_position:] = -100 # mask out the EoS token. 
            #print('answer_id after:', answer_id) 

            source_ids.append(source_id)
            target_ids.append(target_id)
            answer_ids.append(answer_id)
            ref_input_ids.append(ref_input_id)
            ref_labels.append(ref_label)
            ref_answer_positions.append(ref_answer_position)
            model_answer_positions.append(model_answer_position)
            ref_eos_positions.append(ref_eos_position)
            answer_eos_positions.append(answer_eos_position)
            masked_answer_ids.append(masked_answer_id) 

            if i % 500 == 0 and Debug: 
                print('--------------------------------i:', i)
                print('ref_input_id:', ref_input_id)
                print('ref_label:', ref_label)
                print('source_id:', source_id)
                print('target_id:', target_id)
                print('answer_id:', answer_id)
                print('masked_answer_id:', masked_answer_id)
                print('ref_answer_position:', ref_answer_position)
                print('model_answer_position:', model_answer_position)
                print('ref_eos_position:', ref_eos_position)
                print('answer_eos_position:', answer_eos_position)
        
        return dict(encoder_input_ids=source_ids, decoder_input_ids=answer_ids, ref_input_ids=ref_input_ids, labels=masked_answer_ids, \
                    ref_answer_position=ref_answer_positions, model_answer_position=model_answer_positions, \
                        ref_eos_position=ref_eos_positions, answer_eos_positions=answer_eos_positions, ref_labels=ref_labels)



    def preprocess(
        sources: Sequence[str], 
        targets: Sequence[str], 
        answers: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer, 
        bot_id: int,
        eot_id: int,
    ) -> Dict:
        print("Tokenizing inputs... This may take some time...")
        sources_id = _tokenize_fn(sources, tokenizer)["input_ids"]
        cot_id = _tokenize_fn(targets, tokenizer)["input_ids"]
        answers_id = _tokenize_fn(answers, tokenizer)["input_ids"]

        # add eos token to accomodate pretrained model's format
        if not training_args.remove_eos:
            sources_id = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in sources_id]
            cot_id = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in cot_id]
        answers_id = [torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long) for x in answers_id]

        if cot_id[0][0] == tokenizer.bos_token_id:
            cot_id = [x[1:] for x in cot_id]
            answers_id = [x[1:] for x in answers_id]

        ref_input_ids = [torch.cat([x, y, z]).to(torch.long) for x, y, z in zip(sources_id, cot_id, answers_id)]
        ref_labels = []
        for x, y in zip(ref_input_ids, sources_id):
            z = x.clone()
            z[:len(y)] = -100
            ref_labels.append(z)
        
        # add eot to source
        sources_id = [torch.tensor(x.numpy().tolist() + [bot_id], dtype=torch.long) for x in sources_id]
        # add eot and eos
        if training_args.remove_eos:
            answers_id = [torch.tensor([eot_id] + x.numpy().tolist(), dtype=torch.long) for x in answers_id]
        else:
            answers_id = [torch.tensor([eot_id, tokenizer.eos_token_id] + x.numpy().tolist(), dtype=torch.long) for x in answers_id]

        answer_prompts = [torch.tensor(tokenizer.encode("The answer is:")), torch.tensor(tokenizer.encode("The next step result is:"))]
        if answer_prompts[0][0] == tokenizer.bos_token_id: # remove the bos
            answer_prompts[0] = answer_prompts[0][1:]
            answer_prompts[1] = answer_prompts[1][1:]
        
        ref_answer_position = [get_answer_token_position(x, answer_prompts, tokenizer) for i, x in enumerate(ref_input_ids)]
        model_answer_position = [get_answer_token_position(x, answer_prompts, tokenizer) for x in answers_id]

        ref_eos_position = [len(x)-1 for x in ref_input_ids]
        model_eos_position = [len(x)-1 for x in answers_id]
        return dict(encoder_input_ids=sources_id, decoder_input_ids=answers_id, ref_input_ids=ref_input_ids, labels=answers_id, \
                    ref_answer_position=ref_answer_position, model_answer_position=model_answer_position, \
                        ref_eos_position=ref_eos_position, model_eos_position=model_eos_position, ref_labels=ref_labels)


    class SupervisedDataset(Dataset):
        QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"
        QUESTION_DA_PROMPT = "\nAnswer the above question. Answer the final number directly in one number.\n"
        def __init__(self, data_name, raw_data, tokenizer, bot, eot):
            super(SupervisedDataset, self).__init__()
            logging.warning("Formatting inputs...")

            if "polynomial" in data_name:
                # Polynomial data processing
                self.data_name = data_name
                self.data_dict = preprocess_polynomial(raw_data, Debug=False)
                self.keys = list(self.data_dict.keys())

            else: 
                assert False, f"Dataset {data_name} is not supported."


        def __len__(self):
            return len(self.data_dict["encoder_input_ids"])

        def __getitem__(self, i) -> Dict[str, torch.Tensor]:
            return {key: self.data_dict[key][i] for key in self.keys}

    @dataclass
    class DataCollatorForSupervisedDataset(object):
        """Collate examples for supervised fine-tuning."""
        tokenizer: transformers.PreTrainedTokenizer
        data_name: str = "default" 

        def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
            encoder_input_ids, decoder_input_ids, ref_input_ids, labels, ref_answer_position, model_answer_position, ref_labels= \
                tuple([instance[key] for instance in instances] for key in ("encoder_input_ids", "decoder_input_ids", "ref_input_ids", "labels", "ref_answer_position", "model_answer_position", "ref_labels"))
            
            # pad left
            reversed_input_ids = [seq.flip(0) for seq in encoder_input_ids]
            encoder_input_ids = torch.nn.utils.rnn.pad_sequence(reversed_input_ids, batch_first=True, padding_value= POLYNOMIAL_TOKEN_DICT['PAD'] if self.data_name in ["polynomial", "parity", "cumulative_sum", "sum_multiplication", "hyper_poly", "poly_affine", "poly_exp", "fsc"] else self.tokenizer.pad_token_id).flip(1)
            
            # pad
            ref_input_ids = torch.nn.utils.rnn.pad_sequence(ref_input_ids, batch_first=True, padding_value= POLYNOMIAL_TOKEN_DICT['PAD'] if self.data_name in ["polynomial", "parity", "cumulative_sum", "sum_multiplication", "hyper_poly", "poly_affine", "poly_exp", "fsc"] else  self.tokenizer.pad_token_id)
            ref_labels = torch.nn.utils.rnn.pad_sequence(ref_labels, batch_first=True, padding_value=IGNORE_INDEX) 

            decoder_input_ids = torch.nn.utils.rnn.pad_sequence(decoder_input_ids, batch_first=True, padding_value= POLYNOMIAL_TOKEN_DICT['PAD'] if self.data_name in ["polynomial", "parity", "cumulative_sum", "sum_multiplication", "hyper_poly", "poly_affine", "poly_exp", "fsc"] else self.tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
          
            return dict(
                encoder_input_ids=encoder_input_ids,
                decoder_input_ids=decoder_input_ids,
                ref_input_ids=ref_input_ids,
                labels=labels,
                encoder_attention_mask=encoder_input_ids.ne(POLYNOMIAL_TOKEN_DICT['PAD'] if self.data_name in ["polynomial", "parity", "cumulative_sum", "sum_multiplication", "hyper_poly", "poly_affine", "poly_exp", "fsc"] else self.tokenizer.pad_token_id),
                ref_answer_position=torch.tensor(ref_answer_position, dtype=torch.long),
                model_answer_position=torch.tensor(model_answer_position, dtype=torch.long),
                ref_attention_mask=ref_input_ids.ne(POLYNOMIAL_TOKEN_DICT['PAD'] if self.data_name in ["polynomial", "parity", "cumulative_sum", "sum_multiplication", "hyper_poly", "poly_affine", "poly_exp", "fsc"] else self.tokenizer.pad_token_id),
                ref_labels=ref_labels,
            )

    def make_supervised_data_module(tokenizer, data_args) -> Dict:
        """Make dataset and collator for supervised fine-tuning."""
        logging.warning("Downloading Data")

        # This implementation is only for polynomial data. 
        if "polynomial" in data_args.data_name:
            
            if data_args.extra_ans_token_in_ref_input:
                Polynomial_Gen = Polynomial_CODI(mod=data_args.polynomial_mod, save_dir=Path(f'./data/polynomial_data/seq_{data_args.total_seqs}/num_{data_args.n_data_per_seq}/mod_{data_args.polynomial_mod}/codi'), cot=True)
            else: 
                assert False, "extra_ans_token_in_ref_input must be True for polynomial data." 
            
            poly_codi = Polynomial_Gen.load_data([i+1 for i in range(data_args.total_seqs)], 'train') 
            
            train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=poly_codi, tokenizer=None, bot=None, eot=None)
            data_collator = DataCollatorForSupervisedDataset(tokenizer=None, data_name=data_args.data_name)
            return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


        else:
            raise NotImplementedError(f"Dataset {data_args.data_name} is not supported.")

    training_args.output_dir = os.path.join(
        training_args.output_dir,
        training_args.expt_name,
        model_args.model_name_or_path.split('/')[-1],
        f"ep_{int(training_args.num_train_epochs)}",
        f"lr_{training_args.learning_rate}",
        f"seed_{training_args.seed}",
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = CustomTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    # Add the callback after trainer is created
    epoch_checkpoint_callback = EpochCheckpointCallback(trainer=trainer, save_every_n_epochs=training_args.save_every_n_epochs)
    trainer.add_callback(epoch_checkpoint_callback) 

    trainer.train()

    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
