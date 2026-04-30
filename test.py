import torch 
import transformers 
import os 
import numpy as np 
from pathlib import Path 
from torch.utils.data import Dataset
from typing import Dict, Sequence
from dataclasses import dataclass
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt 
import pandas as pd 
import csv 
from datetime import datetime 

from src.model import (
    CODI,
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from src.data.data_processing import Polynomial_CODI, Polynomial
from src.data.data_processing import TOKEN_DICT as POLYNOMIAL_TOKEN_DICT 
from src.test_helpers import  DataCollatorForSupervisedDataset


IGNORE_INDEX = -100

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


class SupervisedDataset(Dataset):

    def __init__(self, data_name, raw_data, tokenizer, bot, eot):
        super(SupervisedDataset, self).__init__()

        if "polynomial" in data_name:
            # Polynomial data processing
            self.data_name = data_name
            self.data_dict = preprocess_polynomial(raw_data, Debug=False)
            self.keys = list(self.data_dict.keys())
        else: 
            raise NotImplementedError

    def __len__(self):
        return len(self.data_dict["encoder_input_ids"])

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return {key: self.data_dict[key][i] for key in self.keys}

# Parse Arguments 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
training_args.print_ref_model_stats = True # for testing purposes 
assert training_args.print_ref_model_stats == True, "print_ref_model_stats must be True"


if "polynomial" in data_args.data_name:
# Data Loading 
    if data_args.extra_ans_token_in_ref_input:
        Polynomial_Gen = Polynomial_CODI(mod=data_args.polynomial_mod, save_dir=Path(f'./data/polynomial_data/seq_{data_args.total_seqs}/num_{data_args.n_data_per_seq}/mod_{data_args.polynomial_mod}/codi'), cot=True)
    else: 
        assert False, "extra_ans_token_in_ref_input must be True"
    poly_codi_test = Polynomial_Gen.load_data([i+1 for i in range(data_args.total_seqs)], 'test')
    poly_codi_train = Polynomial_Gen.load_data([i+1 for i in range(data_args.total_seqs)], 'train')
    codi_train = poly_codi_train
    codi_test = poly_codi_test

else:
    raise NotImplementedError(f"Dataset {data_args.data_name} is not supported.")

train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=codi_train, tokenizer=None, bot=None, eot=None)
test_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=codi_test, tokenizer=None, bot=None, eot=None)
data_collator = DataCollatorForSupervisedDataset(tokenizer=None, data_name=data_args.data_name)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=int(len(test_dataset)),  # adjust as needed
    shuffle=False,  # typically False for test data, True for training
    collate_fn=data_collator,  # this is the key - use your data collator here
    num_workers=0,  # set to 0 for debugging, can increase for performance
    pin_memory=True if torch.cuda.is_available() else False
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=int(len(train_dataset)),  # adjust as needed
    shuffle=False,  # typically False for test data, True for training
    collate_fn=data_collator,  # this is the key - use your data collator here
    num_workers=0,  # set to 0 for debugging, can increase for performance
    pin_memory=True if torch.cuda.is_available() else False
)


# Model Loading 
model = CODI(model_args, training_args, lora_config=None) # we are not using LoRA for polynomial dataset 
state_dict = torch.load(os.path.join(model_args.ckpt_dir, "pytorch_model.bin")) 
model.load_state_dict(state_dict, strict=False) 
model.eval() 

# Test 
def test_model(input_dataloader):

    model.to(device)
    teacher_acc = []
    student_acc = []
    batch = next(iter(input_dataloader))
    seq_step = int(len(batch['encoder_input_ids']) / data_args.total_seqs)

    for i in range(data_args.total_seqs):
        print(f"----------- Sequence {i+1}----------")
        _batch = {k: v[i*seq_step:(i+1)*seq_step] for k, v in batch.items()}

        student_acc_sub = [] 
        teacher_acc_sub = [] 
        for j in range(10): 
            sub_seq_step = seq_step // 10 
            sub_batch = {k: v[j*sub_seq_step:(j+1)*sub_seq_step].to(device) for k, v in _batch.items()}
            output = model.forward(**sub_batch) 
            output_cpu = {} 
            for k, v in output.items():
                if isinstance(v, torch.Tensor):
                    output_cpu[k] = v.detach().cpu()
                else:
                    output_cpu[k] = v
            
            teacher_acc_sub.append(output_cpu['teacher_accuracy'].item())
            student_acc_sub.append(output_cpu['student_accuracy'].item())

        teacher_acc.append(np.mean(teacher_acc_sub))
        student_acc.append(np.mean(student_acc_sub))

    _batch = {k: v.to('cpu') for k, v in _batch.items()}
    model.to('cpu')

    return teacher_acc, student_acc 

test_teacher_acc, test_student_acc = test_model(test_dataloader)
train_teacher_acc, train_student_acc = test_model(train_dataloader) 

# Save results to CSV
results_csv_path = os.path.join(model_args.ckpt_dir, "test_results.csv") 


# Prepare data for CSV
# If accuracies are lists per sequence, save all of them
results_data = {
    'sequence_id': list(range(1, len(test_teacher_acc) + 1)),
    'test_teacher_acc': test_teacher_acc,
    'test_student_acc': test_student_acc,
    'train_teacher_acc': train_teacher_acc,
    'train_student_acc': train_student_acc,
}

# Create DataFrame and save
df = pd.DataFrame(results_data)

# Add summary row with averages
summary_row = {
    'sequence_id': 'AVERAGE',
    'test_teacher_acc': np.mean(test_teacher_acc),
    'test_student_acc': np.mean(test_student_acc),
    'train_teacher_acc': np.mean(train_teacher_acc),
    'train_student_acc': np.mean(train_student_acc),
}
df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

# Save to CSV
df.to_csv(results_csv_path, index=False)
print(f"\n{'='*80}")
print(f"📊 Test Results Summary:")
print(f"{'='*80}")
print(f"  Test Teacher Acc:  {np.mean(test_teacher_acc):.4f}")
print(f"  Test Student Acc:  {np.mean(test_student_acc):.4f}")
print(f"  Train Teacher Acc: {np.mean(train_teacher_acc):.4f}")
print(f"  Train Student Acc: {np.mean(train_student_acc):.4f}")
print(f"{'='*80}")
print(f"✅ Results saved to: {results_csv_path}\n")
