import torch 
import transformers 
import numpy as np 
from pathlib import Path 
from torch.utils.data import Dataset
from typing import Dict, Sequence
from dataclasses import dataclass
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt 
import pandas as pd 
import copy 
import csv 
from datetime import datetime 

import sys, importlib
from pathlib import Path
sys.path.insert(0, str(Path().resolve().parent))

from src.model_interp import (
    CODI,
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from src.data.data_processing import Polynomial_CODI, Polynomial 
from src.data.data_processing import TOKEN_DICT as POLYNOMIAL_TOKEN_DICT 


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

