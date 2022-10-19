import pandas as pd
import sys
import numpy as np # linear algebra
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast
from typing import Tuple



# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

class bert_preprocess:
    def __init__(self) -> None:
        pass

    def batch_encoder(self,
                      str1:str,
                      str2:str,
                      max_length:int
                      )->Tuple[list,list,list]:
        # import BERT-base pretrained model
        bert = AutoModel.from_pretrained('bert-base-uncased',return_dict=False)
        
        
        
      
        tokens_train = tokenizer.encode_plus(
            
            str(str1),
            str(str2),
            max_length = max_length,
            padding = "max_length",
            truncation=True
            )


        return tokens_train["input_ids"],tokens_train["token_type_ids"],tokens_train["attention_mask"]

    def create_tokenizer_list(
        self,
        df:pd.DataFrame,
        col1:str,
        col2:str,
        max_length:int)->Tuple[list,list]:
        input_ids = []
        attention_mask = []
        for val in df.index.tolist():
            id,_,att = batch_encoder(df.loc[val,col1],df.loc[val,col2],max_length)
            input_ids.append(id)
            attention_mask.append(att)
        
        return input_ids,attention_mask




