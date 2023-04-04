# -*- encoding: utf-8 -*-
'''
@create_time: 2023/04/03 10:44:15
@author: lichunyu
'''
import os
import sys

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertModel,
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    BertForSequenceClassification,
    Trainer,
    HfArgumentParser,
    TrainingArguments
)
import deepspeed

from focal_loss import FocalLoss


class DistModel(nn.Module):

    def __init__(self, ) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained("/home/lichunyu/pretrain_models/bert-base-chinese")
        self.config = config
        self.bert = AutoModel.from_config(config)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fn = FocalLoss(gamma=2, alpha=[1,5], reduction="mean")
            loss = loss_fn(logits, labels)
            # print(loss)
            return {"logits": logits, "loss": loss}
        return {"logits": logits}


class TextDataset(Dataset):

    def __init__(
            self,
            df,
            max_length=128
    ) -> None:
        super().__init__()
        self.text = df["EVENT_NAME"].tolist()
        self.label = df["EVENT_TYPE"].tolist()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("/home/lichunyu/pretrain_models/bert-base-chinese")

    def __getitem__(self, index):
        text = self.text[index]
        inputs = self.tokenizer(
                text,
                add_special_tokens = True,
                truncation='longest_first',
                max_length = self.max_length,
                padding = 'max_length',
                return_attention_mask = True,
                return_tensors = 'pt',
        )
        label = torch.tensor(self.label[index])
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = label
        return inputs

    def __len__(self):
        return len(self.text)



if __name__ == "__main__":
    # parser = HfArgumentParser((TrainingArguments, ))
    # if len(sys.argv) >= 2 and sys.argv[-1].endswith(".json"):
    #     training_args, = parser.parse_json_file(json_file=os.path.abspath(sys.argv[-1]), allow_extra_keys=True)
    # else:
    #     training_args, = parser.parse_args_into_dataclasses()
    df = pd.read_csv("/home/lichunyu/datasets/foodsafety_data/foodsafety_data.csv", sep="\t")
    train_dataset = TextDataset(df=df)
    # model = DistModel()
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset
    # )
    # if training_args.do_train:
    #     trainer.train()


    ...