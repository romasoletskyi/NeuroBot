# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

import params
from model.model import BertModel


class TextDataset(Dataset):
    def __init__(self, tokens):
        self.tokens = tokens

    def __len__(self):
        return self.tokens['input_ids'].shape[1] - params.chunk_size + 1

    def __getitem__(self, idx):
        return {key: self.tokens[key][:, idx: idx + params.chunk_size] for key in self.tokens}


def collate_fn(examples):
    return {key: torch.cat([example[key] for example in examples], 0) for key in examples[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_path', type=str)
    parser.add_argument('output_model', type=str)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BertModel()
    model.model.to(device)

    train_text = open(args.corpus_path + '-train', 'r').read()
    test_text = open(args.corpus_path + '-test', 'r').read()
    train_dataset = TextDataset(model.tokenizer(train_text, return_tensors='pt'))
    test_dataset = TextDataset(model.tokenizer(test_text, return_tensors='pt'))
    train_dataloader = DataLoader(train_dataset, batch_size=params.lm_batch_size,
                                  shuffle=True, drop_last=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=params.lm_batch_size,
                                 shuffle=False, drop_last=False, collate_fn=collate_fn)

    optimizer = AdamW(model.model.parameters(), lr=5e-5)

    for epoch in range(100):
        train_losses, test_losses = [], []

        print('Train')
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            loss = model.train_forward({key: batch[key].to(device) for key in batch}).loss
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

            if len(train_losses) == 1000:
                print(np.mean(train_losses))
                train_losses = []

        print(np.mean(train_losses))

        print('Test')
        for batch in tqdm(test_dataloader):
            with torch.no_grad():
                loss = model.train_forward({key: batch[key].to(device) for key in batch}).loss
                test_losses.append(loss.item())

            if len(test_losses) == 1000:
                print(np.mean(test_losses))
                test_losses = []

        print(np.mean(test_losses))
        model.model.save_pretrained(args.output_model)


if __name__ == "__main__":
    main()
