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
import json
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

import params
from model.model import GPTModel


class TextDataset(Dataset):
    def __init__(self, tokens):
        self.tokens = tokens

    def __len__(self):
        return self.tokens['input_ids'].shape[1] - params.chunk_size + 1

    def __getitem__(self, idx):
        return {key: self.tokens[key][:, idx: idx + params.chunk_size] for key in self.tokens}


class FocusTextDataset(Dataset):
    def __init__(self, tokens, tokenizer):
        self.tokens = tokens
        self.indices = self.compute_indices(tokenizer)

    def compute_indices(self, tokenizer) -> torch.Tensor:
        new_tokens = json.load(open(params.names_path, 'r'))
        new_tokens = [tokenizer.vocab[token.lower()] for token in new_tokens]
        token_bound = min(new_tokens)
        focus_token = tokenizer.vocab[params.focus_sender.lower()]

        start_index = torch.arange(self.tokens['input_ids'].shape[1] - params.chunk_size)
        inner_index = torch.arange(params.chunk_size)

        batches: torch.Tensor = self.tokens['input_ids'].squeeze(0)[start_index[:, None] + inner_index[None, :]]
        where_last_sender = params.chunk_size - 2 - torch.argmax(
            (torch.flip(batches[:, :-1], [1]) >= token_bound).long(), dim=1)

        is_focus = (batches[:, -1] == focus_token) + (
                    batches[torch.arange(batches.shape[0]), where_last_sender] == focus_token)
        indices = torch.arange(len(is_focus))

        return indices[is_focus]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        focus_idx = self.indices[idx]
        item = {key: self.tokens[key][:, focus_idx: focus_idx + params.chunk_size] for key in self.tokens}
        item['labels'] = item['input_ids']
        return item


def collate_fn(examples):
    return {key: torch.cat([example[key] for example in examples], 0) for key in examples[0]}


def accuracy_metric(pred: torch.Tensor, ref: torch.Tensor):
    return (torch.sum(pred == ref) / torch.prod(torch.tensor(pred.shape))).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_path', type=str)
    parser.add_argument('output_model', type=str)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPTModel()
    model.model.to(device)

    train_text = open(args.corpus_path + '-train', 'r').read().lower()
    test_text = open(args.corpus_path + '-test', 'r').read().lower()
    train_dataset = FocusTextDataset(model.tokenizer(train_text, return_tensors='pt'), model.tokenizer)
    test_dataset = FocusTextDataset(model.tokenizer(test_text, return_tensors='pt'), model.tokenizer)

    epoch_number = 1
    optimizer = AdamW(model.model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, 1500,
                                                (len(train_dataset) * epoch_number) // params.lm_batch_size)

    for epoch in range(epoch_number):
        train_dataloader = DataLoader(train_dataset, batch_size=params.lm_batch_size,
                                      shuffle=True, drop_last=True, collate_fn=collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=params.lm_batch_size,
                                     shuffle=False, drop_last=False, collate_fn=collate_fn)

        train_len = len(train_dataloader)
        test_len = len(test_dataloader)
        train_iter_num = params.iter_num
        test_iter_num = (train_iter_num * test_len) // train_len

        train_dataloader = iter(train_dataloader)
        test_dataloader = iter(test_dataloader)

        output_path = os.path.join(args.output_model, str(epoch))
        if not os.path.isdir(output_path):
            os.mkdir(output_path)

        for i in range(train_len // train_iter_num):
            train_losses, test_losses = [], []
            train_accuracy, test_accuracy = [], []

            model.model.train()
            for _ in tqdm(range(train_iter_num)):
                batch = next(train_dataloader)
                optimizer.zero_grad()
                predictions = model.train_forward({key: batch[key].to(device) for key in batch})
                loss = predictions.loss
                train_losses.append(loss.item())
                accuracy = accuracy_metric(torch.argmax(predictions.logits[..., :-1, :], -1).detach().cpu(),
                                           batch['labels'][..., 1:])
                train_accuracy.append(accuracy)

                loss.backward()
                optimizer.step()
                scheduler.step()

            model.model.eval()
            for _ in tqdm(range(test_iter_num)):
                batch = next(test_dataloader)
                with torch.no_grad():
                    predictions = model.train_forward({key: batch[key].to(device) for key in batch})
                    loss = predictions.loss
                    test_losses.append(loss.item())
                    accuracy = accuracy_metric(torch.argmax(predictions.logits[..., :-1, :], -1).detach().cpu(),
                                               batch['labels'][..., 1:])
                    test_accuracy.append(accuracy)

            print(i, 'Train', np.mean(train_losses), np.mean(train_accuracy), 'Test', np.mean(test_losses),
                  np.mean(test_accuracy))

            model.model.save_pretrained(output_path)
            model.tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    main()
