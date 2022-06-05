import json
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Union, Tuple, List

import params


@dataclass
class Message:
    sender: str
    time: datetime
    text: str


class TimeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.len_emb = nn.Embedding(params.len_embedding, 4)

        with open(params.names_path, 'r') as f:
            self.senders = json.load(f)
        self.senders_to_int = {sender: i for i, sender in enumerate(self.senders)}
        self.senders_emb = nn.Embedding(len(self.senders), 8)

        self.encoder = nn.Sequential(
            nn.Linear(14 * params.time_buffer_limit, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, messages_list: List[List[Message]]) -> torch.Tensor:
        x = torch.stack([self.embed_state(messages) for messages in messages_list])
        x = self.encoder(x)

        x[:, 0] = torch.sigmoid(x[:, 0])
        x[:, 1] = F.relu(x[:, 1]) + 1e-3

        return x

    def embed_state(self, messages: List[Message]):
        time, relative_time = [], []

        for message in messages:
            message_time = message.time.time()
            message_timedelta = timedelta(hours=message_time.hour,
                                          minutes=message_time.minute,
                                          seconds=message_time.second)
            time.append(message_timedelta.total_seconds() / params.time_norm)
            relative_time.append((messages[-1].time - message.time).total_seconds() / params.time_norm)

        time = torch.tensor(np.array(time))[:, None]
        relative_time = torch.tensor(np.array(relative_time))[:, None]

        try:
            message_lens = np.array([len(message.text.split()) for message in messages])
        except AttributeError:
            print([message.text for message in messages])
            for message in messages:
                print(message.text.split(), len(message.text.split()))
            raise AttributeError

        message_lens = torch.tensor(np.clip(message_lens, 0, params.len_embedding - 1))
        message_lens = self.len_emb(message_lens)

        message_senders = torch.tensor(np.array([self.senders_to_int[message.sender] for message in messages]))
        message_senders = self.senders_emb(message_senders)

        embedding = torch.flatten(torch.cat([time, relative_time, message_lens, message_senders], dim=1))
        return embedding.float()


class CausalModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.add_names()

    def add_names(self):
        new_tokens = json.load(open(params.names_path, 'r'))
        new_tokens = [token.lower() for token in new_tokens]
        new_tokens = list(set(new_tokens) - set(self.tokenizer.vocab))
        self.tokenizer.add_tokens(new_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def train_forward(self, tokens):
        return self.model(**tokens)


class BertModel(CausalModel):
    def __init__(self):
        CausalModel.__init__(self, AutoModelForCausalLM.from_pretrained('cointegrated/rubert-tiny2', is_decoder=True),
                             AutoTokenizer.from_pretrained('cointegrated/rubert-tiny2'))


class GPTModel(CausalModel):
    def __init__(self, train_mode=True):
        if train_mode:
            config = AutoConfig.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')
            config.attn_pdrop = params.train_dropout
            config.resid_pdrop = params.train_dropout
            config.summary_first_dropout = params.train_dropout

            model = AutoModelForCausalLM.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2', config=config)
            tokenizer = AutoTokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')
        else:
            model = AutoModelForCausalLM.from_pretrained(params.model_path)
            tokenizer = AutoTokenizer.from_pretrained(params.model_path)

        CausalModel.__init__(self, model, tokenizer)


class Model:
    def __init__(self, user_id):
        self.buffer = deque(maxlen=params.message_buffer_limit)

        self.time_model = TimeNet()
        self.time_model.load_state_dict(torch.load(params.time_model_path))
        self.time_model.eval()

        self.text_model = GPTModel(train_mode=False)
        self.text_model.model.eval()

        self.user_id = user_id
        self.sleep = False
        self.sender_dict = json.load(open(params.senders_path, 'r'))

        new_tokens = json.load(open(params.names_path, 'r'))
        new_tokens = [self.text_model.tokenizer.vocab[token.lower()] for token in new_tokens]
        self.token_bound = min(new_tokens)
        self.focus_token = self.text_model.tokenizer.vocab[params.focus_sender.lower()]

    def generate(self, max_len=params.message_max_len) -> Union[None, Tuple[float, str]]:
        if self.sleep:
            return None

        prob, rate = self.time_model([list(self.buffer)[-params.time_buffer_limit:]])[0]
        if np.random.random() > prob.item():
            return None
        wait_time = time.time() + params.message_time_norm * np.random.exponential(1 / rate.item())

        previous_text = ' '.join(
            [message.sender + ' ' + message.text for message in self.buffer] + [params.focus_sender]).lower()
        input_ids = self.text_model.tokenizer(previous_text, return_tensors='pt')['input_ids'][:, -params.chunk_size:]

        output = self.text_model.model.generate(
            input_ids,
            do_sample=True,
            max_length=params.chunk_size + max_len,
            top_k=50,
            top_p=0.95,
            pad_token_id=self.text_model.tokenizer.eos_token_id
        )[0, params.chunk_size:]

        end = (output >= self.token_bound) & (output != self.focus_token)
        output = output[:torch.argmax(end.long())]
        reply = self.text_model.tokenizer.decode(output, skip_special_tokens=True)

        return wait_time, reply

    def update_buffer(self, message):
        sender = message.sender.first_name
        if message.sender.last_name:
            sender += ' ' + message.sender.last_name
        text = message.text

        if message.from_id.user_id == self.user_id and text == params.stop_word:
            self.sleep = True
        elif message.from_id.user_id == self.user_id and text == params.restart_word:
            self.sleep = False

        self.buffer.append(Message(self.sender_dict[sender] if sender in self.sender_dict else sender,
                                   message.date, text))
