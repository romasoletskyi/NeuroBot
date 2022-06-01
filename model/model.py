import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
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
            nn.Linear(14 * params.message_buffer_limit, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, messages: List[Message]):
        x = self.embed_state(messages)
        x = self.encoder(x)

        x[0] = F.sigmoid(x[0])
        x[1] = F.relu(x[1])

        return x

    def embed_state(self, messages: List[Message]):
        time, relative_time = [], []

        for message in messages:
            message_time = message.time.time()
            message_timedelta = timedelta(hours=message_time.hour,
                                          minutes=message_time.minute,
                                          seconds=message_time.second)
            time.append(message_timedelta.total_seconds() / params.time_norm)
            relative_time.append((messages[-1].time - message.time) / params.time_norm)

        time = torch.tensor(np.array(time))[:, None]
        relative_time = torch.tensor(np.array(time))[:, None]

        message_lens = np.array([len(message.text.split()) for message in messages])
        message_lens = torch.tensor(np.clip(message_lens, 1, params.len_embedding) - 1)
        message_lens = self.len_emb(message_lens)

        message_senders = torch.tensor(np.array([self.senders_to_int[message.sender] for message in messages]))
        message_senders = self.senders_emb(message_senders)

        embedding = torch.cat([time, relative_time, message_lens, message_senders])
        return embedding


class Model:
    def __init__(self):
        self.buffer = deque(maxlen=params.message_buffer_limit)

    def generate(self, max_len=params.message_max_len) -> Union[None, Tuple[float, str]]:
        pass

    def update_buffer(self, message):
        self.buffer.append(message)
