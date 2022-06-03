import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Sequence, Tuple
from tqdm import tqdm

import params
from model.model import Message, TimeNet
from torch.utils.data import Dataset, DataLoader


class MessageDataset(Dataset):
    def __init__(self, data: pd.DataFrame, indices: Sequence[int]):
        super().__init__()
        self.data = data
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        message_index = self.indices[idx]
        next_message = self.data.iloc[message_index]
        next_message = Message(next_message['sender'], next_message['time'], next_message['content'])

        messages = self.data[message_index - params.time_buffer_limit: message_index]
        messages = [Message(sender, time, text) for sender, time, text in zip(messages['sender'],
                                                                              messages['time'],
                                                                              messages['content'])]
        return messages, next_message


def collate_fn(batch):
    return [x[0] for x in batch], [x[1] for x in batch]


def calculate_loss(pred: torch.Tensor, history_batch: Sequence[Sequence[Message]],
                   message_batch: Sequence[Message]) -> Tuple[torch.Tensor, torch.Tensor]:
    prob_pred, time_pred = pred[:, 0], pred[:, 1]

    is_focus_sender = torch.tensor(np.array([message.sender == params.focus_sender for message in message_batch]))
    prob_loss = F.binary_cross_entropy(prob_pred, is_focus_sender.float())

    time_deltas = [message.time - history[-1].time for history, message in zip(history_batch, message_batch)]
    time_deltas = torch.tensor(np.array([t.total_seconds() / params.message_time_norm for t in time_deltas]))
    time_loss = -torch.mean(torch.log(time_pred) - time_pred * time_deltas)

    return prob_loss, time_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('model_path', type=str)
    args = parser.parse_args()

    data = pd.read_csv(args.data_path, parse_dates=['time'])

    indices = data.index[data.index >= params.time_buffer_limit]
    shuffle_indices = np.arange(len(indices))
    np.random.shuffle(shuffle_indices)
    train_indices = indices[shuffle_indices[:int(0.8 * len(indices))]]
    test_indices = indices[shuffle_indices[int(0.8 * len(indices)):]]

    train_dataset = MessageDataset(data, train_indices)
    train_dataloader = DataLoader(train_dataset, batch_size=params.time_net_batch_size,
                                  shuffle=True, drop_last=True, collate_fn=collate_fn)
    test_dataset = MessageDataset(data, test_indices)
    test_dataloader = DataLoader(test_dataset, batch_size=params.time_net_batch_size,
                                 shuffle=False, drop_last=False, collate_fn=collate_fn)

    model = TimeNet()
    optimizer = torch.optim.Adam(model.parameters())

    for epochs in range(10):
        prob_losses, time_losses = [], []

        for history_batch, message_batch in tqdm(train_dataloader):
            optimizer.zero_grad()

            pred = model(history_batch)
            prob_loss, time_loss = calculate_loss(pred, history_batch, message_batch)
            loss = prob_loss + time_loss
            prob_losses.append(prob_loss.item())
            time_losses.append(time_loss.item())

            loss.backward()
            optimizer.step()

        avg_train_prob_loss = np.mean(prob_losses)
        avg_train_time_loss = np.mean(time_losses)
        prob_losses, time_losses = [], []

        for history_batch, message_batch in tqdm(test_dataloader):
            with torch.no_grad():
                pred = model(history_batch)
                prob_loss, time_loss = calculate_loss(pred, history_batch, message_batch)
                prob_losses.append(prob_loss.item())
                time_losses.append(time_loss.item())

        avg_test_prob_loss = np.mean(prob_losses)
        avg_test_time_loss = np.mean(time_losses)
        print('Train loss:', 'prob', avg_train_prob_loss, 'time', avg_train_time_loss,
              'Test loss:', 'prob', avg_test_prob_loss, 'time', avg_test_time_loss)

        torch.save(model.state_dict(), args.model_path)


if __name__ == "__main__":
    main()
