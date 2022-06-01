import argparse
import json
import time

from telethon import TelegramClient, events
from typing import Tuple

import params
from model.model import Model


def read_info() -> Tuple[int, str, int]:
    with open('bot/info.json', 'r') as file:
        credentials = json.load(file)

    return credentials['api_id'], credentials['api_hash'], credentials['chat_id']


def main():
    api_id, api_hash, chat_id = read_info()
    client = TelegramClient('anon', api_id, api_hash)
    model = Model()

    @client.on(events.NewMessage(chats=chat_id))
    async def chat_update(event):
        model.update_buffer(await event.message)

        reply = model.generate()
        if reply is not None:
            wait_time, text = reply
            time.sleep(wait_time)
            await client.send_message(chat_id, text)

    client.start()

    messages = await client.get_messages(chat_id, params.message_buffer_limit)
    for message in messages:
        model.update_buffer(message)

    client.run_until_disconnected()


if __name__ == "__main__":
    main()
