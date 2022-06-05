import argparse
import json
import time

from telethon import TelegramClient, events
from typing import Tuple

import params
from model.model import Model


def read_info() -> Tuple[int, str, int, int]:
    with open('bot/info.json', 'r') as file:
        credentials = json.load(file)

    return credentials['api_id'], credentials['api_hash'], credentials['chat_id'], credentials['user_id']


def main():
    verbose = True
    api_id, api_hash, chat_id, user_id = read_info()
    client = TelegramClient('anon', api_id, api_hash)
    model = Model(user_id)

    @client.on(events.NewMessage(chats=chat_id, incoming=True))
    async def chat_update(event):
        model.update_buffer(await event.message)

        reply = model.generate()
        if verbose:
            print(reply)
        if reply is not None:
            message_time, text = reply

            wait_time = message_time - time.time()
            if wait_time > 0:
                time.sleep(wait_time)

            await client.send_message(chat_id, text)

    client.start()

    async def initialize():
        messages = await client.get_messages(chat_id, params.message_buffer_limit)
        messages = messages[::-1]
        for message in messages:
            model.update_buffer(message)

    client.loop.run_until_complete(initialize())
    client.run_until_disconnected()


if __name__ == "__main__":
    main()
