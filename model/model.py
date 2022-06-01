from collections import deque
from typing import Union, Tuple, List

import params


class Model:
    def __init__(self):
        self.buffer = deque(maxlen=params.message_buffer_limit)

    def generate(self, max_len=params.message_max_len) -> Union[None, Tuple[float, str]]:
        pass

    def update_buffer(self, message):
        self.buffer.append(message)
