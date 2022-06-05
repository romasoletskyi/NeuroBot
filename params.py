message_buffer_limit = 256
time_buffer_limit = 32
message_max_len = 32

len_embedding = 20
time_norm = 86400
message_time_norm = 240
time_net_batch_size = 64

chunk_size = 256
lm_batch_size = 16
iter_num = 1000
train_dropout = 0.3

dataset_path = 'data/telegram-full.csv'
names_path = 'data/telegram-full.json'
focus_sender = '[Николай Галлиулин]'
model_path = 'model/gpt'
