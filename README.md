# NeuroBot
Telegram chatbot which employs transfromer (from https://huggingface.co/) to imitate a person from a chat. Folders are
* preprocessing - set of scripts, which allows to process html export of chat (can be easily done in telegram itself) into a corpus to train on
* model - scripts to train neural network models
* bot - simple telegram bot, which uses models to chat
