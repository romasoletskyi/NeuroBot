import argparse
import json
import os
import pandas as pd

from typing import List


def read_databases(input_path: str, output_path: str):
    files = os.listdir(input_path)
    output_file = os.path.basename(os.path.normpath(output_path))
    databases = []

    for file in files:
        if file.endswith('.csv') and file != output_file:
            json_file = file.rstrip('.csv') + '.json'
            if json_file in files:
                database = pd.read_csv(os.path.join(input_path, file),
                                       header=None,
                                       names=['id', 'time', 'sender', 'fwd', 'reply', 'content'])
                with open(os.path.join(input_path, json_file), 'r') as f:
                    sender_names = json.load(f)
                databases.append(database.replace({'sender': sender_names}))

    return databases


def merge_two_databases(left: pd.DataFrame, right: pd.DataFrame):
    left = pd.merge(left, right, how='outer', on='id')
    columns = ['time', 'sender', 'fwd', 'reply', 'content']

    for column in columns:
        left[column + '_x'].fillna(left[column + '_y'], inplace=True)

    left.drop(columns=[column + '_y' for column in columns], inplace=True)
    left.rename(columns={column + '_x': column for column in columns}, inplace=True)

    return left


def merge_databases(databases: List[pd.DataFrame]):
    database = databases[0]

    for i in range(1, len(databases)):
        database = merge_two_databases(database, databases[i])

    return database


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()

    databases = read_databases(args.input_path, args.output_path)
    database = merge_databases(databases)

    database['time'] = pd.to_datetime(database['time'], dayfirst=True)
    database['content'] = database['content'].fillna(' ')
    database.to_csv(args.output_path, index=False)

    with open(args.output_path.rstrip('.csv') + '.json', 'w') as f:
        json.dump(list(database['sender'].unique()), f, ensure_ascii=False)


if __name__ == "__main__":
    main()
