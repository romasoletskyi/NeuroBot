import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()

    data = pd.read_csv(args.input_path)
    messages = [sender + ' ' + content for sender, content in zip(data['sender'], data['content'])]
    messages = messages[int(0.6 * len(messages)):]
    split_index = int(0.5 * len(messages))

    with open(args.output_path + '-train', 'w') as f:
        print(' '.join(messages[split_index:]), file=f)

    with open(args.output_path + '-test', 'w') as f:
        print(' '.join(messages[:split_index]), file=f)


if __name__ == "__main__":
    main()
