import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()

    data = pd.read_csv(args.input_path)
    messages = [sender + ' ' + content for sender, content in zip(data['sender'], data['content'])]

    with open(args.output_path, 'w') as f:
        print(' '.join(messages), file=f)


if __name__ == "__main__":
    main()
