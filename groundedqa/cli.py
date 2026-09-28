import argparse


import pandas as pd


from .client import GroqClient
from .qa import ask as ask_question




def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)




def cmd_ask(args) -> None:
    df = load_dataset(args.data)
    client = GroqClient()
    result = ask_question(
        args.question, df, client,
        temperature=args.temperature, max_tokens=args.max_tokens,
    )
    print(result)
    print(client.tracker.summary())




def cmd_compare(args) -> None:
    """Runs the same question at temperature 0 and temperature 1, three times each."""
    df = load_dataset(args.data)
    client = GroqClient()
    print(f"Question: {args.question}\n")
    for temperature in (0, 1):
        print(f"--- Temperature {temperature} ---")
        for run in range(1, 4):
            result = ask_question(
                args.question, df, client,
                temperature=temperature, max_tokens=args.max_tokens,
            )
            print(f"Run {run}: {result}")
        print()
    print(client.tracker.summary())




def main() -> None:
    parser = argparse.ArgumentParser(prog="groundedqa")
    parser.add_argument("--data", default="mock_data.csv", help="Path to the CSV dataset")
    parser.add_argument("--max-tokens", dest="max_tokens", type=int, default=512)
    subparsers = parser.add_subparsers(dest="command", required=True)


    ask_parser = subparsers.add_parser("ask", help="Ask a single question")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--temperature", type=float, default=0.0)
    ask_parser.set_defaults(func=cmd_ask)


    compare_parser = subparsers.add_parser("compare", help="Compare temperature 0 vs 1, 3 runs each")
    compare_parser.add_argument("question")
    compare_parser.set_defaults(func=cmd_compare)


    args = parser.parse_args()
    args.func(args)




if __name__ == "__main__":
    main()


