import argparse
import sys

import pandas as pd

from model.predict import predict_single


def main():
    parser = argparse.ArgumentParser(description="Batch disaster tweet prediction")
    parser.add_argument("--input", required=True, help="Input CSV with a text column")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        sys.exit(f"Input file not found: {args.input}")

    text_col = next((c for c in ["text", "tweet"] if c in df.columns), None)
    if text_col is None:
        sys.exit(f"Input CSV must have a 'text' or 'tweet' column. Found: {list(df.columns)}")

    print(f"Running predictions on {len(df)} rows...")
    results = [predict_single(str(t)) for t in df[text_col]]

    out = pd.DataFrame({
        "text": df[text_col],
        "label": [r["label"] for r in results],
        "score": [round(r["confidence"], 4) for r in results],
    })
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} predictions to {args.output}")


if __name__ == "__main__":
    main()
