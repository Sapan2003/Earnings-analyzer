sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
import os
from ingestion.embedder import embed_company, get_chroma_client, get_collection
from utils.logger import get_logger

logger = get_logger("preembed", "ingestion.log")

BLUE_CHIP_TICKERS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    # Finance
    "JPM", "BAC", "GS",
    # Healthcare
    "JNJ", "UNH",
    # Consumer
    "WMT", "KO", "PG",
    # Industrial
    "BA", "CAT"
]


def already_embedded(ticker: str) -> bool:
    """Check if ticker already has data in ChromaDB"""
    try:
        client = get_chroma_client()
        collection = get_collection(client)
        existing = collection.get(
            where={"ticker": ticker.upper()}
        )
        return len(existing["ids"]) > 0
    except Exception:
        return False


def preembed_all(quarters: int = 8):
    """
    Pre-embeds all blue chip companies.
    Skips companies already in ChromaDB.
    """
    print(f"Pre-embedding {len(BLUE_CHIP_TICKERS)} "
          f"blue chip companies...")
    print("=" * 60)

    success = []
    failed = []
    skipped = []

    for ticker in BLUE_CHIP_TICKERS:
        print(f"\nProcessing {ticker}...")

        # Skip if already embedded
        if already_embedded(ticker):
            print(f"  Skipping {ticker} — already in ChromaDB")
            skipped.append(ticker)
            continue

        try:
            total = embed_company(ticker, quarters=quarters)
            if total > 0:
                print(f"  {ticker}: {total} chunks embedded")
                success.append(ticker)
            else:
                print(f"  {ticker}: No chunks — check filing availability")
                failed.append(ticker)
        except Exception as e:
            print(f"  {ticker}: Failed — {e}")
            failed.append(ticker)

    # Summary
    print("\n" + "=" * 60)
    print("PRE-EMBEDDING COMPLETE")
    print("=" * 60)
    print(f"Successful: {len(success)} — {success}")
    print(f"Skipped:    {len(skipped)} — {skipped}")
    print(f"Failed:     {len(failed)} — {failed}")


if __name__ == "__main__":
    quarters = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    preembed_all(quarters=quarters)