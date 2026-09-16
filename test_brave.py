import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import asyncio

from veritascore.retriever.brave import BraveRetriever


async def main():
    retriever = BraveRetriever()

    results = await retriever.search(
        "current Prime Minister of Bangladesh August 2026",
        max_results=5,
    )

    print(f"\nFound {len(results)} results\n")

    for i, result in enumerate(results, 1):
        print("=" * 80)
        print(f"RESULT {i}")
        print(f"TITLE: {result.title}")
        print(f"URL:   {result.url}")
        print(f"SCORE: {result.relevance_score}")
        print(f"TEXT:  {result.snippet}")


if __name__ == "__main__":
    asyncio.run(main())
