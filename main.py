import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from typesafe_sdk import AsyncTypeSafeClient, TypeSafeError

from src.typesafe_ai_langgraph.typesafe_ai_langgraph_workflow import build_workflow

ROOT = Path(__file__).resolve().parent


async def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Route mock emails with Jev and LangGraph.")
    parser.add_argument("--email-id", help="Run one example, e.g. email-01 (default: all 10).")
    args = parser.parse_args()
    examples = json.loads((ROOT / "data/mock_emails.json").read_text())
    if args.email_id:
        examples = [item for item in examples if item["email"]["id"] == args.email_id]
        if not examples:
            parser.error(f"Unknown email ID: {args.email_id}")
    if not os.getenv("TYPESAFE_API_KEY", "").strip():
        parser.error("Set TYPESAFE_API_KEY in .env or your environment.")

    model = os.getenv("TYPESAFE_DEFAULT_MODEL", "").strip() or "jev-1.12"
    async with AsyncTypeSafeClient(model=model, timeout=30.0) as client:
        workflow = build_workflow(client)
        matched = 0
        for example in examples:
            result = await workflow.ainvoke({"email": example["email"]})
            match = result["intent"] == example["expected_intent"]
            matched += match
            print(json.dumps({
                "id": result["email"]["id"],
                "subject": result["email"]["subject"],
                "intent": result["intent"],
                "expected_intent": example["expected_intent"],
                "match": match,
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
                "destination": result["destination"],
                "model": result["model"],
            }, indent=2), flush=True)
    print(f"\nMatched {matched}/{len(examples)} expected labels.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except TypeSafeError as error:
        raise SystemExit(f"TypeSafe request failed ({type(error).__name__}). Check credentials, model access, and connectivity.") from None
