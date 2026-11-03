"""Post sample products to /products/classify and summarise the results.

    python run_samples.py                       # all samples
    python run_samples.py --only s-008 s-011    # selected product ids
    python run_samples.py --url http://localhost:8000

Each run's full responses are saved to results/<timestamp>.json, so you can
compare prompt or model changes later. Products run one at a time on purpose:
Ollama would only queue parallel requests anyway.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import httpx

HERE = Path(__file__).parent


def short(path: str | None, width: int = 60) -> str:
    if not path:
        return "-"
    return path if len(path) <= width else "..." + path[-(width - 3):]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--file", type=Path, default=HERE / "sample_products.json")
    ap.add_argument("--only", nargs="*", help="product ids to run")
    args = ap.parse_args()

    samples = json.loads(args.file.read_text())
    if args.only:
        samples = [s for s in samples if s["product"]["product_id"] in args.only]

    results = []
    with httpx.Client(base_url=args.url, timeout=180) as client:
        for s in samples:
            pid = s["product"]["product_id"]
            started = time.perf_counter()
            try:
                res = client.post("/products/classify", json=s["product"])
                res.raise_for_status()
                body = res.json()
                error = None
            except httpx.HTTPError as e:
                body, error = None, f"{e!r} {getattr(getattr(e, 'response', None), 'text', '')[:300]}"
            secs = time.perf_counter() - started
            results.append({"sample": s, "response": body, "error": error, "seconds": round(secs, 1)})

            print(f"\n{pid}  [{s['case']}]  {s['product']['title']}")
            if error:
                print(f"  ERROR    {error}")
                continue
            steps = body["steps"]
            revisions = sum(1 for st in steps if st["role"] == "classifier") - 1
            invalid = sum(1 for st in steps if not st["valid_choice"])
            top = body["candidates"][0] if body["candidates"] else None
            print(f"  chosen   {short(body['final_category_path'])}")
            print(f"  expected {short(s['expected'])}")
            if top:
                print(f"  top hit  {short(top['full_path'])}  ({top['similarity']:.3f})")
            print(f"  conf {body['confidence']:.2f}  {body['decision']:<13}"
                  f"  steps {len(steps)}  revisions {revisions}  invalid {invalid}  {secs:.1f}s")

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    ok = [r for r in results if r["response"]]
    decisions = {}
    for r in ok:
        decisions[r["response"]["decision"]] = decisions.get(r["response"]["decision"], 0) + 1
    print(f"\n{len(ok)}/{len(results)} succeeded  "
          + "  ".join(f"{k}: {v}" for k, v in sorted(decisions.items()))
          + f"  avg {sum(r['seconds'] for r in results) / max(len(results), 1):.1f}s"
          + f"\nsaved {out}")


if __name__ == "__main__":
    main()
