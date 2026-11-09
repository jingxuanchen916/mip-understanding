"""Run one model on the data and append one JSON line per prompt.

    python run.py --model meta-llama/llama-4-scout
    python run.py --model Qwen/Qwen3-4B-Instruct-2507 --base-url http://localhost:8000/v1

Re-running the same command resumes: rows already in the output file are skipped.
"""

import argparse
import asyncio
import json
import os
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import urlparse

import datasets
import openai
import yaml

HERE = Path(__file__).resolve().parent
TASKS = ("arithmetic", "language", "ner", "news", "parity", "sentiment", "tweets", "wsd")
DATA = "jingxuanchen916/mip-understanding"  # on the Hugging Face Hub
OPENROUTER_URL = "https://openrouter.ai/api/v1"
MAX_ATTEMPTS = 3  # per call, as in the paper: a response that fails its check is retried
# Bad key, no credit, no access, unknown model: no retry or later prompt can succeed,
# so the run stops instead of recording every row as a failure.
FATAL_STATUS = {401, 402, 403, 404}


# --- Data -------------------------------------------------------------------

def load(tasks=None, source=DATA):
    """The rows, optionally only some tasks."""
    ds = datasets.load_dataset(source, split="test")
    if tasks:
        ds = ds.filter(lambda row: row["task"] in tasks)
    return ds


# --- Prompts ----------------------------------------------------------------

TEMPLATES = yaml.safe_load((HERE / "prompts.yaml").read_text(encoding="utf-8"))


def format_instances(instances):
    return "\n\n".join(f"Instance {i}: {text}" for i, text in enumerate(instances, 1))


def build_prompt(task, instances):
    """Template, a blank line, then "Instance i: <text>" blocks separated by blank lines."""
    return TEMPLATES[task].strip() + "\n\n" + format_instances(instances).strip()


# --- Answers ----------------------------------------------------------------
# The answer is the "answer" field of the last JSON object in the response; it is
# correct when it equals the gold count (or sum, for arithmetic) as a number.

def parse_answer(text):
    """The answer as a string, or None if no JSON object carries one."""
    if not text:
        return None
    candidates = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates += re.findall(r"\{[^{}]*\}", text, re.DOTALL)  # flat objects only
    candidates.append(text.strip())

    # Latest candidate first: models put the final answer at the end.
    for candidate in reversed(candidates):
        try:
            data = json.loads(candidate)
        except ValueError:
            continue
        if not isinstance(data, dict) or data.get("answer") is None:
            continue
        answer = str(data["answer"]).strip()
        if (answer.startswith('"') and answer.endswith('"')) or (
            answer.startswith("'") and answer.endswith("'")
        ):
            answer = answer[1:-1]
        return answer
    return None


def is_correct(prediction, answer):
    """Numeric equality: "12" matches "12.0", and "1,234" matches "1234"."""
    if prediction is None:
        return False
    try:
        return Decimal(prediction.replace(",", "").strip()) == Decimal(answer)
    except (InvalidOperation, ValueError):
        return False


# --- Model client -----------------------------------------------------------

class Client:
    """Any OpenAI-compatible endpoint: OpenRouter by default, or e.g. a local vLLM server."""

    def __init__(self, base_url=OPENROUTER_URL, api_key=None, timeout=1200, transient_retries=5):
        if api_key is None:
            # The OpenRouter key only ever goes to OpenRouter; other endpoints read
            # API_KEY, and a local vLLM server accepts any key.
            env = "OPENROUTER_API_KEY" if urlparse(base_url).hostname == "openrouter.ai" else "API_KEY"
            api_key = os.getenv(env) or "EMPTY"
        # The SDK's own retries are off: after a timeout it would silently resend a
        # long generation from scratch. Timeouts surface as failed attempts instead.
        self._client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key,
                                          timeout=timeout, max_retries=0)
        self.transient_retries = transient_retries

    async def complete(self, model, prompt, *, max_tokens=20000, temperature=0, extra_body=None):
        """One chat completion; rate limits, 5xx and dropped connections are retried with backoff."""
        for k in range(self.transient_retries + 1):
            try:
                resp = await self._client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                )
                break
            except openai.APITimeoutError:
                raise
            except (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError):
                if k == self.transient_retries:
                    raise
                await asyncio.sleep(min(5 * 2 ** k, 120))

        if not resp.choices:  # OpenRouter can return 200 with an error body
            raise RuntimeError(f"empty response: {(resp.model_extra or {}).get('error')}")
        choice = resp.choices[0]
        extra = choice.message.model_extra or {}
        return {
            "content": choice.message.content,
            "reasoning": extra.get("reasoning") or extra.get("reasoning_content"),
            "finish_reason": choice.finish_reason,
            # Which upstream served the call matters: on OpenRouter the provider
            # decides whether a model reasons when no reasoning setting is sent.
            "provider": (resp.model_extra or {}).get("provider"),
            "usage": resp.usage.model_dump() if resp.usage else None,
            "response_id": resp.id,
        }


# --- Running ----------------------------------------------------------------

def has_answer(text):
    return parse_answer(text) is not None


async def call(client, prompt, args):
    """Up to MAX_ATTEMPTS requests, until a response carries an answer."""
    attempts, response = [], None
    for _ in range(MAX_ATTEMPTS):
        try:
            response = await client.complete(args.model, prompt, max_tokens=args.max_tokens,
                                             temperature=args.temperature, extra_body=args.extra_body)
        except Exception as e:
            if isinstance(e, openai.APIStatusError) and e.status_code in FATAL_STATUS:
                raise
            attempts.append({"error": f"{type(e).__name__}: {e}"})
            if isinstance(e, openai.BadRequestError):  # e.g. prompt too long; retrying won't help
                break
            continue
        attempts.append({k: response[k] for k in ("finish_reason", "provider", "usage", "response_id")})
        if has_answer(response["content"]):
            return response, attempts
    return response, attempts  # the last response received, which had no answer


async def run_row(client, row, args):
    record = {key: row[key] for key in ("id", "task", "n_instances", "selection_seed", "target", "answer")}
    record["model"] = args.model
    response, attempts = await call(client, build_prompt(row["task"], row["instances"]), args)
    prediction = parse_answer(response["content"]) if response else None
    record.update(
        prediction=prediction,
        valid=prediction is not None,
        correct=is_correct(prediction, row["answer"]),
        n_attempts=len(attempts),
        # The last response received (the accepted one if any).
        content=response["content"] if response else None,
        reasoning=response["reasoning"] if response else None,
        attempts=attempts,
        request={"base_url": args.base_url, "max_tokens": args.max_tokens,
                 "temperature": args.temperature, "extra_body": args.extra_body},
    )
    return record


async def main(args):
    rows = load(args.task, source=args.data)
    out = args.out or Path("results") / (args.model.replace("/", "__") + ".jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = []
    if out.exists():
        text = out.read_text()
        for line in text.splitlines():
            try:
                done.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a line cut off when a run was killed: that row runs again
        if text and not text.endswith("\n"):
            with open(out, "a") as f:
                f.write("\n")
    done_ids = {record["id"] for record in done}
    todo = [row for row in rows if row["id"] not in done_ids]
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"{args.model} | {len(todo)} to run, {len(done_ids)} already in {out}")
    if not todo:
        return

    client = Client(base_url=args.base_url, timeout=args.timeout)
    try:  # a typo in the model name or a server that is not up would fail every row
        await client.complete(args.model, "Reply with OK.", max_tokens=args.max_tokens,
                              temperature=args.temperature, extra_body=args.extra_body)
    except openai.OpenAIError as e:
        raise SystemExit(f"{args.model} at {args.base_url} does not respond: {e}")

    semaphore = asyncio.Semaphore(args.concurrency)
    finished = 0
    start = time.time()

    async def worker(row):
        nonlocal finished
        async with semaphore:
            result = await run_row(client, row, args)
        with open(out, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        finished += 1
        if result["valid"]:
            status = "correct" if result["correct"] else "wrong"
        else:  # say why: the last attempt's error, or how the response ended
            last = result["attempts"][-1] if result["attempts"] else {}
            why = last.get("error") or f"no answer, finish_reason={last.get('finish_reason')}"
            status = f"INVALID ({why[:150]})"
        print(f"[{finished}/{len(todo)} {time.time() - start:.0f}s] {row['id']} "
              f"-> {result['prediction']!r} (truth {row['answer']!r}) {status}", flush=True)

    try:
        await asyncio.gather(*(worker(row) for row in todo))
    except openai.APIStatusError as e:
        raise SystemExit(f"Stopped: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", nargs="+", choices=TASKS, help="default: all eight")
    ap.add_argument("--limit", type=int, help="run at most this many rows (for a quick test)")
    ap.add_argument("--base-url", default=OPENROUTER_URL,
                    help="any OpenAI-compatible endpoint (default: OpenRouter, with OPENROUTER_API_KEY; "
                         "other endpoints read API_KEY)")
    ap.add_argument("--max-tokens", type=int, default=20000)
    ap.add_argument("--temperature", type=float, default=0)
    ap.add_argument("--extra-body", type=json.loads,
                    help='JSON passed through to the API, e.g. \'{"provider": {"only": ["parasail"]}}\'')
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=1200, help="seconds per request")
    ap.add_argument("--data", default=DATA, help="Hub id or local folder of the data")
    ap.add_argument("--out", type=Path, help="default: results/<model>.jsonl")
    asyncio.run(main(ap.parse_args()))
