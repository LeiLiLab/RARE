from __future__ import annotations
import asyncio
import logging
from typing import List, Sequence, Tuple, Dict, Any
import argparse
import json
import aiolimiter
import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# Pydantic response schema
class CoTResponse(BaseModel):
    cot_answer: str = Field(..., description="Full chain-of-thought reasoning.")
    answer:     str = Field(..., description="Short final answer for the user.")

# Prompt helpers
def format_prompt(prompt_pair: Tuple[str, str]) -> List[Dict[str, str]]:
    system_msg, user_msg = prompt_pair
    return [{"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg}]

def format_prompt_gemma(prompt_pair: Tuple[str, str]) -> List[Dict[str, str]]:
    """Gemma models want a single user message."""
    system_msg, user_msg = prompt_pair
    return [{"role": "user", "content": f"{system_msg}\n{user_msg}"}]

# OpenAI chat helper (with throttle)
async def _throttled_openai_chat_completion_acreate(
    *,
    client: AsyncOpenAI,
    model: str,
    messages: Sequence[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
    n: int,
    response_model: type[BaseModel] | None = None,
) -> Any:
    """Robust wrapper around client.beta.chat.completions.create with retries."""
    async with limiter:
        for attempt in range(3):
            try:
                return await client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    n=n,
                    response_format=response_model or {"type": "text"},
                )
            except openai.RateLimitError as e:
                logging.warning(f"Rate-limit ({e}); sleeping 60 s")
                await asyncio.sleep(60)
            except openai.APIConnectionError as e:
                logging.warning(f"API connection error ({e}); retrying in 10 s")
                await asyncio.sleep(10)
            except Exception as e:
                logging.warning(f"Attempt {attempt+1} failed: {e}")
                await asyncio.sleep(30)
        # All retries exhausted:
        logging.error("All retries exhausted - returning empty response")
        return {"choices": [{"message": {"content": ""}}]}

async def _generate_from_openai_chat_completion(
    *,
    client: AsyncOpenAI,
    messages_list: List[Sequence[Dict[str, str]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    requests_per_minute: int,
    n: int,
    response_model: type[BaseModel] | None,
) -> List[Any]:
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    tasks = [
        _throttled_openai_chat_completion_acreate(
            client=client,
            model=model,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
            n=n,
            response_model=response_model,
        )
        for msgs in messages_list
    ]
    return await asyncio.gather(*tasks)

def vllm_inference(
    client: AsyncOpenAI,
    prompts: List[Tuple[str, str]],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
    top_p: float = 1.0,
    requests_per_minute: int = 120,
    num_responses_per_prompt: int = 1,
    *,
    structured: bool = True,
) -> List[Dict[str, str]]:
    """
    Run inference against an OpenAI-compatible endpoint (vLLM or OpenAI):

    Returns a list of dicts like
        {"cot_answer": "...", "answer": "..."}
    """
    # Choose prompt formatter
    formatter = format_prompt_gemma if "gemma" in model.lower() else format_prompt
    messages_list = [formatter(p) for p in prompts]

    predictions = asyncio.run(
        _generate_from_openai_chat_completion(
            client=client,
            messages_list=messages_list,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            requests_per_minute=requests_per_minute,
            n=num_responses_per_prompt,
            response_model=CoTResponse if structured else None,
        )
    )

    # Normalize output: always list of dicts
    results: List[Dict[str, str]] = []
    for resp in predictions:
        try:
            choice = resp.choices[0]
            if structured and hasattr(choice.message, "parsed"):
                # SDK parsed a CoTResponse object
                results.append(choice.message.parsed.model_dump())
            else:
                # Fallback: free text in content
                results.append({"cot_answer": "", "answer": choice.message.content})
        except Exception as e:
            logging.warning(f"Unexpected response shape: {e}; raw={resp}")
            results.append({"cot_answer": "", "answer": ""})
    return results

# 5. Minimal CLI demo
if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="Demo vLLM structured inference.")
    parser.add_argument("--base_url", type=str, default="")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model name as served by vLLM")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    # Parse CLI prompts into (system, user) tuples
    prompt_tuples = []
    prompt_tuples.append(("", "Write introduction about yourself"))

    # Create client (no key needed for local vLLM)
    client = AsyncOpenAI(api_key="EMPTY", base_url=args.base_url)

    # Run inference
    out = vllm_inference(
        client=client,
        prompts=prompt_tuples,
        model=args.model,
        temperature=args.temperature,
        structured=True,
    )

    print(json.dumps(out, indent=2, ensure_ascii=False))
