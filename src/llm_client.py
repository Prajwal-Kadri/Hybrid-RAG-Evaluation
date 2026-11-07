"""All LLM calls go here, so the rest of the pipeline is provider-agnostic."""

from __future__ import annotations
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from functools import cache
from openai import OpenAI
import os

@cache 
def get_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    return model, tokenizer

@cache
def get_gateway_client():
    return OpenAI(
        api_key=os.environ["CMU_AI_GATEWAY_API_KEY"],
        base_url="https://ai-gateway.andrew.cmu.edu",
        max_retries=0,
        timeout=60.0,
    )

def call_llm(prompt: str, **kwargs) -> str:
    """Return the LLM's text response. kwargs includes model_name, temperature
    and seed from config["generation"]; don't hard-code the model."""
    model_name = kwargs.pop("model_name")
    temperature = kwargs.pop("temperature")
    seed = kwargs.get("seed",0)
    message = [{"role": "user", "content": prompt}]
    response = get_gateway_client().chat.completions.create(
        model=model_name,
        messages=message,
        temperature=temperature,
        max_completion_tokens=1024,
    )
    text = response.choices[0].message.content
    if not isinstance(text, str) or not text.strip():
        raise ValueError("LLM returned empty response")
    return text

    # raise NotImplementedError


def score_sequence_likelihood(
    prompt: str, completion: str, length_normalize: bool = True, model_name: str | None = None, **kwargs
) -> float:
    """Log-probability of `completion` given `prompt` (higher = more likely):
    mean per token if length_normalize, else the sum.

    Needs real token log-probs or logits (e.g. an open model via
    transformers); a self-rating does not count. This model may differ from
    the generation model. Load it once.
    """
    
    model, tokenizer = get_model_and_tokenizer(model_name)
    message = [{"role": "user", "content": prompt}]
    prompt_id = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt = True, return_dict = True, return_tensors="pt").to(model.device) #[1,P]
    completion_id = tokenizer(completion, add_special_tokens = False, return_tensors="pt")["input_ids"].to(model.device) #[1,C]
    input_id = torch.cat([prompt_id["input_ids"],completion_id],dim=1).to(model.device) #[1,P+C]
    inputs = {
        "input_ids": input_id,
    }
    with torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits #(1,seq_len,V)
        log_probs = torch.log_softmax(logits, dim=-1) #(1,seq_len,V)
        score = log_probs[:, :-1, :].gather(-1, inputs["input_ids"][:,1:].unsqueeze(-1)).squeeze(-1) #(1,seq_len-1)
        score = score[:, prompt_id["input_ids"].shape[1]-1:].sum() if not length_normalize else score[:, prompt_id["input_ids"].shape[1]-1:].mean()
    return score.item()
    # raise NotImplementedError
