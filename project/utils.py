import copy
import requests
import time
import os
from loguru import logger
import openai
from groq import Groq
import google.generativeai as google_genai
import json
import random
from litellm import completion
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re


from dotenv import load_dotenv
load_dotenv()


def generate(model, messages, max_tokens=2048, temperature=0.7, **kwargs):
    if "/" in model:
        if "R1" in model:
            response = generate_together(messages=messages, model=model, temperature=temperature,
                                         max_tokens=50000)
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
            response = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL)
        else:
            response = generate_together(messages=messages, model=model, max_tokens=max_tokens, temperature=temperature)
    else:
        response = generate_openai(messages=messages, model=model, temperature=temperature)
    return response


def generate_together(model, messages, max_tokens=2048, temperature=0.7, **kwargs):
    output = None
    request_id = random.randint(1000, 9999)  # Generate unique request ID for tracking

    key = os.getenv("TOGETHER_API_KEY")


    logger.info(f"[Together-{request_id}] Starting request for model: {model}")

    for attempt, sleep_time in enumerate([1, 2, 4, 8, 16, 32], 1):
        res = None
        try:

            endpoint = "https://api.together.xyz/v1/chat/completions"
            time.sleep(2)

            # logger.info(f"[Together-{request_id}] Attempt {attempt}: Sending request...")
            res = requests.post(
                endpoint,
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": (temperature if temperature > 1e-4 else 0),
                    "messages": messages,
                },
                headers={
                    "Authorization": f"Bearer {key}",
                },
            )


            output = res.json()["choices"][0]["message"]["content"]
            # logger.info(f"[Together-{request_id}] Successfully received response")
            break

        except Exception as e:
            response = "failed before response" if res is None else res
            logger.error(f"[Together-{request_id}] {e} on response: {response}")
            # logger.info(f"[Together-{request_id}] Retrying in {sleep_time}s...")
            time.sleep(sleep_time)

    if output is None:
        logger.error(f"[Together-{request_id}] Failed to get response after all attempts")
        return output

    return output.strip()


def generate_openai(model, messages, max_tokens=2048, temperature=0.7, **kwargs):

    key = os.getenv("OPENAI_API_KEY")

    client = openai.OpenAI(api_key=key)

    if model in ["o1-preview-2024-09-12", "o1-mini-2024-09-12"]:
        messages = [msg for msg in messages if msg["role"] != "system"]

    for sleep_time in [1, 2, 4, 8, 16, 32, 64]:
        try:

            if model in ["o1-preview-2024-09-12", "o1-mini-2024-09-12"]:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
            else:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            output = completion.choices[0].message.content
            break

        except Exception as e:
            logger.error(e)
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    output = output.strip()

    return output


def clean_text(text):
    return text.strip().replace(' ', '').lower()

'''
def extract_text_within_box(text):
    text = text.split("boxed{")[-1]
    stack = []  # To handle nested curly braces
    result = []  # Temporary storage to accumulate characters inside braces

    for char in text:
        if char == '{':
            stack.append('{')

        if char == '}':
            if stack:  # If the stack is not empty, it's a nested brace, so pop from the stack
                stack.pop()
            else:
                return clean_text(''.join(result))

        result.append(char)'
'''


def extract_text_within_box(text):
    if not text:
        return None
        
    # Match different variations of boxed notation
    patterns = [
        r'\\boxed{([^{}]*(?:{[^{}]*}[^{}]*)*)}',  # LaTeX style with escape
        r'\\boxed\s*{([^{}]*(?:{[^{}]*}[^{}]*)*)}',  # With possible space
        r'boxed{([^{}]*(?:{[^{}]*}[^{}]*)*)}',  # Without escape
        r'\\boxed\[(.*?)\]',  # Alternative bracket style
        r'boxed\[(.*?)\]',    # Alternative without escape
        r'Answer:\s*([0-9]+)',  # Plain "Answer: X" format
        r'The\s+(?:final\s+)?answer\s+is\s+([0-9]+|[0-9]+/[0-9]+)',  # "The answer is X" format
        r'The\s+(?:final\s+)?answer\s+is\s+([^\.\n]+)'  # More general "The answer is X" format
    ]
    
    for pattern in patterns:
        matches = re.search(pattern, text, re.DOTALL)
        if matches:
            return clean_text(matches.group(1))
    
    # If no matches with above patterns, try to extract from the final part of the text
    # Often answers appear at the end after "therefore" or similar
    final_sentences = text.split('.')[-3:]  # Look at last few sentences
    for sentence in final_sentences:
        # Look for numbers or fractions in the final sentences
        number_match = re.search(r'([0-9]+|[0-9]+/[0-9]+)$', sentence.strip())
        if number_match:
            return clean_text(number_match.group(1))
    
    # If still not found, check if there's any boxed-like content
    if 'box' in text.lower() and '{' in text and '}' in text:
        try:
            # Find the position of "box" and then extract content between next { and matching }
            box_pos = text.lower().find('box')
            open_brace = text.find('{', box_pos)
            if open_brace != -1:
                # Simple brace matching
                level = 1
                for i in range(open_brace + 1, len(text)):
                    if text[i] == '{':
                        level += 1
                    elif text[i] == '}':
                        level -= 1
                        if level == 0:
                            return clean_text(text[open_brace + 1:i])
        except:
            pass
    
    return None

def extract_steps(text, final_response=""):
    all_steps_list = [x.strip().strip("#").strip() for x in re.findall(r"(Step \d+:.*?)(?=Step \d+:|$)", text, re.DOTALL)]
    combined_answer = ""
    if final_response:
        answer = re.findall(r"(" + final_response + r":.*?)(?=Step \d+:|$)", text, re.DOTALL)
        combined_answer = "\n".join(answer)
    return all_steps_list, combined_answer

def pprint(text):
    if isinstance(text, list):
        for x in text:
            if isinstance(x, dict):
                for k,v in x.items():
                    print(f"{k}: {v}"[:100])
                print()
            else:
                print(x)