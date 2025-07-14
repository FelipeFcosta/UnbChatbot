#!/usr/bin/env python3

import requests
import json
import argparse

# --- Configuration ---
# CHATBOT_URL = "https://vaniamarnivania--unb-chatbot-raft-gguf-web-endpoint--b3543c-dev.modal.run"
# CHATBOT_URL = "https://felipecostasdc--unb-chatbot-raft-gguf-web-endpoint-m-443271-dev.modal.run"
# CHATBOT_URL = "https://bcadairton--unb-chatbot-raft-gguf-web-endpoint-model-c93bc2-dev.modal.run"
# CHATBOT_URL = "https://marnivania12b--unb-chatbot-raft-gguf-web-endpoint-mo-eca5ca-dev.modal.run" # 12b_run9
CHATBOT_URL = "https://fejota12b--unb-chatbot-raft-gguf-web-endpoint-modele-b3f164-dev.modal.run" # 12b_run10
DEFAULT_MAX_TOKENS = 6144
DEFAULT_TEMPERATURE = 0.5
DEFAULT_TOP_P = 0.95
# --- End Configuration ---

def query_chatbot(prompt: str, max_tokens: int, temperature: float, top_p: float) -> dict:
    """
    Sends a prompt to the chatbot API and returns the JSON response.
    """
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p
    }

    # print(f"Sending request to: {CHATBOT_URL}")
    print(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")

    try:
        response = requests.post(CHATBOT_URL, headers=headers, json=payload, timeout=180) # Increased timeout
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        return response.json()  # Assuming the response is JSON
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Response content: {response.text}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"An unexpected error occurred: {req_err}")
    except json.JSONDecodeError:
        print("Error: Could not decode JSON response.")
        print(f"Response content: {response.text if 'response' in locals() else 'No response object'}")
    return None

def main():
    parser = argparse.ArgumentParser(description="Query the chatbot API.")
    parser.add_argument("prompt", type=str, help="The prompt to send to the chatbot.")
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum number of tokens to generate (default: {DEFAULT_MAX_TOKENS})."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})."
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=DEFAULT_TOP_P,
        help=f"Nucleus sampling top_p (default: {DEFAULT_TOP_P})."
    )

    args = parser.parse_args()

    response_data = query_chatbot(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )

    if response_data:
        print("\n--- Chatbot Response ---")
        # Pretty print the JSON response
        print(json.dumps(response_data, indent=2, ensure_ascii=False))
        # If you know a specific key holds the main text, you might print that directly
        # e.g., if response_data = {"choices": [{"text": "Answer here"}]}, then:
        # print(response_data.get("choices", [{}])[0].get("text", "No text found in response."))

if __name__ == "__main__":
    main()