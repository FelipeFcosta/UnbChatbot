import requests
import json
import time

# --- Configuration ---
API_URL = "https://liteofspace--unb-chatbot-gemma-web-endpoint-generate-web-dev.modal.run"
HEADERS = {"Content-Type": "application/json"}
# Increased timeout since requests are sequential now and one long request could block others
REQUEST_TIMEOUT = 60 * 4  # Timeout in seconds (e.g., 4 minutes)

# --- List of Questions ---
questions = [
    "ENADE é obrigatório pra quem está se formando?",
    "Quais os currículos de Engenharia de Computação?",
    "o que é a cadeia de seletividade no currículo 1741/2? ela existe nesse currículo?",
    "O que significa os números 25/15 na hora de fazer a matrícula no SIGAA?",
    "Eu trabalhei muito nesse semestre, o equivalente ao dobro de horas da disciplina de estágio de engenharia de computação, posso integralizar as disciplinas de Estágio 1 e Estágio 2 em um único semestre?",
    "Quantos porcento da carga horária do curso eu preciso integralizar para começar o TCC em engenharia de computação?"
]
# --- --- --- --- --- ---

def fetch_answer(question):
    """Sends a single question to the API and returns the response."""
    payload = json.dumps({"prompt": question})
    try:
        print(f"   Sending request for: '{question[:50]}...'") # Indicate which request is being sent
        response = requests.post(API_URL, headers=HEADERS, data=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        # Attempt to parse JSON, handle potential errors
        try:
            data = response.json()
            answer = data.get("response", "Error: 'response' key not found in JSON")
            return answer
        except json.JSONDecodeError:
            print(f"   Warning: Could not decode JSON response. Status Code: {response.status_code}")
            return f"Error: Could not decode JSON response. Raw response: {response.text[:200]}..." # Show partial raw response

    except requests.exceptions.Timeout:
        print(f"   Error: Request timed out after {REQUEST_TIMEOUT} seconds.")
        return f"Error: Request timed out for question: '{question}'"
    except requests.exceptions.RequestException as e:
        print(f"   Error making request: {e}")
        return f"Error making request: {e}"
    except Exception as e:
        print(f"   An unexpected error occurred: {e}")
        return f"An unexpected error occurred: {e}"

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Starting sequential QA process with {len(questions)} questions...")
    print(f"API Endpoint: {API_URL}\n")

    total_start_time = time.time()

    # Iterate through questions sequentially
    for i, question in enumerate(questions):
        print("-" * 40)
        print(f"Processing question {i+1}/{len(questions)}...")
        print(f"❓ Question:\n{question}")

        start_time = time.time()
        answer = fetch_answer(question)
        end_time = time.time()

        print(f"\n💡 Answer (received in {end_time - start_time:.2f}s):\n{answer}")
        print("-" * 40 + "\n")
        # Optional short pause between requests if needed
        # time.sleep(1) 

    total_end_time = time.time()
    print("=" * 40)
    print("All questions processed.")
    print(f"Total time taken: {total_end_time - total_start_time:.2f} seconds")
    print("=" * 40)