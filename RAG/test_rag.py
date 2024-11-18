from langchain_groq import ChatGroq
from query_data import query_rag
# from langchain_community.llms.ollama import Ollama

# EVAL_PROMPT
EVAL_PROMPT = """
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response?
"""

def test_query():
    try:
        print("Starting query test...")
        assert query_and_validate(
            question="What is the pdf about?"
            # expected_response="Delication Of Power"
        )
        print("Test passed.")
    except AssertionError:
        print("Test failed. Assertion error occurred.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def query_and_validate(question: str):
    try:
        print(f"Querying for question: {question}")
        response_text = query_rag(question)
        print(f"Response from query_rag: {response_text}")

        # Format the prompt using the actual response
        prompt = EVAL_PROMPT.format(
            actual_response=response_text
        )

        print(f"Formatted prompt: {prompt}")

        # Instantiate the ChatGroq object
        llm = ChatGroq(
            temperature=0,
            groq_api_key="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",  # Check if your API key is valid
            model_name="llama-3.1-70b-versatile"
        )
        print("ChatGroq instance created.")

        # Invoke the model with the prompt
        response = llm.invoke(prompt)
        print(f"Response from ChatGroq: {response}")

        # Check if response.content is valid
        if hasattr(response, 'content'):
            print(f"Response content: {response.content}")
        else:
            print("Error: Response object does not have 'content' attribute.")
            return False

        return True

    except Exception as e:
        print(f"An error occurred during query_and_validate: {str(e)}")
        return False

if __name__ == "__main__":
    test_query()
