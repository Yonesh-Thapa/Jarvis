# full, runnable code here
import os
import openai

class KnowledgeOracle:
    """
    Acts as the AI's interface to an external Large Language Model (LLM).
    --- UPGRADE: This version is configured to use the high-speed Groq API. ---
    """
    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            print("WARNING: GROQ_API_KEY environment variable not set. Knowledge Oracle will be disabled.")
            self.client = None
        else:
            # The 'openai' library is used to connect to Groq's OpenAI-compatible endpoint
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            print("KnowledgeOracle initialized and connected to Groq API.")

    def query_llm(self, prompt: str) -> str | None:
        """
        Sends a prompt to the external LLM and returns its response.
        """
        if not self.client:
            print("ORACLE_QUERY_FAIL: Oracle is disabled (API key not set).")
            return None

        print(f"--- Oracle Query (Groq): Sending prompt... ---")
        print(f"  > {prompt}")
        
        try:
            chat_completion = self.client.chat.completions.create(
                model="llama3-8b-8192", # Llama 3 8B on Groq is excellent and fast
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Provide a very concise, single-sentence explanation suitable for a learning AI."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=60,
                temperature=0.7,
            )
            response = chat_completion.choices[0].message.content
            print(f"--- Oracle Response Received ---")
            print(f"  < {response}")
            return response
        except Exception as e:
            print(f"ORACLE_QUERY_FAIL: An error occurred while contacting the API: {e}")
            return None