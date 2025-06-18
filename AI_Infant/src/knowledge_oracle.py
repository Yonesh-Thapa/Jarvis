# full, runnable code here
import os
import openai

class KnowledgeOracle:
    """
    Acts as the AI's interface to an external Large Language Model (LLM).
    It consults this "oracle" only when its own knowledge is insufficient.
    This follows the principle of least action by using an energy-intensive
    tool only as a last resort.
    """
    def __init__(self):
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            print("WARNING: DEEPSEEK_API_KEY environment variable not set. Knowledge Oracle will be disabled.")
            self.client = None
        else:
            # The 'openai' library is a standard client for many LLM APIs,
            # including DeepSeek, by changing the base_url.
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com"
            )
            print("KnowledgeOracle initialized and connected to DeepSeek API.")

    def query_llm(self, prompt: str) -> str | None:
        """
        Sends a prompt to the external LLM and returns its response.

        Args:
            prompt (str): The question to ask the oracle.

        Returns:
            str | None: The textual response from the LLM, or None if an error occurred.
        """
        if not self.client:
            print("ORACLE_QUERY_FAIL: Oracle is disabled (API key not set).")
            return None

        print(f"--- Oracle Query: Sending prompt... ---")
        print(f"  > {prompt}")
        
        try:
            chat_completion = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Provide a concise, single-sentence explanation."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=50,
                temperature=0.7,
            )
            response = chat_completion.choices[0].message.content
            print(f"--- Oracle Response Received ---")
            print(f"  < {response}")
            return response
        except Exception as e:
            print(f"ORACLE_QUERY_FAIL: An error occurred while contacting the API: {e}")
            return None