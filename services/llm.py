from google import genai
from langfuse import observe

@observe()
def generate_response(user_input, system_prompt, temperature, api_key):
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_input,
        config={
            "system_instruction": system_prompt,
            "temperature": temperature
        }
    )
    return response.text