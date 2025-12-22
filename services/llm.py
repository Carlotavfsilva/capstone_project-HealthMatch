from google import genai
from google.genai import types

def generate_response(user_input, system_prompt, temperature, api_key):
    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_input,
        config=config
    )

    return response.text