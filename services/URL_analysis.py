from google import genai
from google.genai import types

def analyze_url_content(url, api_key):
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"Analyze and summarize the content from this URL: {url}",
        config=types.GenerateContentConfig(
            tools=[{"url_context": {}}]
        )
    )
    return response.text