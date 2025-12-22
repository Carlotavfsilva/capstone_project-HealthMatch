from google import genai
from google.genai import types

def generate_response(
    user_input,
    system_prompt,
    temperature,
    api_key,
    use_url_tool=False
):
    client = genai.Client(api_key=api_key)

    # 1️⃣ Tentar com URL tool (se aplicável)
    if use_url_tool:
        try:
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                tools=[{"url_context": {}}]
            )

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_input,
                config=config
            )
            return response.text

        except Exception:
            pass  # cai para fallback silencioso

    # 2️⃣ Fallback SEM tool (sempre funciona)
    fallback_config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_input,
        config=fallback_config
    )
    return response.text