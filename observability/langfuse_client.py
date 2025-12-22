from langfuse import Langfuse

def init_langfuse(public_key, secret_key, host):
    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host
    )