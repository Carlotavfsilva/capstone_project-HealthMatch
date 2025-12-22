import re

def is_valid_url(text):
    return re.match(r"http[s]?://", text) is not None