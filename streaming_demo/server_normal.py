"""
Step 1: Normal (non-streaming) endpoint.
The server "computes" a paragraph word by word (simulated with sleep),
but the client sees NOTHING until the entire response is ready.
"""

import time
from fastapi import FastAPI

app = FastAPI()

PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog. "
    "This sentence is being generated word by word on the server, "
    "but you will only see it after all words are ready. "
    "Imagine waiting 10 seconds for a response with no feedback."
)


@app.get("/generate")
def generate():
    words = PARAGRAPH.split()
    result = []
    for word in words:
        time.sleep(0.3)  # simulate computation per word
        result.append(word)
    return {"text": " ".join(result)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
