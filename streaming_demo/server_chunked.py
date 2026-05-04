"""
Step 2: Raw chunked streaming endpoint.
The server yields each word as soon as it's "computed".
The HTTP connection stays open. Bytes flow incrementally.
This is what's literally happening on the wire.
"""

import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog. "
    "This sentence is being generated word by word on the server, "
    "and you see each word the moment it is ready. "
    "No more waiting for the full response."
)


def word_generator():
    words = PARAGRAPH.split()
    for word in words:
        time.sleep(0.3)  # simulate computation per word
        yield word + " "


@app.get("/generate/stream")
def generate_stream():
    return StreamingResponse(word_generator(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
