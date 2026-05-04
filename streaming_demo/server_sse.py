"""
Step 3: Server-Sent Events (SSE) endpoint.
Same streaming concept, but now wrapped in the SSE protocol:
  - Content-Type: text/event-stream
  - Each chunk is formatted as "data: <payload>\n\n"
  - A final "data: [DONE]\n\n" signals end of stream

This is EXACTLY the protocol OpenAI uses for streaming responses.
"""

import time
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog. "
    "This sentence is being streamed using Server-Sent Events. "
    "Each word arrives as a structured event with a data field. "
    "OpenAI uses this exact same protocol for streaming tokens."
)


def sse_generator():
    words = PARAGRAPH.split()
    for i, word in enumerate(words):
        time.sleep(0.3)  # simulate computation per word
        # SSE format: "data: <json payload>\n\n"
        event_data = json.dumps({"word": word, "index": i})
        yield f"data: {event_data}\n\n"
    # Signal that the stream is complete (OpenAI convention)
    yield "data: [DONE]\n\n"


@app.get("/generate/sse")
def generate_sse():
    return StreamingResponse(sse_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
