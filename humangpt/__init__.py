"""HumanGPT — human-in-the-loop OpenAI-compatible mock endpoint.

A QA/test harness that exposes an OpenAI-compatible HTTP API where no AI
generates responses. Every request is queued and answered by a human operator
through a web UI; the human's typed answer is returned in the exact OpenAI
response shape (chat completions and Responses API, non-stream and SSE stream).
"""

__version__ = "0.1.0"
