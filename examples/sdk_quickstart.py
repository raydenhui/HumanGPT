"""How to point the official openai-python SDK at HumanGPT.

Run the server first (``humangpt``), then in *another* terminal:

    python examples/sdk_quickstart.py

Each call parks a request in the operator web UI (http://127.0.0.1:8000/).
Answer them there — type your reply (or flag a tool call) and the parked
client call completes.
"""

from __future__ import annotations

import json

from openai import OpenAI

client = OpenAI(
    api_key="sk-any-value-works-in-any-mode",
    base_url="http://127.0.0.1:8000/v1",
)

print("== 1) normal chat ==")
print("   -> open http://127.0.0.1:8000/ and answer it")
completion = client.chat.completions.create(
    model="human-gpt",
    messages=[
        {"role": "system", "content": "You answer like a pirate."},
        {"role": "user", "content": "Say hello."},
    ],
)
print(f"   assistant: {completion.choices[0].message.content!r}")
print(f"   finish_reason: {completion.choices[0].finish_reason}  usage: {completion.usage.model_dump()}")

print("\n== 2) streaming chat ==")
print("   -> answer the next request in the UI")
stream = client.chat.completions.create(
    model="human-gpt",
    messages=[{"role": "user", "content": "Count to three."}],
    stream=True,
)
pieces = []
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        pieces.append(chunk.choices[0].delta.content)
print("   streamed:", repr("".join(pieces)))

print("\n== 3) tool call ==")
print("   -> answer with 'this reply is a tool_call', name=get_weather, "
      'arguments={"city": "Paris"}')
completion = client.chat.completions.create(
    model="human-gpt",
    messages=[{"role": "user", "content": "Weather in Paris?"}],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ],
    tool_choice="auto",
)
call = completion.choices[0].message.tool_calls[0]
print(f"   finish_reason={completion.choices[0].finish_reason}")
print(f"   tool call: {call.function.name}({json.dumps(json.loads(call.function.arguments))})")

print("\n== 4) Responses API ==")
print("   -> answer the responses request in the UI")
response = client.responses.create(
    model="human-gpt",
    instructions="Be terse.",
    input="What color is the ocean?",
)
text = "".join(
    part.text
    for item in response.output
    if item.type == "message"
    for part in item.content
    if part.type == "output_text"
)
print(f"   response: {text!r}")

print("\nAll four flows completed. Check /history in the UI for the audit trail.")
