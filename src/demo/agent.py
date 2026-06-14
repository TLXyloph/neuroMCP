"""Focus-gated code review demo.

The agent reviews code findings. Before surfacing each one it checks
signal quality and brain state, adapting verbosity accordingly.

Usage:
    python -m src.demo.agent \
      --checkpoint checkpoints/eegnet.pt \
      --subject <BEST_SUBJECT> \
      --run 4

Requires ANTHROPIC_API_KEY in the environment. Extended thinking is
streamed to /tmp/neuromcp_thinking.log; run `tail -f` on it in a
second terminal to watch Claude reason in real time.
"""
import argparse
import json
import time
import anthropic
from dotenv import load_dotenv
from ..server.server import get_signal_quality, get_brain_state, startup

load_dotenv()

THINKING_LOG = "/tmp/neuromcp_thinking.log"

FINDINGS = [
    "The authentication middleware doesn't validate JWT expiration. "
    "An expired token is accepted indefinitely.",
    "Database queries in user_service.py are concatenated strings — "
    "SQL injection risk on the username field.",
    "The cache TTL is hardcoded to 60 seconds with no configuration hook. "
    "This breaks under high load.",
]

SYSTEM = """You are a code reviewer. You have access to two tools that read the operator's
EEG brain state in real time. Before surfacing each code review finding, you MUST:

1. Call get_signal_quality() — if snr < 3, say exactly: "Signal noisy — take a breath, I'll wait."
   and do not surface the finding yet.
2. Call get_brain_state(confidence_threshold=0.65).
   - If state is "REST": surface the finding in full detail.
   - If state is "LEFT_IMAGERY" or "RIGHT_IMAGERY": surface only a one-sentence summary,
     prefixed with "You seem focused on something — one thing: "
   - If state is "LOW_CONFIDENCE": briefly note you'll check again, then recheck once.

Work through all findings. Be concise. Do not explain this process to the operator."""

TOOLS = [
    {
        "name": "get_signal_quality",
        "description": "Return current EEG signal quality metrics.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_brain_state",
        "description": "Return decoded motor imagery state with calibrated confidence.",
        "input_schema": {
            "type": "object",
            "properties": {
                "confidence_threshold": {
                    "type": "number",
                    "description": "Minimum confidence to return a state. Default 0.7.",
                    "default": 0.7,
                }
            },
            "required": [],
        },
    },
]


def execute_tool(name: str, inputs: dict) -> dict:
    if name == "get_signal_quality":
        return get_signal_quality().model_dump()
    elif name == "get_brain_state":
        threshold = inputs.get("confidence_threshold", 0.7)
        return get_brain_state(confidence_threshold=threshold).model_dump()
    raise ValueError(f"Unknown tool: {name}")


def _rule(label: str = "") -> None:
    w = 60
    if label:
        pad = max(0, w - len(label) - 2)
        print(f"\n{'─' * (pad // 2)} {label} {'─' * (pad - pad // 2)}")
    else:
        print(f"\n{'─' * w}")


def _stream_turn(
    client: anthropic.Anthropic, messages: list, turn: int
) -> tuple[list, list, str]:
    """Stream one Claude turn. Returns (content_blocks, tool_calls, stop_reason)."""
    print("\nClaude: ", end="", flush=True)

    current_type: str | None = None
    current_thinking = ""

    try:
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=16000,
            thinking={"type": "enabled", "budget_tokens": 5000},
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        ) as stream:
            for event in stream:
                if event.type == "content_block_start":
                    current_type = event.content_block.type
                    if current_type == "thinking":
                        current_thinking = ""
                        with open(THINKING_LOG, "a") as f:
                            f.write(
                                f"\n{'=' * 50}\n"
                                f"Turn {turn}  {time.strftime('%H:%M:%S')}\n"
                                f"{'=' * 50}\n"
                            )

                elif event.type == "content_block_delta":
                    d = event.delta
                    if d.type == "text_delta":
                        print(d.text, end="", flush=True)
                    elif d.type == "thinking_delta":
                        current_thinking += d.thinking

                elif event.type == "content_block_stop":
                    if current_type == "thinking" and current_thinking:
                        with open(THINKING_LOG, "a") as f:
                            f.write(current_thinking + "\n")

            final = stream.get_final_message()

    except anthropic.BadRequestError:
        # Extended thinking not supported; fall back to standard call.
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )
        for block in response.content:
            if hasattr(block, "text"):
                print(block.text, end="", flush=True)
        final = response

    tool_calls = [
        {"id": b.id, "name": b.name, "input": b.input}
        for b in final.content
        if b.type == "tool_use"
    ]
    return final.content, tool_calls, final.stop_reason


def _print_banner(subject: int, run: int) -> None:
    w = 60
    print("\n" + "=" * w)
    print("NeuroMCP Demo — Focus-Gated Code Review")
    print(f"Model : EEGNet + MC Dropout  (N=50 passes)")
    print(f"Subject: {subject}  |  Run: {run}  |  Classes: REST / LEFT / RIGHT")
    print("-" * w)
    print("Brain states")
    print("  REST           -> full finding detail")
    print("  LEFT/RIGHT     -> one-sentence summary (operator is focused)")
    print("  LOW_CONFIDENCE -> agent rechecks before surfacing")
    print("=" * w + "\n")


def run_agentic_loop(subject: int, run: int) -> None:
    _print_banner(subject, run)

    open(THINKING_LOG, "w").close()
    print(f"Extended thinking: {THINKING_LOG}")
    print(f"Watch live in another terminal: tail -f {THINKING_LOG}")

    _rule("findings")
    for i, f in enumerate(FINDINGS, 1):
        print(f"  {i}. {f}")

    _rule()
    print("\nYou: ", end="", flush=True)
    user_input = input()

    findings_block = "\n".join(f"{i+1}. {f}" for i, f in enumerate(FINDINGS))
    opening = f"{user_input}\n\nFindings to review:\n{findings_block}"

    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": opening}]

    turn = 0
    while True:
        turn += 1
        content_blocks, tool_calls, stop_reason = _stream_turn(client, messages, turn)
        print()

        if stop_reason == "end_turn":
            break

        tool_results = []
        for tc in tool_calls:
            _rule(f"tool: {tc['name']}")
            if tc["input"]:
                print(f"  input:  {json.dumps(tc['input'])}")
            result = execute_tool(tc["name"], tc["input"])
            print(f"  result: {json.dumps(result)}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tc["id"],
                "content": json.dumps(result),
            })

        messages.append({"role": "assistant", "content": content_blocks})
        messages.append({"role": "user", "content": tool_results})
        _rule()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--run", type=int, required=True)
    args = parser.parse_args()

    print("Starting EEG pipeline...")
    startup(args.checkpoint, args.subject, args.run)
    print("Waiting 3s for buffer to fill...")
    time.sleep(3.0)

    run_agentic_loop(args.subject, args.run)


if __name__ == "__main__":
    main()
