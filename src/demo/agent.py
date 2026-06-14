"""Focus-gated code review demo.

The agent reviews code findings. Before surfacing each one it checks
signal quality and brain state, adapting verbosity accordingly.

Usage:
    python -m src.demo.agent \
      --checkpoint checkpoints/eegnet.pt \
      --subject <BEST_SUBJECT> \
      --run 4

Requires ANTHROPIC_API_KEY in the environment (the agentic loop calls the
Claude API). The EEG pipeline is started in-process via startup().
"""
import argparse
import json
import time
import anthropic
from dotenv import load_dotenv
from ..server.server import get_signal_quality, get_brain_state, startup

# Load ANTHROPIC_API_KEY (and any other vars) from a local .env if present.
load_dotenv()

FINDINGS = [
    "The authentication middleware doesn't validate JWT expiration. An expired token is accepted indefinitely.",
    "Database queries in user_service.py are concatenated strings — SQL injection risk on the username field.",
    "The cache TTL is hardcoded to 60 seconds with no configuration hook. This breaks under high load.",
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
    client = anthropic.Anthropic()
    messages = [
        {
            "role": "user",
            "content": (
                f"Please review these {len(FINDINGS)} findings:\n\n"
                + "\n".join(f"{i+1}. {f}" for i, f in enumerate(FINDINGS))
            ),
        }
    ]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )

        # Print any text blocks
        for block in response.content:
            if hasattr(block, "text"):
                print(f"Agent: {block.text}\n")

        if response.stop_reason == "end_turn":
            break

        # Handle tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  [tool] {block.name}({json.dumps(block.input)})")
                result = execute_tool(block.name, block.input)
                print(f"  [result] {json.dumps(result)}\n")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})


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
