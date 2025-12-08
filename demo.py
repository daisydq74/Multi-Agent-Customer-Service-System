import asyncio
import os

import httpx

from langgraph_sdk.types import Message, MessageSendParams, Role
from shared.message_utils import build_text_message

ROUTER_RPC = os.getenv("ROUTER_RPC", "http://localhost:8010/rpc")


def build_request(prompt: str) -> dict:
    base_message = build_text_message(prompt, role=Role.user)
    params = MessageSendParams(
        message=Message(messageId=base_message.messageId, role=Role.user, parts=base_message.parts)
    )
    return {"jsonrpc": "2.0", "id": "demo", "method": "message/send", "params": params.model_dump()}


def print_response(scenario: str, prompt: str, result: dict | None) -> None:
    print(f"=== Scenario: {scenario} ===")
    print(f"Prompt: {prompt}")
    if not result:
        print("No result returned")
    else:
        print("Router task result:", result)
    print()


async def main():
    test_scenarios = [
        ("Simple Query", "Get customer information for ID 5"),
        ("Coordinated Query", "I'm customer 12345 and need help upgrading my account"),
        ("Complex Query", "Show me all active customers who have open tickets"),
        ("Escalation", "I've been charged twice, please refund immediately!"),
        ("Multi-Intent", "Update my email to new@email.com and show my ticket history"),
    ]

    async with httpx.AsyncClient() as client:
        for scenario, prompt in test_scenarios:
            request_body = build_request(prompt)
            response = await client.post(ROUTER_RPC, json=request_body)
            response.raise_for_status()
            result = response.json().get("result")
            print_response(scenario, prompt, result)


if __name__ == "__main__":
    asyncio.run(main())
