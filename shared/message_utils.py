import uuid

from langgraph_sdk.types import Message, Role, TextPart


def build_text_message(
    text: str, role: Role = Role.agent, task_id: str | None = None, context_id: str | None = None
) -> Message:
    return Message(
        messageId=str(uuid.uuid4()),
        role=role,
        parts=[TextPart(text=text)],
        taskId=task_id,
        contextId=context_id,
    )
