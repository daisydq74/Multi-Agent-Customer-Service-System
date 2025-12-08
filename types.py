from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Role(str, Enum):
    user = "user"
    agent = "agent"


class TextPart(BaseModel):
    text: str


class Message(BaseModel):
    messageId: str = Field(default_factory=lambda: uuid.uuid4().hex)
    role: Role
    parts: List[TextPart]
    taskId: Optional[str] = None
    contextId: Optional[str] = None


class TaskState(str, Enum):
    running = "running"
    completed = "completed"
    canceled = "canceled"


class TaskStatus(BaseModel):
    state: TaskState
    message: Optional[Message] = None


class Task(BaseModel):
    id: str
    contextId: str
    history: List[Message]
    status: TaskStatus


class TaskStatusUpdateEvent(BaseModel):
    taskId: str
    contextId: str
    status: TaskStatus
    final: bool


Event = TaskStatusUpdateEvent


class TaskPushNotificationConfig(BaseModel):
    task_id: str
    push_notification_config: Dict[str, Any]


class TaskQueryParams(BaseModel):
    id: str


class TaskIdParams(BaseModel):
    id: str


class MessageSendParams(BaseModel):
    message: Message


class GetTaskPushNotificationConfigParams(BaseModel):
    id: str


class ListTaskPushNotificationConfigParams(BaseModel):
    limit: Optional[int] = None


class DeleteTaskPushNotificationConfigParams(BaseModel):
    id: str


class AgentSkill(BaseModel):
    id: str
    name: str
    description: str
    tags: List[str]
    inputModes: List[str]
    outputModes: List[str]
    examples: List[str]


class AgentCapabilities(BaseModel):
    streaming: bool = True


class AgentProvider(BaseModel):
    organization: str
    url: str


class AgentCard(BaseModel):
    name: str
    description: str
    url: str
    version: str
    skills: List[AgentSkill]
    defaultInputModes: List[str]
    defaultOutputModes: List[str]
    capabilities: AgentCapabilities
    provider: AgentProvider
    documentationUrl: Optional[str] = None
    preferredTransport: Optional[str] = None


__all__ = [
    "AgentCard",
    "AgentCapabilities",
    "AgentProvider",
    "AgentSkill",
    "DeleteTaskPushNotificationConfigParams",
    "Event",
    "GetTaskPushNotificationConfigParams",
    "ListTaskPushNotificationConfigParams",
    "Message",
    "MessageSendParams",
    "Role",
    "Task",
    "TaskIdParams",
    "TaskPushNotificationConfig",
    "TaskQueryParams",
    "TaskState",
    "TaskStatus",
    "TaskStatusUpdateEvent",
    "TextPart",
]
