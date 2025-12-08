from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, AsyncGenerator, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langgraph_sdk.types import (
    AgentCard,
    DeleteTaskPushNotificationConfigParams,
    Event,
    GetTaskPushNotificationConfigParams,
    ListTaskPushNotificationConfigParams,
    Message,
    MessageSendParams,
    Role,
    Task,
    TaskIdParams,
    TaskPushNotificationConfig,
    TaskQueryParams,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)


class SimpleAgentRequestHandler:
    """Lightweight JSON-RPC handler that stores tasks in-memory."""

    def __init__(self, agent_name: str, skill_callback):
        self.agent_name = agent_name
        self._tasks: Dict[str, Task] = {}
        self._history: Dict[str, List[Message]] = {}
        self._skill_callback = skill_callback

    def _new_ids(self) -> tuple[str, str]:
        task_id = str(uuid.uuid4())
        context_id = str(uuid.uuid4())
        return task_id, context_id

    async def on_get_task(self, params: TaskQueryParams) -> Task | None:
        return self._tasks.get(params.id)

    async def on_cancel_task(self, params: TaskIdParams) -> Task | None:
        task = self._tasks.get(params.id)
        if not task:
            return None
        task.status = TaskStatus(state=TaskState.canceled)
        return task

    async def on_message_send(self, params: MessageSendParams) -> Task | Message:
        task_id, context_id = self._new_ids()
        inbound_message = params.message
        inbound_message.taskId = task_id
        inbound_message.contextId = context_id

        reply = await self._skill_callback(inbound_message)
        status = TaskStatus(state=TaskState.completed, message=reply)
        task = Task(
            id=task_id,
            contextId=context_id,
            history=[inbound_message, reply],
            status=status,
        )
        self._tasks[task_id] = task
        self._history.setdefault(task_id, []).extend([inbound_message, reply])
        return task

    async def on_message_send_stream(
        self, params: MessageSendParams
    ) -> AsyncGenerator[Event, None]:
        task_id, context_id = self._new_ids()
        inbound_message = params.message
        inbound_message.taskId = task_id
        inbound_message.contextId = context_id

        start_status = TaskStatus(state=TaskState.running)
        yield TaskStatusUpdateEvent(
            taskId=task_id,
            contextId=context_id,
            status=start_status,
            final=False,
        )

        reply = await self._skill_callback(inbound_message)
        final_status = TaskStatus(state=TaskState.completed, message=reply)
        task = Task(
            id=task_id,
            contextId=context_id,
            history=[inbound_message, reply],
            status=final_status,
        )
        self._tasks[task_id] = task
        self._history.setdefault(task_id, []).extend([inbound_message, reply])

        yield TaskStatusUpdateEvent(
            taskId=task_id,
            contextId=context_id,
            status=final_status,
            final=True,
        )

    async def on_set_task_push_notification_config(
        self, params: TaskPushNotificationConfig
    ) -> TaskPushNotificationConfig:
        return params

    async def on_get_task_push_notification_config(
        self, params: TaskIdParams | GetTaskPushNotificationConfigParams
    ) -> TaskPushNotificationConfig:
        return TaskPushNotificationConfig(task_id=params.id, push_notification_config={})

    async def on_resubscribe_to_task(
        self, params: TaskIdParams
    ) -> AsyncGenerator[Event, None]:
        task = self._tasks.get(params.id)
        if not task:
            return
        yield TaskStatusUpdateEvent(
            taskId=task.id,
            contextId=task.contextId,
            status=task.status,
            final=True,
        )

    async def on_list_task_push_notification_config(
        self, params: ListTaskPushNotificationConfigParams
    ) -> list[TaskPushNotificationConfig]:
        return []

    async def on_delete_task_push_notification_config(
        self, params: DeleteTaskPushNotificationConfigParams
    ) -> None:
        return None


class RPCRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: Dict[str, Any] | None = None
    id: str | int | None = None


def register_agent_routes(app: FastAPI, agent_card: AgentCard, handler: SimpleAgentRequestHandler) -> None:
    @app.get("/.well-known/agent-card.json")
    async def agent_card_route():
        return agent_card.model_dump()

    @app.post("/rpc")
    async def rpc_endpoint(request: RPCRequest):
        params = request.params or {}
        method = request.method

        if method == "message/send":
            result = await handler.on_message_send(MessageSendParams(**params))
        elif method == "message/send_stream":
            send_params = MessageSendParams(**params)

            async def event_gen():
                async for event in handler.on_message_send_stream(send_params):
                    yield json.dumps(event.model_dump()) + "\n"

            return StreamingResponse(event_gen(), media_type="application/json")
        elif method == "task/get":
            result = await handler.on_get_task(TaskQueryParams(**params))
            if result is None:
                raise HTTPException(status_code=404, detail="Task not found")
        elif method == "task/cancel":
            result = await handler.on_cancel_task(TaskIdParams(**params))
            if result is None:
                raise HTTPException(status_code=404, detail="Task not found")
        else:
            raise HTTPException(status_code=404, detail="Unknown method")

        return {"jsonrpc": "2.0", "id": request.id, "result": result.model_dump()}

    @app.get("/health")
    async def health():
        return {"status": "ok"}
