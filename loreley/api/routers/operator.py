"""Operator console endpoints."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from loreley.api.schemas.operator import (
    OperatorStatusOut,
    OperatorTaskOut,
    OperatorTaskPageOut,
)
from loreley.api.services.operator import (
    OperatorTaskAlreadyActiveError,
    OperatorTaskNotFoundError,
    create_baseline_ensure_task,
    get_operator_task,
    list_operator_tasks,
    operator_status,
    run_baseline_ensure_task,
)

router = APIRouter()


@router.get("/operator/status", response_model=OperatorStatusOut)
def get_operator_status() -> OperatorStatusOut:
    return OperatorStatusOut.model_validate(operator_status())


@router.post("/operator/tasks/baseline-ensure", response_model=OperatorTaskOut)
def create_operator_baseline_ensure_task(
    background_tasks: BackgroundTasks,
) -> OperatorTaskOut:
    try:
        task = create_baseline_ensure_task()
    except OperatorTaskAlreadyActiveError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    background_tasks.add_task(run_baseline_ensure_task, task.id)
    return OperatorTaskOut.model_validate(task)


@router.get("/operator/tasks", response_model=OperatorTaskPageOut)
def get_operator_tasks(
    limit: int = Query(default=50, ge=1, le=200),
) -> OperatorTaskPageOut:
    rows = list_operator_tasks(limit=limit)
    return OperatorTaskPageOut(items=[OperatorTaskOut.model_validate(row) for row in rows])


@router.get("/operator/tasks/{task_id}", response_model=OperatorTaskOut)
def get_operator_task_detail(task_id: UUID) -> OperatorTaskOut:
    try:
        return OperatorTaskOut.model_validate(get_operator_task(task_id=task_id))
    except OperatorTaskNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
