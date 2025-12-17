"""Experiment endpoints."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException

from loreley.api.schemas.experiments import ExperimentDetailOut, ExperimentOut
from loreley.api.services.experiments import get_experiment, list_experiments

router = APIRouter()


@router.get("/repositories/{repository_id}/experiments", response_model=list[ExperimentOut])
def get_experiments(repository_id: UUID) -> list[ExperimentOut]:
    experiments = list_experiments(repository_id=repository_id)
    return [
        ExperimentOut(
            id=e.id,
            repository_id=e.repository_id,
            config_hash=e.config_hash,
            name=e.name,
            status=e.status,
            created_at=e.created_at,
            updated_at=e.updated_at,
        )
        for e in experiments
    ]


@router.get("/experiments/{experiment_id}", response_model=ExperimentDetailOut)
def get_experiment_detail(experiment_id: UUID) -> ExperimentDetailOut:
    experiment = get_experiment(experiment_id=experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    return ExperimentDetailOut(
        id=experiment.id,
        repository_id=experiment.repository_id,
        config_hash=experiment.config_hash,
        name=experiment.name,
        status=experiment.status,
        created_at=experiment.created_at,
        updated_at=experiment.updated_at,
        config_snapshot=dict(experiment.config_snapshot or {}),
    )


