"""Experiment endpoints."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException

from loreley.api.schemas.experiments import ExperimentDetailOut, ExperimentOut
from loreley.api.services.experiments import get_experiment, list_experiments

router = APIRouter()


@router.get("/repositories/{repository_id}/experiments", response_model=list[ExperimentOut])
def get_experiments(repository_id: UUID) -> list[ExperimentOut]:
    return list_experiments(repository_id=repository_id)


@router.get("/experiments/{experiment_id}", response_model=ExperimentDetailOut)
def get_experiment_detail(experiment_id: UUID) -> ExperimentDetailOut:
    experiment = get_experiment(experiment_id=experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    return experiment


