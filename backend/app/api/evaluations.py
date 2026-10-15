"""Evaluation Run routes. Import reads only the operator-configured run directory."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request, Response

from app.services.evaluations import EvaluationService

router = APIRouter(prefix="/api/evaluations")


def service(request: Request) -> EvaluationService:
    result: EvaluationService = request.app.state.evaluations
    return result


Service = Annotated[EvaluationService, Depends(service)]


@router.get("")
def runs(evaluations: Service) -> dict[str, Any]:
    return evaluations.list_runs()


@router.post("/import")
def import_committed(evaluations: Service) -> dict[str, Any]:
    return evaluations.import_committed()


@router.get("/compare")
def compare(run_a: str, run_b: str, evaluations: Service) -> dict[str, Any]:
    return evaluations.compare(run_a, run_b)


@router.get("/{run_id}")
def run(run_id: str, evaluations: Service) -> dict[str, Any]:
    return evaluations.get(run_id)


@router.delete("/{run_id}", status_code=204)
def delete(run_id: str, evaluations: Service) -> Response:
    evaluations.delete(run_id)
    return Response(status_code=204)
