"""Uploaded filenames are display metadata, never executable input."""

import csv
import io
import re
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.requests import ClientDisconnect
from starlette.responses import FileResponse, JSONResponse

from app.errors import ServiceError
from app.services.recordings import RecordingService

router = APIRouter(prefix="/api")


def service(request: Request) -> RecordingService:
    result: RecordingService = request.app.state.recordings
    return result


Service = Annotated[RecordingService, Depends(service)]


class Analyze(BaseModel):
    model_config = ConfigDict(extra="forbid")
    keywords: list[str] = Field(min_length=1, max_length=6)


@router.get("/models/active")
def active_model(jobs: Service) -> dict[str, Any]:
    return jobs.active_model()


@router.post("/recordings", status_code=201)
async def upload(request: Request, jobs: Service) -> dict[str, Any]:
    recording = uuid4().hex
    path = jobs.audio_dir / f"{recording}.wav"
    stored = False
    try:
        async with request.form(max_files=1, max_fields=0) as form:
            if list(form.keys()) != ["file"] or len(form.getlist("file")) != 1:
                raise ServiceError("invalid_upload", "Send exactly one WAV in the file field.", 422)
            file = form["file"]
            if not isinstance(file, UploadFile):
                raise ServiceError("invalid_upload", "Send a multipart WAV file.", 422)
            filename = file.filename or ""
            if len(filename) > 200 or not re.fullmatch(r"[\w .()-]+\.wav", filename, re.IGNORECASE):
                raise ServiceError(
                    "invalid_filename", "Use a plain WAV filename without paths or commands.", 422
                )
            size = 0
            with path.open("xb") as output:
                while chunk := await file.read(65536):
                    size += len(chunk)
                    if size > jobs.settings.max_upload_bytes:
                        raise ServiceError("upload_too_large", "WAV exceeds the byte limit.", 413)
                    output.write(chunk)
            result = jobs.add(recording, filename, path)
            stored = True
            return result
    except HTTPException as exc:
        raise ServiceError("invalid_upload", "Invalid multipart WAV upload.", 422) from exc
    except ClientDisconnect as exc:
        raise ServiceError("upload_disconnected", "Upload disconnected.", 400) from exc
    finally:
        if not stored:
            path.unlink(missing_ok=True)


@router.get("/recordings")
def recordings(jobs: Service) -> list[dict[str, Any]]:
    return jobs.list_recordings()


@router.get("/recordings/{recording}")
def recording_metadata(recording: str, jobs: Service) -> dict[str, Any]:
    return jobs.get(recording)


@router.post("/recordings/{recording}/analyze", status_code=202)
def analyze(recording: str, body: Analyze, jobs: Service) -> dict[str, Any]:
    return jobs.analyze(recording, body.keywords)


@router.get("/recordings/{recording}/detections")
def detections(recording: str, jobs: Service) -> dict[str, Any]:
    return jobs.results(recording)


CSV_HEADER = ("recording_id", "keyword", "start_s", "end_s", "confidence", "model_id")


@router.get("/recordings/{recording}/detections.csv")
def detections_csv(recording: str, jobs: Service) -> Response:
    payload = jobs.results(recording)
    if payload["status"] != "completed":
        return JSONResponse(payload)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow(CSV_HEADER)
    model_id = payload["model_id"] or ""
    for detection in payload["detections"]:
        writer.writerow(
            [
                payload["recording_id"],
                detection["keyword"],
                detection["start_s"],
                detection["end_s"],
                detection["confidence"],
                model_id,
            ]
        )
    recording_id = payload["recording_id"]
    filename = (
        f"{recording_id}.csv" if re.fullmatch(r"[0-9a-f]{32}", recording_id) else "detections.csv"
    )
    return Response(
        content=buffer.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/recordings/{recording}/audio")
def audio(recording: str, jobs: Service) -> FileResponse:
    path = jobs.audio_path(recording)
    if not path.is_file():
        raise ServiceError("audio_unavailable", "Recording audio is missing.", 410)
    return FileResponse(path, media_type="audio/wav")


@router.delete("/recordings/{recording}", status_code=204)
def delete(recording: str, jobs: Service) -> Response:
    jobs.delete(recording)
    return Response(status_code=204)
