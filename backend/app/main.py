from fastapi import FastAPI
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.api.health import router

app = FastAPI(title="Keyword Spotting", version="0.1.0")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"])
app.include_router(router)
