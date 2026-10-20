from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine import Engine
from explain import Explainer

WEB = Path(__file__).resolve().parent.parent / "web"

engine = Engine()
explainer = Explainer(engine)
app = FastAPI(title="Transformer Explainer 3D")
app.add_middleware(GZipMiddleware, minimum_size=2048)


class RunRequest(BaseModel):
    text: str
    target: int | None = None


class DetailRequest(BaseModel):
    run_id: str
    key: str
    token: int = -1
    dim: int | None = None
    bin: int | None = None
    head: int = 0
    key_token: int | None = None
    mode: str = "forward"


@app.post("/api/run")
def run(req: RunRequest):
    _, payload = engine.run(req.text, req.target)
    return payload


@app.post("/api/detail")
def detail(req: DetailRequest):
    run = engine.runs.get(req.run_id)
    if run is None:
        raise HTTPException(404, "Run expired; run the model again.")
    try:
        return explainer.explain(run, req.key, req.token, req.dim, req.bin, req.head, req.key_token, req.mode)
    except KeyError as exc:
        raise HTTPException(400, f"Unknown stage {exc}") from exc


@app.get("/api/tokenize")
def tokenize(text: str):
    ids = engine.tokenize(text)
    return {"ids": ids, "tokens": [engine.tok.decode([i]) for i in ids]}


app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
