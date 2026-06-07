"""FastAPI surface for Track A. Thin: validate -> call engine -> return.

Run:  uvicorn controller_service:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from . import config, controllers, evals, model
from .controllers import ControllerNotFound
from .schemas import (ComposeReq, DiagnoseReq, EvaluateReq, ExecuteReq,
                      ForgettingReq, JailbreakReq, PairReq, TrainReq)

app = FastAPI(title="OpenMirror Controller Engine", version="0.2")


@app.exception_handler(ControllerNotFound)
async def _not_found(_req: Request, exc: ControllerNotFound):
    return JSONResponse(status_code=404, content={"detail": f"controller not found: {exc}"})


@app.get("/health")
def health():
    return {
        "model": config.MODEL_NAME,
        "device": model.device(),
        "model_loaded": model.is_loaded(),
        "gates": config.GATES,
        "max_log_gate": config.MAX_LOG_GATE,
        "controller_dir": str(config.CONTROLLER_DIR.resolve()),
    }


@app.get("/controllers")
def list_controllers():
    return {"controllers": controllers.list_controllers()}


@app.post("/train")
def train(req: TrainReq):
    if not req.examples:
        raise HTTPException(400, "no examples provided")
    return controllers.train(req.task_id, req.examples, steps=req.steps, lr=req.lr,
                             batch_size=req.batch_size, max_length=req.max_length)


@app.post("/compose")
def compose(req: ComposeReq):
    return controllers.compose(req.controller_ids, req.weights, new_id=req.new_id)


@app.post("/execute")
def execute(req: ExecuteReq):
    gen = controllers.generator_for(req.controller_id)
    return {"controller_id": req.controller_id,
            "completion": gen(req.prompt, req.max_new_tokens)}


@app.post("/evaluate")
def evaluate(req: EvaluateReq):
    items = [it.model_dump() for it in req.items]
    return evals.evaluate(req.controller_id, items, max_new_tokens=req.max_new_tokens)


@app.get("/inspect/{controller_id}")
def inspect(controller_id: str, dense: bool = False):
    return controllers.inspect_controller(controller_id, dense=dense)


@app.post("/pair")
def pair(req: PairReq):
    return controllers.pair(req.a, req.b)


@app.post("/diagnose")
def diagnose(req: DiagnoseReq):
    items = [it.model_dump() for it in req.items]
    return evals.diagnose(req.skill, items, threshold=req.threshold,
                          max_new_tokens=req.max_new_tokens)


@app.post("/forgetting")
def forgetting(req: ForgettingReq):
    items = [it.model_dump() for it in req.items]
    return evals.forgetting(req.controller_id, items, max_new_tokens=req.max_new_tokens)


@app.post("/jailbreak")
def jailbreak(req: JailbreakReq):
    return evals.jailbreak(req.controller_id, req.needle, req.prompts,
                           baseline_controller_id=req.baseline_controller_id,
                           max_new_tokens=req.max_new_tokens)
