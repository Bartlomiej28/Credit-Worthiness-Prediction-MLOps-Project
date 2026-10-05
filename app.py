import asyncio
import os
import secrets
import threading
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

from credit_worthiness.constants import APP_HOST, APP_PORT, FORM_FILE_PATH
from credit_worthiness.logger import logging
from credit_worthiness.pipeline.prediction_pipeline import CreditWorthinessClassifier, CreditWorthinessData
from credit_worthiness.pipeline.training_pipeline import TrainPipeline
from credit_worthiness.utils.main_utils import read_yaml_file
from credit_worthiness.utils.validation import FormValidator

app = FastAPI(
    title="Credit Worthiness Prediction",
    description="Production ready MLOps pipeline that scores the credit risk of a loan applicant",
)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

_configured_origins = os.getenv("CORS_ALLOW_ORIGINS", "*")
origins = [origin.strip() for origin in _configured_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FORM_FIELDS = read_yaml_file(file_path=FORM_FILE_PATH)["fields"]
FIELD_NAMES = [field["name"] for field in FORM_FIELDS]
VALIDATOR = FormValidator(FORM_FIELDS)

_classifier: Optional[CreditWorthinessClassifier] = None
_classifier_lock = threading.Lock()
_training_lock = threading.Lock()


def get_classifier() -> CreditWorthinessClassifier:
    """
    Returns the process wide classifier, loading the production model on first use.
    """
    global _classifier

    with _classifier_lock:
        if _classifier is None:
            classifier = CreditWorthinessClassifier()
            classifier.get_model()
            _classifier = classifier
        return _classifier


def reset_classifier() -> None:
    """
    Drops the cached classifier so that the next request picks up the new model.
    """
    global _classifier

    with _classifier_lock:
        _classifier = None


class DataForm:
    """
    Collects the values of the HTML form described by config/form.yaml.
    """

    def __init__(self, request: Request):
        self.request: Request = request
        self.values: Dict[str, Optional[str]] = {name: None for name in FIELD_NAMES}

    async def get_credit_data(self) -> None:
        form = await self.request.form()
        for name in FIELD_NAMES:
            self.values[name] = form.get(name)


def build_response(prediction: str, probability: Optional[float]) -> Dict[str, Any]:
    """
    Turns the raw model output into the wording shown by the user interface.
    """
    is_creditworthy = prediction == "good"
    context = "Creditworthy - credit can be granted" if is_creditworthy else "Not creditworthy - high risk of default"

    return {
        "credit_score": prediction,
        "is_creditworthy": is_creditworthy,
        "context": context,
        "risk_probability": None if probability is None else round(probability * 100, 2),
    }


def run_prediction(values: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scores an already validated set of applicant features with the production model.
    """
    credit_data = CreditWorthinessData(**{name: values[name] for name in FIELD_NAMES})
    credit_df = credit_data.get_input_data_frame()

    prediction, probability = get_classifier().predict_with_probability(dataframe=credit_df)
    return build_response(prediction, probability)


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        request,
        "creditworthiness.html",
        {"fields": FORM_FIELDS, "result": None},
    )


@app.get("/health")
async def health():
    """
    Liveness: the process is up and able to answer. It deliberately says nothing
    about the model, so that a container without one can still be trained.
    """
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    """
    Readiness: the production model is loaded and predictions can be served. The
    container healthcheck points here, so that a model-less container is reported
    as unhealthy instead of silently answering every prediction with a 500.
    """
    try:
        classifier = get_classifier()
        return {"status": "ready", "model": repr(classifier.model)}

    except Exception as e:
        logging.info(f"Readiness probe failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "not ready",
                "detail": "No trained model is available. Run the training pipeline first.",
            },
        )


@app.post("/train")
async def train_route_client(x_train_token: Optional[str] = Header(default=None)):
    """
    Triggers a full training run. The endpoint stays disabled unless TRAIN_API_TOKEN
    is set, because an open training endpoint lets anybody burn down the instance.
    """
    expected_token = os.getenv("TRAIN_API_TOKEN")

    if not expected_token:
        return JSONResponse(
            status_code=403,
            content={"status": False, "error": "Training is disabled, TRAIN_API_TOKEN is not set."},
        )

    if not x_train_token or not secrets.compare_digest(x_train_token, expected_token):
        return JSONResponse(status_code=401, content={"status": False, "error": "Invalid training token."})

    if not _training_lock.acquire(blocking=False):
        return JSONResponse(
            status_code=409,
            content={"status": False, "error": "A training run is already going on."},
        )

    try:
        await asyncio.to_thread(TrainPipeline().run_pipeline)
        reset_classifier()
        return {"status": True, "message": "Training successful."}

    except Exception as e:
        logging.exception("The training pipeline failed")
        return JSONResponse(status_code=500, content={"status": False, "error": str(e)})

    finally:
        _training_lock.release()


@app.post("/")
async def predict_route_client(request: Request):
    form = DataForm(request)
    await form.get_credit_data()

    errors = VALIDATOR.validate(form.values)
    if errors:
        return templates.TemplateResponse(
            request,
            "creditworthiness.html",
            {
                "fields": FORM_FIELDS,
                "result": None,
                "error": " ".join(errors),
                "submitted": form.values,
            },
            status_code=422,
        )

    try:
        return templates.TemplateResponse(
            request,
            "creditworthiness.html",
            {
                "fields": FORM_FIELDS,
                "result": run_prediction(form.values),
                "submitted": form.values,
            },
        )

    except Exception:
        logging.exception("The prediction failed")
        return templates.TemplateResponse(
            request,
            "creditworthiness.html",
            {
                "fields": FORM_FIELDS,
                "result": None,
                "error": "The prediction could not be computed, please try again later.",
                "submitted": form.values,
            },
            status_code=500,
        )


@app.post("/predict")
async def predict_api(payload: Dict[str, Any]):
    """
    JSON counterpart of the HTML form, handy for smoke tests and integrations.
    """
    errors = VALIDATOR.validate(payload)
    if errors:
        return JSONResponse(status_code=422, content={"status": False, "errors": errors})

    try:
        return {"status": True, **run_prediction(payload)}

    except Exception:
        logging.exception("The prediction failed")
        return JSONResponse(
            status_code=500,
            content={"status": False, "error": "The prediction could not be computed."},
        )


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
