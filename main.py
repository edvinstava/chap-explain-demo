"""Chapkit service wrapper so chap-core can discover this model over REST.

Runs the same train.py/predict.py entry points as the MLproject file, but
serves them as a chapkit ML service with full model metadata. With
SERVICEKIT_ORCHESTRATOR_URL set, the service self-registers with chap-core
on startup (and pings to stay live), which makes the model appear in the
Modeling App without any chap-core configuration changes. Without the env
var, registration silently no-ops and this is just a local REST service.

Run from inside this project's environment (chapkit needs Python 3.13):

    uv run --with chapkit python main.py
"""

import os
from pathlib import Path

from chapkit import BaseConfig
from chapkit.api import AssessedStatus, MLServiceBuilder, MLServiceInfo, ModelMetadata, PeriodType
from chapkit.artifact import ArtifactHierarchy
from chapkit.ml import ShellModelRunner


class ChapExplainDemoConfig(BaseConfig):
    """Configuration for chap_explain_demo (no user options; scripts ignore config.yml)."""


# Same commands as the MLproject entry points, in ShellModelRunner's template
# vocabulary. The literal `model` path is preserved across train -> predict
# via the workspace copy.
runner: ShellModelRunner[ChapExplainDemoConfig] = ShellModelRunner(
    train_command="python train.py {data_file} model",
    predict_command="python predict.py model {historic_file} {future_file} {output_file}",
    config_format="chap_core",
)

info = MLServiceInfo(
    id="chap-explain-demo",
    display_name="Explainability demo (SHAP + LIME)",
    description=(
        "Purely exogenous gradient boosted tree on lagged covariates with tidy "
        "SHAP and LIME explanation outputs, built for visualization research on "
        "covariate importance. Optimized for clear explanations, not accuracy."
    ),
    model_metadata=ModelMetadata(
        author="Edvin Stava",
        author_note="Demo model for explainability visualization research",
        author_assessed_status=AssessedStatus.gray,
        contact_email="edvin@dhis2.org",
        repository_url="https://github.com/edvinstava/chap-explain-demo",
    ),
    period_type=PeriodType.monthly,
    required_covariates=["rainfall", "mean_temperature", "population"],
    allow_free_additional_continuous_covariates=True,
)

hierarchy = ArtifactHierarchy(
    name="chap_explain_demo",
    level_labels={0: "ml_training_workspace", 1: "ml_prediction"},
)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///data/chapkit.db")
if DATABASE_URL.startswith("sqlite") and ":///" in DATABASE_URL:
    db_path = Path(DATABASE_URL.split("///")[1])
    db_path.parent.mkdir(parents=True, exist_ok=True)

app = (
    MLServiceBuilder(
        info=info,
        config_schema=ChapExplainDemoConfig,
        hierarchy=hierarchy,
        runner=runner,
        database_url=DATABASE_URL,
    )
    .with_registration()
    .build()
)


if __name__ == "__main__":
    from chapkit.api import run_app

    run_app("main:app", host="0.0.0.0", port=int(os.getenv("SERVICEKIT_PORT", "9090")))
