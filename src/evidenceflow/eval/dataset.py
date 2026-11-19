"""Load and validate the human-annotated Retrieval Evaluation Dataset.

COMPATIBILITY_ONLY: legacy HitRate semantics; new code uses eval/ir_metrics.py.
"""

import json
from json import JSONDecodeError
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from evidenceflow.domain.evaluation import EvaluationQuestion


class EvaluationDatasetError(ValueError):
    """Signals that the evaluation data's JSON structure or business schema is invalid."""


# Reuse the Domain Model as the single data contract so the Loader does not maintain a duplicate
# field definition.
_QUESTIONS_ADAPTER = TypeAdapter(list[EvaluationQuestion])


def load_evaluation_questions(path: Path) -> list[EvaluationQuestion]:
    """Load questions from a JSON file and apply Pydantic plus exact-duplicate validation.

    Args:
        path: Path to an evaluation data file whose top level is a JSON array.

    Returns:
        The questions in file order, already validated against the Domain Model.

    Raises:
        FileNotFoundError: the data file does not exist.
        EvaluationDatasetError: the JSON, top-level structure, schema, or question
            uniqueness is invalid.
    """

    dataset_path = Path(path)
    if not dataset_path.is_file():
        # Keep the dedicated file exception type so callers can tell path problems from
        # data-content problems.
        raise FileNotFoundError(f"Evaluation dataset does not exist: {dataset_path}")

    try:
        raw_data = json.loads(dataset_path.read_text(encoding="utf-8"))
    except JSONDecodeError as exc:
        # Line and column information helps a human fixing a syntax error in questions.json.
        raise EvaluationDatasetError(
            f"Invalid JSON in evaluation dataset at line {exc.lineno}, "
            f"column {exc.colno}: {dataset_path}"
        ) from exc
    except UnicodeDecodeError as exc:
        raise EvaluationDatasetError(
            f"Evaluation dataset must be UTF-8 encoded: {dataset_path}"
        ) from exc

    # Check the top-level type explicitly to give a more direct format hint than a generic Pydantic
    # error.
    if not isinstance(raw_data, list):
        raise EvaluationDatasetError("Evaluation dataset top level must be a JSON array")

    try:
        questions = _QUESTIONS_ADAPTER.validate_python(raw_data)
    except ValidationError as exc:
        raise EvaluationDatasetError(f"Invalid evaluation dataset schema: {exc}") from exc

    # Only identical question text is checked; semantic deduplication needs human judgment and is
    # not this Loader's job.
    question_texts = [question.question for question in questions]
    if len(question_texts) != len(set(question_texts)):
        raise EvaluationDatasetError("Evaluation dataset contains duplicate question text")

    return questions
