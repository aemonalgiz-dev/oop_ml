"""What a saved model is: a readable document of everything it learned.

Why not pickle
--------------
Pickle is one line and it executes arbitrary code on load. A model file that
crosses any trust boundary -- a deploy, an artifact store, a user upload --
becomes an exploit vector, and "only load files you trust" is a rule that
survives exactly until the first shared bucket. This library serialises
*learned parameters as JSON* instead: a document is pure data, loading it runs
no code of the document's choosing, and a person can open the file and read
what the model learned -- which, for a library whose point is showing how
things work, makes the saved artifact a teaching document too.

Three properties carry the safety story, and each is enforced rather than
promised:

* **A closed registry.** A document names its model by a bare class name that
  is looked up in a fixed dict of known classes. There is no dotted-path
  import, because importing whatever string a document supplies is the pickle
  problem in miniature.
* **Loading revalidates.** Hyperparameters go back through the model's
  ordinary pydantic constructor, and learned values are rebuilt through the
  same validating constructors that guard fresh data -- a tampered document
  claiming components in ascending-variance order, or a probability of 7.0,
  meets the same typed refusal the equivalent bug would.
* **No silent versions.** The document carries a format version, and a
  mismatch is a typed error naming both numbers, not a best-effort parse.

The honest cost is size: numpy arrays are written as nested JSON lists, so a
model that remembers its training rows -- k-nearest neighbours, the kernel
models, a bagged ensemble keeping rows for out-of-bag scoring -- produces a
document proportional to that data. Readability and safety are bought with
bytes, which is the right trade at this library's scale; a binary sidecar
format would be an optimisation to measure, not a correction.
"""

from __future__ import annotations

import json
from typing import Any

from oop_ml.core.exceptions import InvalidDocumentError

FORMAT_VERSION = 1
"""The document format this build writes and the only one it reads.

Bumped when the layout of an existing model's document changes shape --
renaming a learned part, changing a codec's fields. Adding a new model type is
not a format change: old readers never see documents naming it.
"""


class ModelDocument:
    """One fitted model as data: its type, its configuration, what it learned.

    Parameters
    ----------
    model_type:
        The bare class name, resolved against the closed registry at load.
    hyperparameters:
        The constructor arguments, exactly as the model's pydantic fields hold
        them (nested models already encoded to plain data).
    learned:
        The fitted state, one entry per learned part, every value already
        encoded to plain data.
    format_version:
        Defaults to the current version; carried explicitly so a document read
        back states what it is rather than what the reader hopes.

    Raises
    ------
    InvalidDocumentError
        If ``model_type`` is blank or either payload is not a dict.
    """

    __slots__ = ("_format_version", "_hyperparameters", "_learned", "_model_type")

    def __init__(
        self,
        model_type: str,
        hyperparameters: dict[str, Any],
        learned: dict[str, Any],
        format_version: int = FORMAT_VERSION,
    ) -> None:
        if not isinstance(model_type, str) or not model_type.strip():
            raise InvalidDocumentError("a document must name its model type")

        if not isinstance(hyperparameters, dict) or not isinstance(learned, dict):
            raise InvalidDocumentError(
                "hyperparameters and learned state must both be mappings"
            )

        self._model_type = model_type.strip()
        self._hyperparameters = dict(hyperparameters)
        self._learned = dict(learned)
        self._format_version = int(format_version)

    @property
    def model_type(self) -> str:
        """The class this document claims to be."""
        return self._model_type

    @property
    def hyperparameters(self) -> dict[str, Any]:
        """The configuration, as a copy."""
        return dict(self._hyperparameters)

    @property
    def learned(self) -> dict[str, Any]:
        """The fitted state, as a copy."""
        return dict(self._learned)

    @property
    def format_version(self) -> int:
        """Which format this document was written in."""
        return self._format_version

    def check_readable(self) -> None:
        """Raise unless this build can read this document.

        Raises
        ------
        InvalidDocumentError
            If the format version differs from :data:`FORMAT_VERSION`. Named
            in both directions, because "too new" and "too old" send a caller
            to different remedies.
        """
        if self._format_version != FORMAT_VERSION:
            raise InvalidDocumentError(
                f"this document is format version {self._format_version} and "
                f"this build reads version {FORMAT_VERSION}"
            )

    def to_json(self) -> str:
        """The document as a JSON string, stable and human-readable."""
        return json.dumps(
            {
                "format_version": self._format_version,
                "model_type": self._model_type,
                "hyperparameters": self._hyperparameters,
                "learned": self._learned,
            },
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str) -> ModelDocument:
        """Parse a document, refusing anything that is not one.

        Raises
        ------
        InvalidDocumentError
            If the text is not JSON, not an object, or missing a required key.
            The version check is separate (:meth:`check_readable`) so a caller
            can inspect an unreadable document's claimed version.
        """
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as error:
            raise InvalidDocumentError(
                f"not a JSON document: {error.msg} at position {error.pos}"
            ) from error

        if not isinstance(raw, dict):
            raise InvalidDocumentError(
                f"a model document is a JSON object; got {type(raw).__name__}"
            )

        missing = sorted(
            key
            for key in ("format_version", "model_type", "hyperparameters", "learned")
            if key not in raw
        )
        if missing:
            raise InvalidDocumentError(f"the document is missing {', '.join(missing)}")

        return cls(
            raw["model_type"],
            raw["hyperparameters"],
            raw["learned"],
            raw["format_version"],
        )

    def __repr__(self) -> str:
        return (
            f"ModelDocument({self._model_type!r}, "
            f"format_version={self._format_version}, "
            f"learned_parts={sorted(self._learned)})"
        )
