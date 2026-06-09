"""The two contracts every tokenizer here keeps, and the templates that keep them.

Two jobs, two bases
-------------------
Turning text into ids is two jobs that are routinely conflated. The first is
deciding where the words are, which is a rule about the writing system: spaces
in English, nothing at all in Chinese, a dictionary or a statistical model where
there are no spaces to read. The second is deciding what a model's units are,
which is a rule about a corpus: the pieces a word is cut into so that a
vocabulary of forty thousand can spell any text at all.

:class:`PreTokenizer` is the first job. It is configured and never fitted, it
reads one text and answers with
:class:`~oop_ml.core.natural_language_processing.tokenization.words.Words`, and every
splitter and segmenter in the word-level and segmentation families is one.
:class:`Tokenizer` is the second. It owns a
:class:`~oop_ml.core.natural_language_processing.tokenization.vocabulary.Vocabulary`,
turns a text into an
:class:`~oop_ml.core.natural_language_processing.tokenization.encoding.Encoding` and
an encoding back into text. Most tokenizers learn their vocabulary and are
:class:`LearnedTokenizer`, which is also a
:class:`~oop_ml.core.base.estimator.Fittable`; the byte and codepoint ones have
nothing to learn and are plain :class:`Tokenizer`, because a guard that says
"fit me first" on an object with nothing to fit would be a small lie.

Why encode and decode are templates
-----------------------------------
Every concrete tokenizer knows how to cut a text into pieces and how to glue
pieces back into text. Nothing else about encoding varies. Looking each piece
up, substituting the unknown token for a piece the vocabulary lacks, refusing a
piece it lacks when there is no unknown token, refusing an id no token owns --
that is one rule, and a rule written once per tokenizer is a rule some
tokenizer eventually writes wrong. So :meth:`Tokenizer.encode` and
:meth:`Tokenizer.decode` are the templates, and a subclass supplies
:meth:`Tokenizer._pieces_of` and :meth:`Tokenizer._text_from` and nothing else.
The same shape as :meth:`~oop_ml.core.network.layer.Layer.respond_to`.

Why encode takes a purpose
--------------------------
Two tokenizers behave differently while a model is learning. Byte pair
encoding with merge dropout skips merges at random so the model meets many
spellings of one word, and the unigram model samples a segmentation rather than
taking its best one. Both regularise the *model*, and both must be switched off
the moment the model is asked to predict, or the same word arrives in a
different form every time and the answer is noise. That is precisely the
question :class:`~oop_ml.core.network.purpose.PassPurpose` already asks of a
dropout layer, so it is asked here in the same words, with the same default:
forgetting to say ``TRAINING`` costs a little regularisation, forgetting to say
``PREDICTING`` costs every answer, and the default protects against the worse
mistake. A deterministic tokenizer ignores the argument.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Self

from pydantic import BaseModel, ConfigDict

from oop_ml.core.base.estimator import Fittable
from oop_ml.core.exceptions import InvalidValuesError, UnknownTokenError
from oop_ml.core.natural_language_processing.tokenization.encoding import (
    Encoding,
    Token,
)
from oop_ml.core.natural_language_processing.tokenization.vocabulary import Vocabulary
from oop_ml.core.natural_language_processing.tokenization.words import Words
from oop_ml.core.network.purpose import PassPurpose


def checked_text(text: object) -> str:
    """The one place a text is confirmed to be a string.

    A bytes object or a list of strings reaching a splitter would iterate as
    integers or as whole words, and both fail somewhere deep inside a rule
    rather than at the boundary where the mistake was made.

    Raises
    ------
    InvalidValuesError
        If ``text`` is not a ``str``.
    """
    if not isinstance(text, str):
        raise InvalidValuesError(
            f"text must be a str, got {type(text).__name__}; a corpus of several "
            f"texts belongs to fit, and one text at a time to split or encode"
        )
    return text


class PreTokenizer(BaseModel, ABC):
    """Decides where the words are. Configured, never fitted.

    A pydantic model so that a rule's parameters -- a pattern, an exception
    list, a dictionary -- are validated at construction, where every other
    hyperparameter in this library is. :meth:`split` is the template and a
    subclass supplies :meth:`_words_of`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    def split(self, text: str) -> Words:
        """The words of ``text``, in source order, each with its span.

        Raises
        ------
        InvalidValuesError
            If ``text`` is not a string.
        """
        return self._words_of(checked_text(text))

    @abstractmethod
    def _words_of(self, text: str) -> Words:
        """The rule itself, on a text already known to be a string."""


class Tokenizer(BaseModel, ABC):
    """Turns text into ids against a closed vocabulary, and ids back into text.

    :meth:`encode` and :meth:`decode` are templates. A subclass supplies the
    cut, :meth:`_pieces_of`, and the glue, :meth:`_text_from`, and the lookups
    in both directions happen here, once.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @property
    @abstractmethod
    def vocabulary(self) -> Vocabulary:
        """Every token this tokenizer can produce, in id order."""

    def encode(
        self, text: str, purpose: PassPurpose = PassPurpose.PREDICTING
    ) -> Encoding:
        """The tokens ``text`` becomes, each with its id.

        A piece the vocabulary lacks becomes the unknown token, text and id
        both, so the encoding always describes what the model will read.

        Parameters
        ----------
        text:
            One text.
        purpose:
            Whether a model is learning from the answer. A stochastic
            tokenizer regularises only under ``TRAINING``; a deterministic one
            ignores this.

        Raises
        ------
        InvalidValuesError
            If ``text`` is not a string.
        UnknownTokenError
            If a piece is absent from a vocabulary that has no unknown token.
        """
        vocabulary = self.vocabulary
        tokens: list[Token] = []
        for piece in self._pieces_of(checked_text(text), purpose):
            if piece in vocabulary:
                tokens.append(Token(piece, vocabulary.id_of(piece)))
            elif vocabulary.unknown_token is not None:
                tokens.append(Token(vocabulary.unknown_token, vocabulary.id_of(piece)))
            else:
                raise UnknownTokenError(
                    f"{type(self).__name__} produced the piece {piece!r}, which "
                    f"its closed vocabulary does not hold"
                )
        return Encoding(tokens)

    def decode(self, token_ids: Sequence[int]) -> str:
        """The text a run of ids stands for.

        Raises
        ------
        UnknownTokenError
            If an id names no token.
        """
        return self._text_from(self.vocabulary.tokens_of(token_ids))

    @abstractmethod
    def _pieces_of(self, text: str, purpose: PassPurpose) -> tuple[str, ...]:
        """Cut a text, already known to be a string, into vocabulary spellings.

        A piece the vocabulary lacks may be returned as is; the template turns
        it into the unknown token.
        """

    @abstractmethod
    def _text_from(self, pieces: Sequence[str]) -> str:
        """Glue vocabulary spellings back into text, undoing any markers."""


class LearnedTokenizer(Tokenizer, Fittable):
    """A tokenizer whose vocabulary is learned from a corpus.

    Construction configures the target size and the rules; ``fit`` learns the
    vocabulary; ``encode`` before ``fit`` raises
    :class:`~oop_ml.core.exceptions.NotFittedError` through the fitted-state
    machinery every model in this library shares.
    """

    @abstractmethod
    def fit(self, corpus: Sequence[str]) -> Self:
        """Learn a vocabulary from ``corpus`` and return ``self``.

        Implementations should validate the corpus, learn into locals, assign
        the private attributes at the end, then call ``self._mark_fitted()``,
        so that a fit which raises part-way leaves no half-fitted tokenizer.
        """
