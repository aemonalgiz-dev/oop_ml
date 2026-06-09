"""One printable symbol for each of the 256 byte values, so bytes can be tokens.

Why bytes need names
--------------------
A byte-level tokenizer's alphabet is the 256 byte values, and a token in this
library is a non-empty *string*: the vocabulary is read by people, written into
saved documents and compared in tests. Many byte values are not printable
(``0x00`` to ``0x1f``), some are whitespace that a text editor collapses
(``0x20``), and the values above ``0x7f`` are not characters at all until a
decoder says which encoding they are in. So each byte needs a name that is one
visible character, and the 256 names have to be distinct.

The mapping GPT-2 chose
-----------------------
Radford et al. (2019) took the 188 byte values that are already printable,
visible Latin-1 characters -- ``!`` to ``~``, ``¡`` to ``¬`` and ``®`` to
``ÿ`` -- and let each stand for itself, so ``b"hello"`` is spelled ``hello``.
The other 68 values, the control characters, the space, the soft hyphen and
``0x7f``, were assigned the codepoints ``256`` upward in byte order, which puts
them in a block of Latin Extended letters that no Latin-1 byte occupies. The
space becomes ``Ġ`` (``U+0120``), which is why a GPT-2 token that begins a word
looks like ``Ġhello``. The whole mapping is a bijection, so it is undone exactly.

It is reproduced here rather than invented afresh because a byte-level
vocabulary written with these names can be read beside one of GPT-2's, and
because the choice is arbitrary in every respect except that it is fixed.
"""

from __future__ import annotations

from oop_ml.core.exceptions import InvalidValuesError

# Latin-1 codepoints that are printable and not whitespace, as three ranges.
_SELF_REPRESENTING: tuple[range, ...] = (
    range(ord("!"), ord("~") + 1),
    range(ord("¡"), ord("¬") + 1),
    range(ord("®"), ord("ÿ") + 1),
)


def _byte_symbols() -> tuple[str, ...]:
    keeps_itself = {value for block in _SELF_REPRESENTING for value in block}
    symbols: list[str] = []
    next_codepoint = 256
    for value in range(256):
        if value in keeps_itself:
            symbols.append(chr(value))
        else:
            symbols.append(chr(next_codepoint))
            next_codepoint += 1
    return tuple(symbols)


BYTE_SYMBOLS: tuple[str, ...] = _byte_symbols()
"""Position ``b`` is the one-character name of byte value ``b``."""

_BYTES_BY_SYMBOL: dict[str, int] = {
    symbol: value for value, symbol in enumerate(BYTE_SYMBOLS)
}


def symbols_of_text(text: str) -> tuple[str, ...]:
    """The text's UTF-8 bytes, each as its symbol."""
    return tuple(BYTE_SYMBOLS[value] for value in text.encode("utf-8"))


def byte_of_symbol(symbol: str) -> int:
    """The byte value a symbol names.

    Raises
    ------
    InvalidValuesError
        If ``symbol`` is not one of the 256 names.
    """
    value = _BYTES_BY_SYMBOL.get(symbol)
    if value is None:
        raise InvalidValuesError(f"{symbol!r} is not the name of a byte value")
    return value


def text_of_symbols(symbols: str) -> str:
    """The text whose UTF-8 bytes are named, in order, by the characters of
    ``symbols``.

    A run of bytes that is not valid UTF-8 -- which a model can emit, since it
    chooses ids one at a time -- decodes to the replacement character rather
    than raising, because a decoder that refuses would refuse most of a
    half-finished generation.

    Raises
    ------
    InvalidValuesError
        If a character of ``symbols`` names no byte.
    """
    return bytes(byte_of_symbol(symbol) for symbol in symbols).decode(
        "utf-8", errors="replace"
    )
