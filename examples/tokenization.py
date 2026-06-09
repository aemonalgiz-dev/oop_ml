"""From text to the ids a model reads, four ways, on one small corpus.

A model reads numbers, so every text model begins with a lookup table, and the
whole question of tokenization is what the rows of that table should be. Whole
words make the table enormous and leave every rare word unknown; single
characters make it tiny and leave every token meaningless on its own. The
subword tokenizers sit between the two, and they disagree about *how* to sit
there.

This script fits four of them to the same corpus and asks each the same two
questions: how does it spell a word it has seen, and how does it spell a word
it has not. Byte pair encoding merges the most frequent adjacent pair, over and
over. WordPiece merges the pair whose symbols appear together more than their
frequencies predict. The unigram model starts from far too many pieces and
prunes the ones the corpus can most afford to lose. The byte tokenizer learns
nothing at all and can therefore never meet an unknown word, at the cost of
three tokens for one accented character.

Two things to notice in the output. Every learned tokenizer spells the unseen
word ``lowest`` out of pieces the corpus taught, with no unknown token, because
its alphabet covers the letters -- and the one word spelled with a letter the
corpus never used is where the unknown token finally appears. And the same
word decodes back to itself under every tokenizer, which is the property that
makes an encoding an encoding rather than a summary.
"""

from __future__ import annotations

import logging

from examples.reporting import Report, configure_logging_from_command_line
from oop_ml import (
    BytePairEncoding,
    ByteTokenizer,
    PassPurpose,
    PennTreebankPreTokenizer,
    UnigramLanguageModel,
    WhitespacePreTokenizer,
    WordPiece,
)

logger = logging.getLogger(__name__)

# Sennrich, Haddow and Birch's own toy corpus: low x5, lower x2, newest x6,
# widest x3. Small enough that every merge can be followed by hand.
CORPUS = [
    "low low low low low",
    "lower lower",
    "newest newest newest newest newest newest",
    "widest widest widest",
]

PROBES = ["newest", "lowest", "wider", "café"]

VOCABULARY_SIZE = 24


def main() -> None:
    report = Report(logger)

    report.heading("Where the words are, before anything is learned")
    sentence = 'Don\'t stop, "she" said.'
    report.line(f"text:         {sentence}")
    report.line(f"whitespace:   {list(WhitespacePreTokenizer().split(sentence).texts)}")
    report.line(
        f"Penn Treebank: {list(PennTreebankPreTokenizer().split(sentence).texts)}"
    )
    report.paragraph(
        "Splitting on spaces leaves the comma glued to stop and the quotes\n"
        "glued to she, so a vocabulary learned from prose would hold stop,\n"
        "stop, and stop. as three unrelated words. The Treebank rules peel\n"
        "punctuation and the clitic n't off, and every word still knows the\n"
        "span of the source it came from."
    )

    report.heading("Four vocabularies of the same size, from the same corpus")
    byte_pair = BytePairEncoding(vocabulary_size=VOCABULARY_SIZE).fit(CORPUS)
    word_piece = WordPiece(vocabulary_size=VOCABULARY_SIZE).fit(CORPUS)
    unigram = UnigramLanguageModel(vocabulary_size=VOCABULARY_SIZE, random_seed=0).fit(
        CORPUS
    )
    byte_level = ByteTokenizer()

    report.line(f"byte pair merges, in order: {[m.merged for m in byte_pair.merges]}")
    report.paragraph(
        "e and s merge first because they are adjacent in newest and widest,\n"
        "nine times between them; the tie with s and t</w>, also nine, goes\n"
        "to the lexicographically smaller pair, and that rule is written down\n"
        "once so the same corpus gives the same vocabulary in any text order."
    )

    tokenizers = [
        ("byte pair", byte_pair),
        ("WordPiece", word_piece),
        ("unigram", unigram),
        ("bytes", byte_level),
    ]

    rows = []
    for probe in PROBES:
        for label, tokenizer in tokenizers:
            encoding = tokenizer.encode(probe)
            rows.append(
                [
                    probe,
                    label,
                    str(len(encoding)),
                    " ".join(encoding.texts),
                    tokenizer.decode(encoding.ids),
                ]
            )
    report.table(["word", "tokenizer", "tokens", "pieces", "decoded"], rows)

    report.paragraph(
        "newest is one token to every learned tokenizer, because it is the\n"
        "commonest word. lowest never appeared, and each spells it from\n"
        "pieces it did learn, in two pieces or four depending on which\n"
        "merges the vocabulary happened to hold at this size. wider is the\n"
        "same story, and WordPiece alone kept wid from widest. café is\n"
        "spelled in letters the corpus never used at all, so the learned\n"
        "tokenizers answer their unknown token and lose the word on\n"
        "decoding, while the byte tokenizer spells the accented e as two\n"
        "bytes and gives every character back exactly: that is the whole\n"
        "trade a closed byte vocabulary makes, no unknowns for longer\n"
        "sequences."
    )

    report.heading("Regularising the model by varying the spelling")
    dropped = BytePairEncoding(
        vocabulary_size=VOCABULARY_SIZE, merge_dropout=0.5, random_seed=3
    ).fit(CORPUS)
    spellings = sorted(
        {
            " ".join(dropped.encode("newest", PassPurpose.TRAINING).texts)
            for _ in range(30)
        }
    )
    report.line(f"predicting: {' '.join(dropped.encode('newest').texts)}")
    report.line(f"training, 30 draws, {len(spellings)} distinct spellings:")
    for spelling in spellings:
        report.line(f"    {spelling}")
    report.paragraph(
        "Under merge dropout the model meets many spellings of one word while\n"
        "it learns, and so learns that est</w> and e s t</w> mean the same\n"
        "thing. The moment it predicts, every merge applies and the spelling\n"
        "is the one it should expect. The default purpose is predicting, for\n"
        "the same reason the dropout layer's is: forgetting to say training\n"
        "costs a little regularisation, and forgetting to say predicting\n"
        "would cost every answer."
    )


if __name__ == "__main__":
    configure_logging_from_command_line()
    main()
