import re
from collections import OrderedDict

import spacy

from .constants import DAYS, MONTHS, NAMES, USCITIES, NA_STATES, COUNTRIES, NATIONALITIES

TERMINAL_PUNCTUATION = list("""!,.:;?""")

PUNCTUATION_SPACE_PATTERN = re.compile(r'(\s+[{}])'.format(
    ''.join(re.escape(p) for p in TERMINAL_PUNCTUATION)))

WORDS_TO_CAPITALIZE_MAP = {word.lower(): word for word in set([w.capitalize() for w in {'i'} | DAYS | MONTHS])
                           | NAMES | USCITIES | NA_STATES | COUNTRIES | NATIONALITIES}

WORDS_TO_CAPITALIZE_STOP_WORDS = {'mobile'}
for word in WORDS_TO_CAPITALIZE_STOP_WORDS:
    del WORDS_TO_CAPITALIZE_MAP[word]

WORDS_TO_CAPITALIZE_PATTERN = re.compile(r'\b({})\b'.format('|'.join(WORDS_TO_CAPITALIZE_MAP.keys())))

CONTRACTION_SUFFIX_PATTERN = re.compile(r'\b({})\b'.format('|'.join(map(lambda x: r'\s+' + x, (
    "n\'t",
    "\'re",
    "\'s",
    "\'d",
    "\'ll"
    "\'t",
    "\'ve",
    "\'m",
)))))

SPACY_PARSER = spacy.load('en_core_web_sm', diable=['ner', 'tagger'])


def postprocess(decoded_tokens, token_threshold):
    for dec in decoded_tokens:
        # remove terminal punctuations that appear as the first or second token
        for i in (0, 1):
            while i < len(dec) and dec[i] in TERMINAL_PUNCTUATION:
                dec.pop(i)
        # remove repetitive unigram
        for i in range(len(dec) - 1, 0, -1):
            if dec[i] == dec[i - 1]:
                dec.pop(i)

    decoded_sentences = OrderedDict([(' '.join(dec), None) for dec in decoded_tokens
                                     if len(dec) >= token_threshold]).keys()
    capitalized_decoded_sentences = []
    for doc in SPACY_PARSER.pipe(decoded_sentences):
        capitalized_decoded_sentences.append(' '.join(sent.text.capitalize() for sent in doc.sents))
    decoded_sentences = capitalized_decoded_sentences

    for i in range(len(decoded_sentences)):
        decoded_sentences[i] = PUNCTUATION_SPACE_PATTERN.sub(lambda x: x.group(1).replace(" ", ""),
                                                             decoded_sentences[i])
        decoded_sentences[i] = WORDS_TO_CAPITALIZE_PATTERN.sub(lambda x: WORDS_TO_CAPITALIZE_MAP[x.group(1)],
                                                               decoded_sentences[i])
        decoded_sentences[i] = CONTRACTION_SUFFIX_PATTERN.sub(lambda x: x.group(1).replace(" ", ""),
                                                              decoded_sentences[i])

    return decoded_sentences
