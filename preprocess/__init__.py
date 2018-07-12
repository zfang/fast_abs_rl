import json
import logging
import os
import re
import string
from collections import Counter, OrderedDict

import numpy as np
import spacy
from commonregex import CommonRegex
from sklearn.preprocessing import normalize

from preprocess.pos_tags import compute_pos_tag_chisquare, POS_TAGS
from preprocess.smooth_inverse_frequency import get_sif_embeddings

PATTERNS = [(re.compile('\s+'), ' ')] + \
           [(re.compile('(\s*{})+'.format(re.escape(p))), '{}'.format(p)) for p in string.punctuation if p != '\\']

dm_single_close_quote = '\u2019'  # unicode
dm_double_close_quote = '\u201d'
# acceptable ways to end a sentence
END_TOKENS = set(list(string.punctuation) + ['...', dm_single_close_quote, dm_double_close_quote])

SPACY_TAGGER = spacy.load('en_core_web_sm', diable=['ner', 'parser'])
SPACY = spacy.load('en_core_web_sm', diable=['ner', 'tagger', 'parser'])

LOGGER = logging.getLogger(__name__)


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if not line:
        return line
    if line[-1] in END_TOKENS:
        return line
    if isinstance(line, str):
        return line + " ."
    if isinstance(line, list):
        return line + ["."]
    raise NotImplementedError(line)


def contains_personal_info(text):
    parsed_text = CommonRegex(text)
    return any([bool(parsed_text.dates),
                bool(parsed_text.times),
                bool(parsed_text.phones),
                bool(parsed_text.links),
                bool(parsed_text.emails),
                bool(parsed_text.ips),
                bool(parsed_text.ipv6s),
                bool(parsed_text.prices),
                bool(parsed_text.hex_colors),
                bool(parsed_text.credit_cards),
                bool(parsed_text.btc_addresses),
                bool(parsed_text.street_addresses),
                ])


def clean_text(text):
    try:
        text = text.lower()
        text = re.sub(r'\\$', '', text)
        text = text.encode('utf8').decode('unicode_escape')
        for pattern, to_replace in PATTERNS:
            text = pattern.sub(to_replace, text)
    finally:
        return text


def maximal_marginal_relevance_sorted(docs, lambda_, relevance_scores, similarity_matrix, limit=0):
    assert 0 <= lambda_ <= 1
    assert len(docs) == len(relevance_scores) == similarity_matrix.shape[0] == similarity_matrix.shape[1]

    indices = set(range(len(docs)))
    selected = OrderedDict()
    limit = limit or len(docs)
    if limit < 0:
        limit = max(len(docs) + limit, 1)
    limit = min(len(docs), limit)

    while len(selected) < limit:
        remaining = list(indices - selected.keys())

        def mmr_score(x):
            score = lambda_ * relevance_scores[x]
            if lambda_ < 1:
                score -= (1 - lambda_) * max([similarity_matrix[x, y] for y in selected.keys()] or [0])
            return score

        scores = list(map(mmr_score, remaining))
        next_selected = remaining[np.argmax(scores)]
        selected[next_selected] = True

    return np.asarray(docs)[list(selected.keys())].tolist()


def preprocess(texts,
               token_threshold=5,
               limit=0,
               lambda_=0.4,
               relevance_scores=None,
               pos_tag_distribution=None,
               pos_tag_chisq_critical_value=1e-3):
    relevance_scores = relevance_scores or np.zeros(len(texts))

    found_sentences = set()

    docs = []
    lemmas = []
    scores = []

    min_threshold = token_threshold
    max_threshold = 15 * token_threshold

    pos_tagging = pos_tag_distribution is not None and pos_tag_chisq_critical_value > 0
    if pos_tagging:
        parser = SPACY_TAGGER
    else:
        parser = SPACY

    for doc, score in zip(parser.pipe(map(clean_text, texts), n_threads=os.cpu_count(), batch_size=10000),
                          relevance_scores):
        if doc.text in found_sentences:
            log_removed(doc.text, 'duplicate')
            continue
        found_sentences.add(doc.text)

        if not (min_threshold <= len(doc) <= max_threshold):
            log_removed(doc.text, 'token count {} is out of range [{}, {}]'.format(
                len(doc),
                min_threshold,
                max_threshold))
            continue

        if contains_personal_info(doc.text):
            log_removed(doc.text, 'personal info')
            continue

        if pos_tagging:
            tag_obs = Counter()
            for token in doc:
                tag_obs[token.pos_] += 1
            total = sum(tag_obs.values())
            tag_exp = {tag: pos_tag_distribution[tag] * total for tag in POS_TAGS}
            _, p = compute_pos_tag_chisquare(tag_obs, tag_exp)
            if p < pos_tag_chisq_critical_value:
                log_removed(doc.text, 'POS tag p value {} < {}'.format(p, pos_tag_chisq_critical_value))
                continue

        raw_tokens = list(filter(lambda t: bool(t.text.strip()), doc))
        tokens = [token.text for token in raw_tokens]
        tokens = fix_missing_period(tokens)

        docs.append(tokens)
        lemmas.append([token.lemma_ for token in doc])
        scores.append(score)

    if not docs:
        return docs

    sif_embeddings = get_sif_embeddings(lemmas)
    normalized_sif_embeddings = normalize(sif_embeddings, axis=1)
    similarity_matrix = np.dot(normalized_sif_embeddings, np.transpose(normalized_sif_embeddings))
    mmr_sorted = maximal_marginal_relevance_sorted(
        docs=docs,
        lambda_=lambda_,
        relevance_scores=scores,
        similarity_matrix=similarity_matrix,
        limit=limit)

    return [' '.join(sent) for sent in mmr_sorted]


def log_removed(sentence, reason):
    LOGGER.debug(json.dumps({'removed': sentence, 'reason': reason}, ensure_ascii=False))
