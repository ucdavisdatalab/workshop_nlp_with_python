#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import json

import spacy
from spacy.tokens import Doc
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def load_doc(path: Path) -> Doc:
    """Load a json file as a spaCy doc."""
    with path.open('r') as fin:
        doc = json.load(fin)
        doc = Doc(NLP.vocab).from_json(doc)

        return doc

def clean(doc: Doc) -> str:
    """Clean a document using spaCy features."""
    cleaned = []
    for tok in doc:
        if not tok.is_alpha:
            continue
        if not tok.is_stop and len(tok) > 2:
            tok = tok.lemma_
            cleaned.append(tok.lower())

    return ' '.join(cleaned)


def main(args: argparse.Namespace) -> None:
    """Run the script."""
    manifest = pd.read_csv(args.manifest, index_col = 0)

    docs = []
    for fname in manifest['file_name']:
        doc = load_doc(args.indir.joinpath(fname))
        docs.append(doc)

    cleaned = [clean(doc) for doc in docs]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(cleaned)
    tfidf = pd.DataFrame(
        tfidf.toarray()
        , index = manifest['file_name']
        , columns = vectorizer.get_feature_names_out()
    )
    tfidf.to_csv(args.outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=Path, help='Plaintext directory')
    parser.add_argument('--manifest', type=Path, help='Data manifest')
    parser.add_argument('--outfile', type=Path, help='Output tf-idf')
    parser.add_argument('--model', type=str, default='en_core_web_md')
    args = parser.parse_args()

    NLP = spacy.load(args.model)
    main(args)
