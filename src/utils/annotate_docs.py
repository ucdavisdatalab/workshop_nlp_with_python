#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import json

import spacy

def lazy_load(paths: list[Path]) -> str:
    """Yield out documents."""
    for path in paths:
        doc = path.open('r')
        yield doc.read()

def main(args: argparse.Namespace) -> None:
    """Run the script."""
    paths = list(args.indir.glob("*.txt"))
    docs = NLP.pipe(lazy_load(paths))

    for path, doc in zip(paths, docs):
        outpath = args.outdir.joinpath(path.name).with_suffix('.json')
        with outpath.open('w') as fout:
            json.dump(doc.to_json(), fout)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=Path, help='Plaintext directory')
    parser.add_argument('--outdir', type=Path, help='Output directory')
    parser.add_argument('--model', type=str, default='en_core_web_md')
    args = parser.parse_args()

    NLP = spacy.load(args.model)
    main(args)
