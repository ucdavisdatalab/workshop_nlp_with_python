#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import spacy
import glob
import os

def lazy_load(paths):
    for path in paths:
        doc = open(path, 'r')
        yield doc.read()

def clean(doc):
    cleaned = []
    for token in doc:
        if token.is_alpha:
            if token.is_stop == False and len(token) > 2:
                token = token.lemma_
                token = token.lower()
                cleaned.append(token)

    return ' '.join(cleaned)

# OPTIONAL
def convert_person(doc):
    tokens = [token.text for token in doc]
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            start = ent.start
            end = ent.end
            
            for idx in range(start, end):
                tokens[idx] = 'PERSON'
    
    result = []
    for idx, token in enumerate(tokens):
        if token == 'PERSON':
            if idx == 0:
                result.append(token)
            elif tokens[idx-1] == 'PERSON':
                continue
            else:
                result.append(token)
        else:
            result.append(token)
                
    result = ' '.join(result)
    result = NLP(result)
    
    return result

# USE IF DOING ABOVE
def prepare_doc(doc):
    no_person = convert_person(doc)
    cleaned = clean(doc)
    return cleaned

def main(args):
    paths = glob.glob(args.indir + '/*.txt')
    paths.sort()

    corpus = [clean(doc) for doc in NLP.pipe(lazy_load(paths))]
    for doc, path in zip(corpus, paths):
        basename = os.path.basename(path)
        outpath = os.path.join(args.outdir, basename)
        with open(outpath, 'w') as fout:
            fout.write(doc)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--indir',
        type=str
    )
    parser.add_argument(
        '--outdir',
        type=str
    )
    args = parser.parse_args()
    NLP = spacy.load('en_core_web_md')
    main(args)
