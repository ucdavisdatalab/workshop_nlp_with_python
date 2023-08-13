#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
print out the genre counts for the wikipedia movie summaries
"""

from argparse import ArgumentParser
import os
import pandas as pd

def format_genre(genre):
    genre = eval(genre)
    return list(genre.values())

def main(args):
    metadata = pd.read_csv(
        os.path.join(args.indir, 'movie.metadata.tsv'),
        sep='\t',
        header=None,
        names=['wiki_id', 'freebase_id', 'name', 'date', 'box', 'runtime', 'lang', 'countries', 'genres']
    )

    metadata = metadata.assign(genres=metadata['genres'].apply(format_genre))
    metadata = metadata.explode('genres')
    print(metadata.value_counts('genres', ascending=False).head(50))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--indir',
        type=str
    )
    args = parser.parse_args()
    main(args)
