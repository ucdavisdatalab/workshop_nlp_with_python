#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
get a selection of genres from the wikipedia movie plot summaries
"""

from argparse import ArgumentParser
import os
import pandas as pd

def format_genre(genre):
    genre = eval(genre)
    return list(genre.values())

def get_selected(df, select, samp):
    df = df.explode('genres')

    selected = pd.DataFrame()
    for s in select:
        s = df[df['genres']==s]
        selected = selected.append(s)

    selected = (
        selected
        .drop_duplicates('wiki_id')
        .sample(frac=samp)
    )
    return selected[['wiki_id', 'name', 'genres', 'summary']]

def main(args):
    metadata = pd.read_csv(
        os.path.join(args.indir, 'movie.metadata.tsv'),
        sep='\t',
        header=None,
        names=['wiki_id', 'freebase_id', 'name', 'date', 'box', 'runtime', 'lang', 'countries', 'genres']
    )
    plots = pd.read_csv(
        os.path.join(args.indir, 'plot_summaries.txt'),
        sep='\t',
        header=None,
        names=['wiki_id', 'summary']
    )

    df = metadata.merge(plots, on='wiki_id')
    df = df.assign(genres=df['genres'].apply(format_genre))
    df = df[df['genres'].apply(len)!=0]
    
    selected = get_selected(df, args.select, args.samp)
    for idx in selected.index:
        path = os.path.join(args.outdir, str(selected.at[idx, 'wiki_id']) + '.txt')
        with open(path, 'w') as f:
            f.write(selected.at[idx, 'summary'])

    selected = (
        selected
        .assign(fname = selected['wiki_id'].apply(str) + '.txt')
        .sort_values('fname')
        .reset_index(drop=True)
    )
    selected[['wiki_id', 'name', 'genres', 'fname']].to_csv(args.manifest)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--indir',
        type=str
    )
    parser.add_argument(
        '--select',
        nargs='+',
        type=str
    )
    parser.add_argument(
        '--samp',
        type=float
    )
    parser.add_argument(
        '--outdir',
        type=str
    )
    parser.add_argument(
        '--manifest',
        type=str
    )
    args = parser.parse_args()
    main(args)
