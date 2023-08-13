#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Get a selection of genres from a directory of Wikipedia movie plot summaries
"""

from argparse import ArgumentParser
import os
import pandas as pd

def format_genre(genre):
    """Evaluate a summary's genres as a list."""
    genre = eval(genre)
    genre = list(genre.values())

    return genre

def get_selected(df, select, samp):
    """Get the selected genres, sampling down as needed."""
    df = df.explode('genres')

    # Compile each genre-specific subset of the corpus
    selected = [df[df['genres']==s] for s in select]
    selected = pd.concat(selected)

    # Drop duplicates, taking the first genre for a plot as the only genre. Then sample the
    # resultant dataframe to a fraction of the total number of selected plots
    selected = (
        selected
        .drop_duplicates('wiki_id')
        .sample(frac=samp)
    )
    # Take only a few columns
    selected = selected[['wiki_id', 'name', 'genres', 'summary']]

    return selected

def main(args):
    # Build a metadata dataframe from the directory
    metadata = pd.read_csv(
        os.path.join(args.indir, 'movie.metadata.tsv'),
        sep='\t',
        header=None,
        names=['wiki_id', 'freebase_id', 'name', 'date', 'box', 'runtime', 'lang', 'countries', 'genres']
    )
    # Read in the plots
    plots = pd.read_csv(
        os.path.join(args.indir, 'plot_summaries.txt'),
        sep='\t',
        header=None,
        names=['wiki_id', 'summary']
    )

    # Merge the metadata and plots on their Wiki ID. Format genres and remove empty genres
    df = metadata.merge(plots, on='wiki_id')
    df = df.assign(genres=df['genres'].apply(format_genre))
    df = df[df['genres'].apply(len)!=0]
    
    # Run the selection and then save each plot summary to an output directory
    selected = get_selected(df, args.select, args.samp)
    print(f"Selected {len(selected)} summaries.") 
    for idx in selected.index:
        path = os.path.join(args.outdir, str(selected.at[idx, 'wiki_id']) + '.txt')
        with open(path, 'w') as f:
            f.write(selected.at[idx, 'summary'])

    # Create a manifest
    selected = (
        selected
        .assign(fname = selected['wiki_id'].astype(str) + '.txt')
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
