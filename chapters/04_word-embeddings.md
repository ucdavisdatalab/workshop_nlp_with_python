---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{code-cell}
:tags: [remove-cell]
import os

os.chdir("..")
```

Word Embeddings
===============

Our sessions so far have worked off the idea of document annotation to produce
metadata about texts. We've used this information for everything from
information retrieval tasks (Chapter 2) to predictive classification (Chapter
3). Along the way, we've also made some passing discussions about how such
annotations work to quantify or identify the semantics of those tasks (our work
with POS tags, for example). But what we haven't yet done is produce a model of
semantic meaning ourselves. This is another core task of NLP, and there are
several different ways to approach building a statistical representation of
tokens' meanings. The present chapter discusses one of the most popular methods
of doing so: **word embeddings**. Below, we'll overview what word embeddings
are, demonstrate how to build and use them, talk about important considerations
regarding bias, and apply all this to a document clustering task.

The corpus we'll use is Melanie Walsh’s [collection][walsh] of ~380 obituaries from
the _New York Times_. If you participated in our Getting Started with Textual
Data series, you'll be familiar with this corpus: we used it [in the context of
tf-idf scores][tfidf]. Our return to it here is meant to chime with that
discussion, for word embeddings enable us to perform a similar kind of text
vectorization. Though, as we'll discuss, the resultant vectors will be
considerably more feature-rich than what we could achieve with tf-idf alone.

[walsh]: https://melaniewalsh.github.io/Intro-Cultural-Analytics/00-Datasets/00-Datasets.html
[tfidf]: https://ucdavisdatalab.github.io/workshop_getting_started_with_textual_data/05_clustering-and-classification.html

```{admonition} Learning objectives
By the end of this chapter, you will be able to:

+ Explain what word embeddings are
+ Use `gensim` to train and load word embeddings models
+ Identify and analyze word relationships in these models
+ Recognize how bias can inhere in embeddings
+ Encode documents with a word embeddings model
```

How It Works
------------

Prior to the advent of [Transformer][transformer] models, word embedding served
as a state-of-the-art technique for representing semantic relationships between
tokens. The technique was first introduced in 2013, and it spawned a host of
different variants that completely flooded the field of NLP until about 2018.
In part, word embedding's popularity stems from the relatively simple intuition
behind it, which is known as the **distributional hypothesis**: "you shall know
a word by the company it keeps!" (J.R. Firth). Words that appear in similar
contexts, in other words, have similar meanings, and what word embeddings do is
represent that context-specific information through a set of features. As a
result, similar words share similar data representations, and we can leverage
that similarity to explore the semantic space of a corpus, to encode documents
with feature-rich data, and more.

[transformer]: https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)

If you're familiar with tf-idf vectors, the underlying data structure of word
embeddings is the same: every word is represented by a vector of features. But
a key difference lies in the **sparsity** of the vectors – or, in the case of
word embeddings, the _lack_ of sparsity. As we saw in the last chapter, tf-idf
vectors can suffer from the [curse of dimensionality][curse], something that's
compounded by the fact that such vectors must contain features for every word
in corpus, regardless of whether a document has that word. This means tf-idf
vectors are highly sparse: they contain many 0s. Word embeddings, on the other
hand, do not. They're what we call **dense** representations. Each one is a
fixed-length, non-sparse vector (of 50-300 dimensions, usually) that is much
more information-rich than tf-idf. As a result, embeddings tend to be capable
of representing more nuanced relationships between corpus words – a performance
improvement that is further boosted by the fact that many of the most popular
models had the advantage of being trained on billions and billions of tokens.

[curse]: https://en.wikipedia.org/wiki/Curse_of_dimensionality

The other major difference between these vectors and tf-idf lies in how the
former are created. While at root, word embeddings represent token
co-occurrence data (just like a document-term matrix), they are the product of
millions of guesses made by a neural network. Training this network involves
making predictions about a target word, based on that word's context. We are
not going to delve into the math behind these predictions (though [this
post][post] does); however, it is worth noting that there are two different
training set ups for a word embedding model:

[post]: https://medium.com/analytics-vidhya/maths-behind-word2vec-explained-38d74f32726b

```{margin} For more on CBOW vs. skip-gram
Check out this blog post, [Words as Vectors][wv].

[wv]: https://iksinc.online/tag/continuous-bag-of-words-cbow/
```

1. **Common Bag of Words (CBOW)**: given a window of words on either side of a
   target, the network tries to predict what word the target should be
2. **Skip-grams**: the network starts with the word in the middle of a window
   and picks random words within this window to use as its prediction targets

As you may have noticed, these are just mirrored versions of one another. CBOW
starts from context, while skip-gram tries to rebuild context. Regardless, in
both cases the network attempts to maximize the likelihood of its predictions,
updating its weights accordingly over the course of training. Words that
repeatedly appear in similar contexts will help shape thse weights, and in turn
the model will associate such words with similar vector representations. If
you'd like to see all this in action, Xin Rong has produced a [fantastic
interactive visualization][vis] of how word embedding models learn.

[vis]: https://ronxin.github.io/wevi/

Of course, the other way to understand how word embeddings work is to use them
yourself. We'll move on to doing so now.

Preliminaries
-------------

Here are the libraries we will use in this chapter.

```{code-cell}
from pathlib import Path
from collections import Counter
from tabulate import tabulate
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
```

We also initialize an input directory and load a file manifest.

```{code-cell}
indir = Path("data/session_three")

manifest = pd.read_csv(indir.joinpath("manifest.csv"), index_col = 0)
manifest.info()
```

And finally we'll load the obituaries. While the past two sessions have
required full-text representations of documents, word embeddings work best with
bags of words, especially when it comes to doing analysis with them.
Accordingly, each of the files in the corpus have already processed by a text
cleaning pipeline: they represent the lowercase, stopped, and lemmatized
versions of the originals.

```{code-cell}
corpus = []
for fname in manifest['file_name']:
    with indir.joinpath(f"obits/{fname}").open('r') as fin:
        doc = fin.read()
        corpus.append(doc.split())
```

With this time, it's time to move to the model.

Using an Embeddings Model
-------------------------

At this point, we are at a crossroads. On the one hand, we could train a word
embeddings model using our corpus documents as is. The `gensim` library offers
functionality for this, and it's a relatively easy operation. On the other, we
could use pre-made embeddings, which are usually trained on a more general –
and much larger – set of documents. There is a trade-off here:

+ Training a corpus-specific model will more faithfully represent the token
  behavior of the texts we'd like to analyze, but these representations could
  be _too_ specific, especially if the model doesn't have enough data to train
  on; the resultant embeddings may be closer to topic models than to word-level
  semantics
+ Using pre-made embeddings gives us the benefit of generalization: the vectors
  will cleave more closely to how we understand language; but such embeddings
  might a) miss out on certain nuances we'd like to capture, or b) introduce
  biases into our corpus (more on this below)

In our case, the decision is difficult. When preparing this reader, we (Tyler
and Carl) found that a model trained on the obituaries alone did not produce
vectors that could fully demonstrate the capabilities of the word embedding
technique. The corpus is just a little too specific, and perhaps a little too
small. We could've used a larger corpus, but doing so would introduce
slow-downs in the workshop session. Because of this, we've gone with a pre-made
model: the Stanford [GloVe][glove] embeddings (the 200-dimension version).
GloVe was trained on billions of tokens, spanning Wikipedia data, newswire
articles, even Twitter. More, the model's developers offer several different
dimension sizes, which are helpful for selecting embeddings with the right
amount of detail.

[glove]: https://nlp.stanford.edu/projects/glove/

That said, going with GloVe introduces its own problems. For one thing, we
can't show you how to train a word embeddings model itself – at least not live.
The code to do so, however, is reproduced below:

```{margin} Model parameters
There are many different parameters to select from in `gensim`. You can find
them in the [Word2Vec documentation][gensim].

[gensim]: https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
```

```python
from gensim.models import Word2Vec

n_dimensions = 100
model = Word2Vec(n_dimensions)
model.build_vocab(corpus)
model.train(corpus, total_words = model.corpus_total_words, epochs = 5)
```

Another problem has to do with the data GloVe was trained on. It's so large
that we can't account for all the content, and this becomes particularly
detrimental when it comes to bias. [Researchers have found][found] that general
embeddings models reproduce gender-discriminatory language, even hate speech,
by virtue of the fact that they are trained on huge amounts of text data, often
without consideration of whether the content of such data is something one
would endorse. GloVe is [known to be biased][biased] in this way. We'll show an
example later on in this chapter and will discuss this in much more detail
during our live session, but for now just note that the effects of bias _do_
shape how we represent our corpus, and it's important to keep an eye out for
this when working with the data.

[found]: https://www.technologyreview.com/2016/07/27/158634/how-vector-space-mathematics-reveals-the-hidden-sexism-in-language/
[biased]: http://arxiv.org/abs/1607.06520

### Loading a model

With all that said, we can move on. Below, we load GloVe embeddings into our
workspace using a `gensim` wrapper.

```{code-cell}
model_path = indir.joinpath("glove/glove-wiki-gigaword_200d.bin")
model = KeyedVectors.load(model_path.as_posix())
```

The `KeyedVectors` object acts much like a Python dictionary, and you can do
certain Python operations directly on it.

```{code-cell}
print("Number of tokens in the model:", len(model))
```

### Token mappings

Each token in the model has an associated index. This mapping is accessible via
`.key_to_index`.

```{code-cell}
:tags: ["output_scroll"]
model.key_to_index
```

If you want the vector representation for a token, use either the key or the
index.

```{code-cell}
tok = np.random.choice(model.index_to_key)
idx = model.key_to_index[tok]
print(f"The index position for '{tok}' is {idx}")
```

Here's its vector:

```{code-cell}
:tags: ["output_scroll"]
model[idx]
```

Here are some random tokens in the model:

```{code-cell}
for tok in np.random.choice(model.index_to_key, 10):
    print(tok)
```

You may find some unexpected tokens in this output. Though it has been
ostensibly trained on an English corpus, GloVe contains multilingual text. It
also contains lots of noisy tokens, which range from erroneous segmentations
("drummer/percussionist" is one token, for example) to password-like strings
and even HTML markup. Depending on your task, you may not notice these tokens,
but they do in fact influence the overall shape of the model, and sometimes
you'll find them cropping up when you're hunting around for similar terms and
the like (more on this soon).

### Out-of-vocabulary tokens

While GloVe's vocabulary sometimes seems _too_ expansive, there are other
instances where it's too restricted.

```{code-cell}
:tags: ["raises-exception"]
assert 'unshaped' in model, "Not in vocabulary!"
```

If the model wasn't trained on a particular word, it won't have a corresponding
vector for that word either. This is crucial. Because models like GloVe only
know what they've been trained on, you need to be aware of any potential
discrepancies between their vocabularies and your corpus data. If you don't
keep this in mind, sending unseen, or **out-of-vocabulary**, tokens to GloVe
will throw errors in your code.

There are a few ways to handle this problem. The most common is to simply _not
encode_ tokens in your corpus that don't have a corresponding vector in GloVe.
Below, we construct two sets for our corpus data. The first contains all
tokens in the corpus, while the second tracks which of those tokens are in
the model. We identify whether the model has a token using its
`.has_index_for()` method.

```{code-cell}
vocab = set(tok for doc in corpus for tok in doc)
in_glove = set(tok for tok in vocab if model.has_index_for(tok))

print("Total words in the corpus vocabulary:", len(vocab))
print("Words not in GloVe:", len(vocab) - len(in_glove))
```

Any subsequent code we write will need to reference these sets to determine
whether it should encode a token.

While this is what we'll indeed do below, obviously it isn't an ideal
situation. But it's one of the consequences of using premade models. There are,
however, a few other ways to handle out-of-vocabulary terms. Some models offer
special "UNK" tokens, which you could associate with all of your problem
tokens. This, at the very least, enables you to have _some_ representation of
your data. A more complex approach involves taking the mean embedding of the
word vectors surrounding an unknown token; and depending on the model, you can
also train it further, adding extra tokens from your domain-specific text.
Instructions for this last option are available [here][train] in the `gensim`
documentation.

[train]: https://radimrehurek.com/gensim/models/word2vec.html#usage-examples

Word Relationships
------------------

Later on we'll use GloVe to encode our corpus texts. But before we do, it's
worth demonstrating more generally some of the properties of word vectors.
Vector representations of text allow us to perform various mathematical
operations on our corpus that approximate semantics. The most common among
these operations is finding the **cosine similarity** between two vectors. Our
Getting Started with Textual Data series has a whole [chapter][chapter] on this
measure, so if you haven't encountered it before, we recommend you read that.
But in short: cosine similarity measures the difference between vectors'
orientation in a feature space (here, the feature space is comprised of each of
the vectors' 200 dimensions). The closer two vectors are, the more likely they
are to share semantic similarities.

[chapter]: https://ucdavisdatalab.github.io/workshop_getting_started_with_textual_data/05_clustering-and-classification.html

### Similar tokens

`gensim` provides easy access to this measure and other such vector space
operations. To find the cosine similarity between the vectors for two words in
GloVe, simply use the model's `.similarity()` method:

```{code-cell}
a, b = 'calculate', 'compute'
print(f"Similarity of '{a}' and '{b}': {model.similarity(a, b):0.2f}")
```

The only difference between the score above and the one that you might produce,
say, with `scikit-learn`'s cosine similarity implementation is that `gensim`
bounds its values from `[-1,1]`, whereas the latter uses a `[0,1]` scale. While
in `gensim` it's still the case that similar words score closer to `1`, highly
dissimilar words will be closer to `-1`.

At any rate, we can get the top _n_ most similar words for a word using
`.most_similar()`. The function defaults to 10 entries, but you can change that
with the `topn` parameter. We'll wrap this in a custom function, since we'll
call it a number of times.

```{margin} Code explanation
As we'll see, a model has a few similarity functions, so we define a `func`
parameter that takes a callable and send that callable data from multiple
positional arguments. We also take into account the possibility of keyword
arguments.
```

```{code-cell}
def show_most_similar(*args, func = model.most_similar, **kwargs):
    """Print cosine similarities."""
    similarities = func(*args, **kwargs)
    print(tabulate(similarities, ['Word', 'Score']))
```

Now we sample some tokens and find their most similar tokens.

```{code-cell}
:tags: ["output_scroll"]
targets = np.random.choice(model.index_to_key, 5)
for tok in targets:
    print(f"Tokens most similar to '{tok}'\n")
    show_most_similar(tok)
    print("\n")
```

It's also possible to find the _least_ similar word. This is useful to show,
because it pressures our idea of what counts as similarity. Mathematical
similarity does not always align with concepts like synonyms and antonyms. For
example, it's probably safe to say that the semantic opposite of "good" – that
is, its antonym – is "evil." But in the world of vector spaces, the least
similar word to "good" is:

```{code-cell}
model.most_similar('good', topn = len(model))[-1]
```

Just noise! Relatively speaking, the vectors for "good" and "evil" are actually
quite similar.

```{code-cell}
a, b = 'good', 'evil'
print(f"Similarity of '{a}' and '{b}': {model.similarity(a, b):0.2f}")
```

How do we make sense of this? Well, it has to do with the way the word
embeddings are created. Since embeddings models are ultimately trained on
co-occurrence data, words that tend to appear in similar kinds of contexts will
be more similar in a mathematical sense than those that don't.

Keeping this in mind is also important for considerations of bias. Since, in
one sense, _embeddings reflect the interchangeability between tokens_, they
will reinforce negative, even harmful patterns in the data (which is to say in
culture at large). For example, consider the most similar words for "doctor"
and "nurse." The latter is locked up within gendered language: according to
GloVe, a nurse is like a midwife is like a mother.

```{code-cell}
for tok in ('doctor', 'nurse'):
    print(f"Tokens most similar to '{tok}'\n")
    show_most_similar(tok)
    print("\n")
```

### Concept modeling

Beyond cosine similarity, there are other word relationships to explore via
vector space math. For example, one way of modeling something like a _concept_
is to think about what other concepts comprise it. In other words: what plus
what creates a new concept? Could we identify concepts by adding together
vectors to create a new vector? Which words would this new vector be closest to
in the vector space? Using the `.similar_by_vector()` method, we can find out.

```{code-cell}
:tags: ["output_scroll"]
concepts = {
    'beach': ('sand', 'ocean'),
    'hotel': ('vacation', 'room'),
    'airplane': ('air', 'car')
}
for concept in concepts:
    a, b = concepts[concept]
    vector = model[a] + model[b]
    print(f"Tokens most similar to '{a}' + '{b}' (for '{concept}')\n")
    show_most_similar(vector, func = model.similar_by_vector)
    print("\n")
```

Not bad! While our target concepts aren't the most similar words for these
synthetic vectors, they're often in the top-10 most similar results.

### Analogies

Most famously, word embeddings enable quasi-logical reasoning. Though
relationships between antonyms and synonyms do not necessarily map to a vector
space, certain analogies do – at least under the right circumstances, and with
particular training data. The logic here is that we identify a relationship
between two words and we subtract one of those words' vectors from the other.
To that new vector we add in a vector for a target word, which forms the
analogy. Querying for the word closest to this modified vector should produce a
similar relation between the result and the target word as that between the
original pair.

Here, we ask: "strong is to stronger what clear is to X?" Ideally, we'd get
"clearer."

```{code-cell}
show_most_similar(
    func = model.most_similar,
    positive = ['stronger', 'clear'],
    negative = ['strong']
)
```

"Paris is to France what Berlin is to X?" Answer: "Germany."

```{code-cell}
show_most_similar(
    func = model.most_similar,
    positive = ['france', 'berlin'],
    negative = ['paris']
)
```

Both of the above produce compelling results, though your mileage may vary.
Consider the following: "arm is to hand what leg is to X?" We'd expect "foot."

```{code-cell}
show_most_similar(
    func = model.most_similar,
    positive = ['hand', 'leg'],
    negative = ['arm']
)
```

Importantly, these results are always going to be specific to the data on which
a model was trained. Claims made on the basis of word embeddings that aspire to
general linguistic truths would be treading on shaky ground here.

Document Similarity
-------------------

While the above word relationships are relatively abstract (and any such
findings therefrom should be couched accordingly), we can ground them with a
concrete task. In this final section, we use GloVe embeddings to encode our
corpus documents. This involves associating a word vector for each token in an
obituary. Of course, GloVe has not been trained on the obituaries, so there may
be important differences in token behavior between that model and the corpus;
but we can assume that the general nature of GloVe will give us a decent sense
of the overall feature space of the corpus. The result will be an enriched
representation of each document, the nuances of which may better help us
identify things like similarities between obituaries in our corpus.

The other consideration for using GloVe with our specific corpus concerns the
out-of-vocabulary words we've already discussed. Before we can encode our
documents, we need to filter out tokens for which GloVe has no representation.
We can do so by referencing the `in_glove` set we produced above.

```{code-cell}
pruned = []
for doc in corpus:
    keep = [tok for tok in doc if tok in in_glove]
    pruned.append(keep)
```

### Encoding

Time to encode. This is an easy operation. All we need to do is run the list of
document's tokens directly into the model object and `gensim` will encode each
accordingly. The result will be an `(n, 200)` array, where `n` is the number of
tokens we passed to the model; each one will have 200 dimensions.

But if we kept this array as is, we'd run into trouble. Matrix operations often
require identically shaped representations, so documents with different lengths
would be incomparable. To get around this, we take the mean of all the vectors
in a document. The result is a 200-dimension vector that stands as a general
representation of a document.

```{code-cell}
embeddings = [np.mean(model[doc], axis = 0) for doc in pruned]
embeddings = np.array(embeddings)
```

Let's quickly check our work.

```{code-cell}
print("Shape of an encoded document:", model[pruned[0]].shape)
print("Shape of a document vectour:", embeddings[0].shape)
```

### Visualizing

From here, we can use these embeddings for any task that requires feature
vectors. For example, let's plot our documents using t-SNE. First, we reduce
the embeddings.

```{margin} Not sure what t-SNE is?
Take a look at this [section][section] on visualizing vectors in Getting
Started With Textual Data.

[section]: https://ucdavisdatalab.github.io/workshop_getting_started_with_textual_data/chapters/05_clustering-and-classification.html#visualizing-scores
```

```{code-cell}
reducer = TSNE(
    n_components = 2,
    learning_rate = 'auto',
    init = 'random',
    random_state = 357,
    n_jobs = -1
)
reduced = reducer.fit_transform(embeddings)
vis = pd.DataFrame({'x': reduced[:,0], 'y': reduced[:,1], 'label': manifest['name']})
```

Now we define a function to make our plot. We'll add some people to look for
along as well (in this case, a few baseball players)

```{code-cell}
def sim_plot(data, hue = None, labels = None, n_colors = 3):
    """Create a scatterplot and optionally color/label its points."""
    fig, ax = plt.subplots(figsize = (10, 10))
    pal = sns.color_palette('colorblind', n_colors = n_colors) if hue else None
    g = sns.scatterplot(
        x = 'x', y = 'y',
        hue = hue, palette = pal, alpha = 0.8,
        data = data, ax = ax
    )
    g.set(xticks = [], yticks = [], xlabel = 'Dim. 1', ylabel = 'Dim. 2')

    if labels:
        to_label = data[data['label'].isin(labels)]
        to_label[['x', 'y', 'label']].apply(lambda x: g.text(*x), axis = 1)

    plt.show()

people = ('Jackie Robinson', 'Lou Gehrig', 'Cy Young')
sim_plot(vis, labels = people)
```

### Clustering

The document embeddings seem to be partitioned into different clusters. We'll
end by using a hierarchical clusterer to see if we can further specify these
clusters. This involves using the `AgglomerativeClustering` object, which we
fit to our embeddings. Hierarchical clustering requires a pre-defined number of
clusters. In this case, we use 18.

```{margin} Why this number of clusters?
We grid searched different numbers and measured the results with a [silhouette
coefficient][coefficient].

[coefficient]: https://towardsdatascience.com/silhouette-coefficient-validating-clustering-techniques-e976bb81d10c
```

```{code-cell}
agg = AgglomerativeClustering(n_clusters = 18)
agg.fit(embeddings)
```

Now we assign the clusterer's predicted labels to the visualization data
DataFrame and re-plot the results.

```{code-cell}
vis.loc[:, 'cluster'] = agg.labels_
sim_plot(vis, hue = 'cluster', n_colors = 18)
```

These clusters seem to be both detailed and nicely partitioned, bracketing off,
for example, classical musicians and composers (cluster 6) from jazz and
popular musicians (cluster 10).

```{code-cell}
:tags: ["output_scroll"]
for k in (6, 10):
    people = vis.loc[vis['cluster'] == k, 'label']
    print("Cluster:", k, "\n")
    for person in people:
        print(person)
    print("\n")
```

Consider further cluster 5, which seems to be about famous scientists.

```{margin} What's going on with "Martian Theory"?
It appears this is actually Percival Lowell, an astronomer (he did, however,
advance the idea that Mars is inhabited). Apparently our metadata is a little
messy!
```

```{code-cell}
for person in vis.loc[vis['cluster'] == 5, 'label']:
    print(person)
```

There are, however, some interestingly noisy clusters, like cluster 12. With
people like Queen Victoria and William McKinley in this cluster, it at first
appears to be about national leaders of various sorts, but the inclusion of
others like Al Capone (the gangster) and Ernie Pyle (a journalist) complicate
this. If you take a closer look, what really seems to be tying these obituaries
together is war. Nearly everyone here was involved in war in some fashion or
another – save for Capone, whose inclusion makes for strange bedfellows.

```{code-cell}
for person in vis.loc[vis['cluster'] == 12, 'label']:
    print(person)
```

Depending on your task, these detailed distinctions may not be so desirable.
But for us, the document embeddings provide a wonderfully nuanced view of the
kinds of people in the obituaries. From here, further exploration might involve
focusing on misfits and outliers. Why, for example, is Capone in cluster 12? Or
why is Lou Gehrig all by himself in his own cluster? Of course, we could always
re-cluster this data, which would redraw such groupings, but perhaps there is
something indeed significant about the way things are divided up as they stand.
Word embeddings help bring us to a point where we can begin to undertake such
investigations – what comes next depends on which questions we want to ask.
