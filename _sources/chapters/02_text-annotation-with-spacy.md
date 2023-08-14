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

Text Annotation with spaCy
==========================

This chapter introduces the general field of natural language processing, or
NLP. While NLP is often used interchangeably with text mining/analytics in
introductory settings, the former differs in important ways from many of the
core methods in the latter. We will highlight a few such differences over the
course of this session, and then more generally throughout the workshop series
as a whole.

```{admonition} Learning objectives
By the end of this chapter, you will be able to:

+ Explain how document annotation differs from other representations of text
  data
+ Know how `spaCy` models and their respective pipelines work
+ Extract linguistic information about text using `spaCy`
+ Describe key terms in NLP, like part-of-speech tagging, dependency parsing,
  etc.
+ Know how/where to look for more information about the linguistic data `spaCy`
  makes available
```

NLP vs. Text Mining: In Brief
-----------------------------

### Data structures

One way to distinguish NLP from text mining has has to do with the various
**data structures** we use in the former. Generally speaking, NLP methods are
**maximally preservative** when it comes to representing textual information in
a way computers can read. Unlike text mining's atomizing focus on **bags of
words**, in NLP we often use literal transcriptions of the input text and run
our analyses directly on that. This is because much of the information NLP
methods provide is context-sensitive: we need to know, for example, the subject
of a sentence in order to do dependency parsing; part-of-speech taggers are
most effective when they have surrounding tokens to consider. Accordingly, our
workflow needs to retain as much information about our documents as possible,
for as long as possible. In fact, many NLP methods _build on each other_, so
data about our documents will grow over the course of processing them (rather
than getting pared down, as with text mining). The dominant paradigm, then, for
thinking about how text data is represented in NLP is **annotation**: NLP tends
to add, associate, or tag documents with extra information.

### Model-driven methods

The other key difference between text mining and NLP lies in the way the latter
tends to be more **model-driven**. NLP methods often rely on statistical models
to create the above information, and ultimately these models have a lot of
assumptions baked into them. Such assumptions range from philosophy of language
(how do we know we're analyzing meaning?) to the kind of training data on which
they're trained (what does the model represent, and what biases might thereby
be involved?). Of course, it's possible to build your own models, and indeed a
later chapter will show you how to do so, but you'll often find yourself using
other researchers' models when doing NLP work. It's thus very important to know
how researchers have built their models so you can do your own work
responsibly.

spaCy Language Models
---------------------

```{margin} Want to know more?
Explosion, the company behind `spaCy`, has a series of useful videos
introducing the framework. [This one][video] is a good place to start.

[video]: https://www.youtube.com/watch?v=9k_EfV7Cns0&t=1365s
```

### `spaCy` pipelines

Much of this workshop series will use language models from `spaCy`, a very
popular NLP library for Python. In essence, a `spaCy` model is a collection of
sub-models arranged into a **pipeline**. The idea here is that you send a
document through this pipeline, and the model does the work of annotating your
document. Once it has finished, you can access these annotations to perform
whatever analysis you'd like to do.

![The spaCy pipeline, which is broken up into separate components, or
pipes](../img/spacy_pipeline.png)

Every component, or **pipe**, in a `spaCy` pipeline performs a different task,
from tokenization to part-of-speech tagging and named-entity recognition. Each
model comes with a specific ordering of these tasks, but you can mix and match
them after the fact. 

### Downloading a model

The specific model we'll be using is `spaCy`'s medium-sized English model:
[en_core_web_md][encore]. It's been trained on the [OntoNotes][onto] corpus and
it features several useful pipes, which we'll discuss below.

[encore]: https://github.com/explosion/spacy-models/releases/tag/en_core_web_sm-3.3.0
[onto]: https://catalog.ldc.upenn.edu/LDC2013T19

If you haven't used `spaCy` before, you'll need to download this model. You can
do so by running the following in a command line interface:

```sh
python -m spacy download en_core_web_md
```

Preliminaries
-------------

Once your model has downloaded, it's time to set up an environment. Here are
the libraries you'll need for this chapter.

```{code-cell}
from pathlib import Path
from collections import Counter, defaultdict
from tabulate import tabulate
import spacy
```

And here's the data directory we'll be working from:

```{code-cell}
indir = Path("data/session_one")
```

Finally, we initialize the model.

```{code-cell}
nlp = spacy.load('en_core_web_md')
```

Annotations
-----------

To annotate a document with the model, simply pass it to the model. We'll use a
short poem by Gertrude Stein to show this.

```{code-cell}
with indir.joinpath("stein_carafe.txt").open('r') as fin:
    poem = fin.read()

carafe = nlp(poem)
```

With this done, we inspect the result...

```{code-cell}
carafe
```

...which seems to be no different from a string representation! This output is
misleading, however. Our annotated poem now has a tone of extra information
associated it, which is accessible via attributes and methods.

```{code-cell}
attributes = [i for i in dir(carafe) if i.startswith("_") == False]
print("Number of attributes in a spaCy doc:", len(attributes))
```

This high number of attributes indicates an important point to keep in mind
when working with `spaCy` and NLP generally: as we mentioned before, the
primary data model for NLP aims to **maximally preserve information** about
your document. It keeps documents intact and in fact adds much more information
about them than Python's base string methods have. In this sense, we might say
that `spaCy` is additive in nature, whereas text mining methods are
subtractive, or reductive.

### Document annotations

`spaCy` annotations apply to either documents or individual tokens. Here are
some document-level annotations:

```{code-cell}
annotations = {'Sentences': 'SENT_START', 'Dependencies': 'DEP', 'TAGS': 'TAG'}
for annotation, tag in annotations.items():
    print(f"{annotation:<12} {carafe.has_annotation(tag)}")
```

Let's look at sentences, which we access with `.sents`.

```{code-cell}
carafe.sents
```

...with a slight hitch: this returns a generator, not a list. `spaCy` aims to
be memory efficient (especially important for big corpora), so many of its
annotations are stored this way. We'll need to iterate through this generator
to see its contents.

```{code-cell}
for sent in carafe.sents:
    print(sent.text)
```

One very useful attribute is `.noun_chunks`. It returns nouns and compound
nouns in a document.

```{code-cell}
for chunk in carafe.noun_chunks:
    print(chunk)
```

See how this picks up not only nouns, but articles and compound information?
Articles could be helpful if you wanted to track singular/plural relationships,
while compound nouns might tell you something about the way a document refers
to the entities therein. The latter could have repeating patterns, and you
might imagine how you could use noun chunks to create and count n-gram tokens
and feed that into a classifier.

Consider this example from _The Odyssey_. Homer used many epithets and
repeating phrases throughout his epic. According to some theories, these act as
mnemonic devices, helping a performer keep everything in their head during an
oral performance (the poem wasn't written down in Homer's day). Using
`.noun_chunks` in conjunction with a Python `Counter`, we may be able to
identify these in Homer's text. Below, we'll do so with _The Odyssey_ Book XI.

First, let's load and model the text.

```{code-cell}
with indir.joinpath("odyssey_book_11.txt").open('r') as fin:
    book11 = fin.read()

odyssey = nlp(book11)
```

Now we pass our noun chunks to the `Counter`. Be sure to grab only the `.text`
attribute from each token; we don't need the other attributes.

```{code-cell}
counts = Counter([chunk.text for chunk in odyssey.noun_chunks])
```

With that done, let's look for repeating noun chunks with three or more words.

```{code-cell}
repeats = []
for chunk, count in counts.items():
    length = len(chunk.split())
    if length > 2 and count > 1:
        repeats.append([chunk, length])

print(tabulate(repeats, ['Chunk', 'Length']))
```

Another way to look at entities of this sort is with `.ents`. `spaCy` uses
**named-entity recognition** (NER) to extract significant objects, or entities,
in a document. In general, anything that has a proper name associated with it
is likely to be an entity, but things like expressions of time and geographic
location are also often tagged.

```{code-cell}
for i in range(5):
    print(odyssey.ents[i])
```

Entities come with labels that differentiate what kind of entity they are.
Using the `.label_` attribute, we extract temporal entities in Book XI.

```{code-cell}
"; ".join(e.text for e in odyssey.ents if e.label_ == 'TIME')
```

And here is a unique list of all the people

```{margin} How many labels are there?
This will depend on the model. Here's the [label scheme][scheme] for the one
we're using.

[scheme]: https://spacy.io/models/en#en_core_web_md-labels
```

```{code-cell}
:tags: ["output_scroll"]
set(e.text for e in odyssey.ents if e.label_ == 'PERSON')
```

Don't see an entity that you know to be in your document? You can add more to
the `spaCy` model. Doing so is beyond the scope of our workshop session, but
the library's `EntityRuler` [documentation][doc] will show you how.

[doc]: https://spacy.io/api/entityruler

### Token annotations

In addition to storing all of this information about documents, `spaCy` creates
a substantial amount of annotations for every token in a document. The same
logic as above applies to accessing this information.

Let's return to the Stein poem. Indexing it will return individual tokens.

```{code-cell}
carafe[3]
```

A token's attributes and methods range from simple booleans, like whether a
token is an alphabetic character:

```{code-cell}
carafe[3].is_alpha
```

...or whether it is a stop word:

```{code-cell}
carafe[3].is_stop
```

...to more complex pieces of information, like tracing back to the sentence in
which this token appears:

```{code-cell}
carafe[3].sent
```

...or the token's vector representation (more on this in the third session):

```{code-cell}
:tags: ["output_scroll"]
carafe[3].vector
```

Here's a listing of some attributes that are relevant for text mining:

```{code-cell}
:tags: ["output_scroll"]
attributes = [[tok.text, tok.is_punct, tok.like_url] for tok in carafe]
print(tabulate(attributes, ['Text', 'Is punctuation', 'Is a URL']))
```

We'll discuss some of the more complex annotations later on, both in this
session and others. For now, let's collect some simple information about each
of the tokens in our document. To do so, we use a list comprehension on the
`.text` attribute of each token.

```{code-cell}
words = ' '.join(tok.text for tok in carafe if tok.is_alpha)
punct = ' '.join(tok.text for tok in carafe if tok.is_punct)

print("Words:", words)
print("Punctuation:", punct)
```

Want some linguistic information? We can get that too. For example, here are
lemmas:

```{margin} You might be wondering about those underscores...

The syntax conventions of `spaCy` use a trailing underscore to access the
actual attribute information for a token. Using an attribute without the
underscore will return an id, which the library uses internally to piece
together output.
```

```{code-cell}
:tags: ["output_scroll"]
lemmas = [[tok.text, tok.lemma_] for tok in carafe]
print(tabulate(lemmas, ['Token', 'Lemma']))
```

With such attributes at your disposal, you might imagine how you could work
`spaCy` into a text mining pipeline. Instead of using separate functions to
clean your corpus, those steps could all be accomplished by accessing
attributes.

But before you do this, you should consider 1) whether the increased
computational/memory overhead is worthwhile for your project; and 2) whether
`spaCy`'s base models will work for the kind of text you're using. This second
point is especially important. While `spaCy`'s base models are incredibly
powerful, they are built for general purpose applications and may struggle with
domain-specific language. Medical text and early modern print are two such
examples of where the base models interpret your documents in unexpected ways,
thereby complicating, maybe even ruining, parts of a text mining pipeline that
relies on them. 

That all said, there are ways to train your own `spaCy` model on a specific
domain. This can be an extensive process, one which exceeds the limits of our
short workshop, but if you want to learn more about doing so, you can visit
[this page][training]. There are also [third party models][models] available,
which you might find useful, though your mileage may vary.

[training]: https://spacy.io/usage/training
[models]: https://spacy.io/universe/category/models

Part-of-Speech Tagging
----------------------

One of the most common tasks in NLP involves assigning **part-of-speech, or
POS, tags** to each token in a document. As we saw in the text mining series,
these tags are a necessary step for certain text cleaning process, like
lemmatization; you might also use them to identify subsets of your data, which
you could separate out and model. Beyond text cleaning, POS tags can be useful
for tasks like **word sense disambiguation**, where you try to determine which
particular facet of meaning a given token represents.

Regardless of the task, the process of getting POS tags from `spaCy` will be
the same. Each token in a document has an associated tag, which is accessible
as an attribute.

```{code-cell}
:tags: ["output_scroll"]
pos_tags = [[tok.text, tok.pos_] for tok in carafe]
print(tabulate(pos_tags, ['Token', 'Tag']))
```

If you don't know what a tag means, use `spacy.explain()`.

```{code-cell}
spacy.explain('CCONJ')
```

`spaCy` actually has two types of POS tags. The ones accessible with the
`.pos_` attribute are the basic tags, whereas those under `.tag_` are more
detailed (these come from the [Penn Treebank project][treebank]).

[treebank]: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

```{code-cell}
:tags: ["output_scroll"]
treebank = [[tok.text, tok.tag_, spacy.explain(tok.tag_)] for tok in carafe]
print(tabulate(treebank, ['Token', 'Tag', 'Explanation']))
```

Dependency Parsing
------------------

Another tool that can help with tasks like disambiguating word sense is
dependency parsing. Dependency parsing involves analyzing the grammatical
structure of text (usually sentences) to identify relationships between the
words therein. The basic idea is that every word in a sentence is linked to at
least one other word via a tree structure, and these linkages are hierarchical. 
Dependency parsing can tell you information about:

1. The primary **subject** of a sentence (and whether it is an **active** or
   **passive** subject)
2. Various **heads**, which determine the syntactic categories of a phrase;
   these are often nouns and verbs
3. Various **dependents**, which modify, either directly or indirectly, their
   heads (think adjectives, adverbs, etc.)
4. The **root** of the sentence, which is often ([but not always!][caveat]) the
   primary verb

Linguists have developed a number of different methods to parse dependencies,
which we won't discuss here. Take note though that most popular one in NLP is
the [Universal Dependencies][ud] framework; `spaCy`, like most NLP models, uses
this. The library also has some functionality for visualizing dependencies,
which will help clarify what it is they are in the first place. Below, we
visualize a sentence from the Stein poem.

[caveat]: https://universaldependencies.org/u/dep/root.html
[ud]: https://universaldependencies.org

```{code-cell}
to_render = list(carafe.sents)[2]
spacy.displacy.render(to_render, style = 'dep')
```

See how the arcs have arrows? Arrows point to the dependents within a phrase
or sentence, that is, they point to modifying relationships between words.
Arrows arc out from a head, and the relationships they indicate are
all specified with labels. As with the POS tags, you can use `spacy.explain()`
on the dependency labels, which we'll do below. The whole list of them is also
available in this [table of typologies][typologies]. Finally, somewhere in the
tree you'll find a word with no arrows pointing to it (here, "spreading"). This
is the root. One of its dependents is the subject of the sentence (here,
"difference").

[typologies]: https://universaldependencies.org/u/dep/all.html

Seeing these relationships are quite useful in and of themselves, but the real
power of dependency parsing comes in all the extra data it can provide about a
token. Using this technique, you can link tokens back to their heads, or find
local groupings of tokens that all refer to the same head.

With this sentence, for example:

```{code-cell}
sentence = odyssey[2246:2260]
sentence.text
```

We can construct a `for` loop to roll through each token and retrieve
dependency info.

```{code-cell}
dependencies = []
for tok in sentence:
    info = [tok.text, tok.dep_, spacy.explain(tok.dep_), tok.head.text]
    dependencies.append(info)

print(tabulate(dependencies, ['Text', 'Dependency', 'Explanation', 'Head']))
```

How many tokens are associated with each head?

```{code-cell}
heads = Counter(head for (tok, dep, exp, head) in dependencies)
print(tabulate(heads.items(), ['Head', 'Count']))
```

We can also find which tokens are associated with each head. `spaCy` has a
special `.subtree` attribute for each token, which produces this grouping. As
you might expect by now, `.subtree` returns a generator, so convert it to a
list or use list comprehension to extract the tokens. We'll do this in a
separate function. Within this function, we use a token's `.text_with_ws`
attribute to return an exact, string-like representation of the string.

```{margin} A word on the root
The subtree of a sentence's root is simply the sentence, so retrieving all the
tokens in the root's `.subtree` will just repeat the whole string.
```

```{code-cell}
def subtree_to_text(subtree):
    """Convert a subtree to its text representation."""
    subtree = ''.join(tok.text_with_ws for tok in subtree)

    return subtree.strip()

subtrees = []
for tok in sentence:
    subtree = subtree_to_text(tok.subtree)
    subtrees.append([tok.text, tok.dep_, subtree])

print(tabulate(subtrees, ['Token', 'Dependency', 'Subtree']))
```

Putting Everything Together
---------------------------------

Now that we've walked through all these options, let's put them into action.
Below, we construct two short examples of how you might combine different
aspects of token attributes to analyze a text. Both are essentially
**information retrieval** tasks, and you might imagine doing something similar
to extract and analyze particular words in your corpus, or to find different
grammatical patterns that could be of significance.

### Finding lemmas

In the first, we use the `.lemma_` attribute to search through Book XI and
match its tokens to a few key words. If you've read _The Odyssey_, you'll know
that Book XI is where Odysseus and his fellow sailors have to travel down to
the underworld Hades, where they speak with the dead. We already saw one
example of this: Odysseus attempts to embrace his dead mother after communing
with her. The whole trip to Hades is an emotionally tumultuous experience for
the travelers, and peppered throughout Book XI are expressions of grief.

With `.lemma_`, we can search for these expressions. We'll roll through the
text and determine whether a token lemma matches one of a selected set. When we
find a match, we get the subtree of this token's _head_. That is, we find the
head upon which this token depends, and then we use that to reconstruct the
local context for the token.

```{code-cell}
target = ('cry', 'grief', 'grieve', 'sad', 'sorrow', 'tear', 'weep')
retrieved = []
for tok in odyssey:
    if tok.lemma_ in target:
        subtree = subtree_to_text(tok.head.subtree)
        retrieved.append([tok.text, subtree])

print(tabulate(retrieved, ['Token', 'Subtree']))
```

### Verb-subject relations

For our second example, we use dependency tags to find the subject of sentences
in Book XI. As before, we iterate through each token in the document, this time
checking to see whether it has the `nsubj` or `nsubjpass` tag for its `.dep_`
attribute. We also check whether a token is a noun (otherwise we'd get many
articles like "who," "them," etc.). If a token matches these two conditions,
we find its head verb as well as the token's subtree. Note that this time, the
subtree will refer directly to the token in question, not to the head. This
will let us capture some descriptive information about each sentence subject.

```{code-cell}
:tags: ["output_scroll"]
subj = []
for tok in odyssey:
    if tok.dep_ in ('nsubj', 'nsubjpass') and tok.pos_ in ('NOUN', 'PROPN'):
        subtree = subtree_to_text(tok.subtree)
        subj.append([tok.text, tok.head.text, tok.head.lemma_, subtree])

print(tabulate(subj, ['Subject', 'Head', 'Head lemma', 'Subtree']))
```

How many times do each of our subjects appear?

```{code-cell}
:tags: ["output_scroll"]
subjects = Counter(subject for (subject, head, lemma, subtree) in subj)
print(tabulate(subjects.items(), ['Subject', 'Count']))
```

Which heads are associated with what subject?

```{code-cell}
:tags: ["output_scroll"]
subject_heads = defaultdict(list)
for item in subj:
    subject, head, *_ = item
    subject_heads[subject].append(head)

associations = [
    [subject, ", ".join(heads)] for subject, heads in subject_heads.items()
]
print(tabulate(associations, headers=["Subject", "Associated heads"]))
```

Such information provides another way of looking at something like topicality.
Rather than using, say, a bag of words approach to build a topic model, you
could instead segment your text into chunks like the above and start tallying
up token distributions. Such distributions might help you identify the primary
subject in a passage of text, whether that be a character or something like a
concept. Or, you could leverage them to investigate how different subjects are
talked about, say by throwing POS tags into the mix to further nuance
relationships across entities.

Our next session will demonstrate what such investigations look like in action.
For now however, the main takeaway is that the above annotation structures
provide you with a host of different ways to segment and facet your text data.
You are by no means limited to single token counts when working computationally
analyzing text. Indeed, sometimes the most compelling ways to explore a corpus
lie in the broader, fuzzier relationships that NLP annotations help us
identify.
