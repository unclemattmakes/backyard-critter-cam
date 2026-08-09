# Regional species starter lists

The species labels are **zero-shot** (BioCLIP 2 matches each crop against a plain list of
common names — see `classify.py`), so adapting the rig to your region is a one-line config
edit, not a retraining job. Copy a list below into your `config_local.py`:

```python
cfg.species_labels = [
    # ... one of the lists below, edited to your yard ...
]
```

then re-label existing crops with `python classify.py --redo` (new crops just use it).

## How to phrase a label (learned the hard way on the PNW yard)

- **Full common names beat bare nouns.** "domestic dog" and "domestic cat" resolve cleanly;
  a bare "dog" once mis-ranked against wild canids. Prefer "brown rat" to "rat".
- **Keep the list tight.** ~20–30 species you actually get (plus a few plausibles). Every label
  is a candidate for *every* crop, so a bloated list dilutes the scores.
- **Frequency-rank it.** Order doesn't change the math, but it keeps the list honest — put your
  top feeder species first so pruning is easy.
- **No decoy labels.** "plate of food" / "empty ground" were tried and are **inert**: BioCLIP is
  organism-only and will never rank a non-organism prompt above a real species. Non-animal
  false-fires are the general-CLIP gate's job (`clipfilter.py`), not the species list's.
- **Regional common names are fine** — BioCLIP 2 was trained on the tree of life with common and
  scientific names; "Eurasian magpie" and "magpie" both work, the longer form is safer.

Every list below keeps the same shape as the built-in PNW starter: mammals first, then birds,
frequency-ranked from regional feeder/garden-count data where it exists.

## US Northeast (suburban/backyard)

```python
cfg.species_labels = [
    # mammals
    "eastern gray squirrel", "eastern chipmunk", "raccoon", "Virginia opossum",
    "white-tailed deer", "eastern cottontail", "groundhog", "striped skunk", "red fox",
    "brown rat", "domestic cat", "domestic dog",
    # birds
    "northern cardinal", "dark-eyed junco", "black-capped chickadee", "mourning dove",
    "blue jay", "American goldfinch", "house finch", "tufted titmouse",
    "white-breasted nuthatch", "downy woodpecker", "American robin", "house sparrow",
    "European starling", "song sparrow", "American crow", "red-bellied woodpecker",
    "Carolina wren", "common grackle",
]
```

## US Southeast

```python
cfg.species_labels = [
    # mammals
    "eastern gray squirrel", "raccoon", "Virginia opossum", "nine-banded armadillo",
    "white-tailed deer", "eastern cottontail", "coyote", "striped skunk",
    "brown rat", "domestic cat", "domestic dog",
    # birds
    "northern cardinal", "Carolina chickadee", "Carolina wren", "tufted titmouse",
    "mourning dove", "blue jay", "red-bellied woodpecker", "eastern towhee",
    "American crow", "house finch", "brown thrasher", "northern mockingbird",
    "American robin", "European starling", "house sparrow", "common grackle",
    "downy woodpecker", "ruby-throated hummingbird",
]
```

## US Southwest (low desert / suburban)

```python
cfg.species_labels = [
    # mammals
    "desert cottontail", "black-tailed jackrabbit", "rock squirrel",
    "round-tailed ground squirrel", "coyote", "javelina", "bobcat", "gray fox",
    "striped skunk", "raccoon", "domestic cat", "domestic dog",
    # birds
    "mourning dove", "white-winged dove", "house finch", "house sparrow",
    "Gambel's quail", "curve-billed thrasher", "cactus wren", "verdin",
    "Anna's hummingbird", "gila woodpecker", "great-tailed grackle",
    "northern mockingbird", "European starling", "greater roadrunner",
]
```

## UK & Ireland (garden)

```python
cfg.species_labels = [
    # mammals
    "red fox", "European hedgehog", "eastern gray squirrel", "red squirrel",
    "European badger", "brown rat", "wood mouse", "domestic cat", "domestic dog",
    # birds (BTO garden-list order, roughly)
    "house sparrow", "blue tit", "great tit", "European robin", "blackbird",
    "wood pigeon", "collared dove", "European goldfinch", "chaffinch", "long-tailed tit",
    "coal tit", "dunnock", "common starling", "Eurasian magpie", "carrion crow",
    "Eurasian jackdaw", "European greenfinch", "great spotted woodpecker", "Eurasian wren",
]
```

## Central Europe (garden)

```python
cfg.species_labels = [
    # mammals
    "red fox", "European hedgehog", "red squirrel", "beech marten", "European badger",
    "roe deer", "brown rat", "wood mouse", "domestic cat", "domestic dog",
    # birds
    "house sparrow", "great tit", "blue tit", "blackbird", "European robin",
    "chaffinch", "European greenfinch", "European goldfinch", "common starling",
    "Eurasian magpie", "carrion crow", "hooded crow", "Eurasian jay",
    "great spotted woodpecker", "collared dove", "wood pigeon", "black redstart",
]
```

## Australia — east coast (suburban)

```python
cfg.species_labels = [
    # mammals
    "common brushtail possum", "common ringtail possum", "eastern grey kangaroo",
    "swamp wallaby", "short-beaked echidna", "grey-headed flying fox", "brown rat",
    "European rabbit", "red fox", "domestic cat", "domestic dog",
    # birds
    "rainbow lorikeet", "noisy miner", "Australian magpie", "sulphur-crested cockatoo",
    "laughing kookaburra", "pied currawong", "crimson rosella", "galah",
    "Australian white ibis", "little corella", "magpie-lark", "willie wagtail",
    "common myna", "brush turkey", "tawny frogmouth", "satin bowerbird",
]
```

---

Whatever the region: after a week of real crops, open the Specimen Catalogue's **Needs Review**
queue and confirm/correct a couple dozen labels. Your verdicts are what the eval harness grades
the classifier against (`eval.py --species`) — the list above is a starting guess; your yard's
data is the answer sheet.
