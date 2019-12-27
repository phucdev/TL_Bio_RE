# TL_Bio_RE
Transfer Learning for Biomedical Relation Extraction Seminar. Applying BioBERT &amp; SciBERT to Relation Extraction (protein-protein-interaction).

Get started:
```python
from tlbiore.data import corpus_processor, utils

aimedPath = '/Users/phuc/develop/python/TL_Bio_RE/data/raw/AIMed-train.xml'
bioinferPath = '/Users/phuc/develop/python/TL_Bio_RE/data/raw/BioInfer-train.xml'

ppi_corpus = corpus_processor.process_corpora([aimedPath, bioinferPath])
# do splits before the transformations
train, dev, test = utils.train_dev_test_split(ppi_corpus.get_examples())

lee_train = corpus_processor.prepare_data_lee(train)
lee_dev = corpus_processor.prepare_data_lee(dev)
lee_test = corpus_processor.prepare_data_lee(test)

utils.export_tsv(lee_train, '/Users/phuc/develop/python/TL_Bio_RE/data/lee/lee-train.tsv')
utils.export_tsv(lee_dev, '/Users/phuc/develop/python/TL_Bio_RE/data/lee/lee-dev.tsv')
utils.export_tsv(lee_test, '/Users/phuc/develop/python/TL_Bio_RE/data/lee/lee-test.tsv')
```