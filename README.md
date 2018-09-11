## Wiki2Model
These scripts allow you to generate language models based on the CBoW model of Word2Vec, trained through text documents extracted directly from Wikipedia in multiple languages. In addition to the possibility of generating templates trained by the original content of Wikipedia, the scripts also allows to generate models trained by the semantically enriched content of Wikipedia. This textual enrichment can be based on the application of named entity recognition (NER) and word sense disambiguation (WSD) procedures.
> Generating a CBoW based model with Wikipedia original documents as external knowledge source:
```
python3 Wiki2Model.py --language EN --download in/db/ --extractor tools/WikiExtractor.py --output in/models/
```
> Generating a CBoW based model with Wikipedia semantically enriched documents as external knowledge source:
```
python3 Wiki2Model_S-Enrich.py --language EN --download in/db/ --extractor tools/WikiExtractor.py --s_enrich tools/S-Enrich_Bfy.jar --output in/models/
```


### Related scripts
* [Wiki2Model.py](https://github.com/joao8tunes/Wiki2Model/blob/master/Wiki2Model.py)
* [Wiki2Model_S-Enrich.py](https://github.com/joao8tunes/Wiki2Model/blob/master/Wiki2Model_S-Enrich.py)
* [WikiExtractor.py](https://github.com/attardi/wikiextractor/blob/master/WikiExtractor.py)
* [S-Enrich_Bfy.jar](https://github.com/joao8tunes/S-Enrich/blob/master/S-Enrich_Bfy/executable/S-Enrich_Bfy.jar)


### Requirements installation (Linux)
> Python 3 + PIP installation as super user:
```
apt-get install python3 python3-pip
```
> Gensim installation as normal user:
```
pip3 install --upgrade gensim
```
> NLTK + Scipy + Numpy installation as normal user:
```
pip3 install -U nltk scipy numpy
```


### See more
Project page on LABIC website: http://sites.labic.icmc.usp.br/Ms-Thesis_Antunes_2018