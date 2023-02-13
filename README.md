# Labelstudio backend with taggers

Labelstudio backend with EstNLTK taggers, including NER tagger, which uses medbert model for tagging

##  Setup

### Environment

`conda env create -f labelstudio-taggers.yml`

`conda activate labelstudio-taggers`

`pip install -U -e .`

NB! The following line is optional (for the example models) and not recommended, as it takes up quite a bit of space

`pip install -r label_studio_ml/examples/requirements.txt`


### Start a backend (with regex taggers)

#### init

`label-studio-ml init bagoftaggers --script bag_of_taggers/bag_of_taggers_extractor.py:BagOfTaggersExtractor`


#### deployment

`label-studio-ml start ./bagoftaggers`
