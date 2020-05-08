# tmpFever
## FEVER Challenge

The FEVER challenge is a natural language processing task where claims are automatically verified against facts in Wikipedia articles. Refer to the following paper for more information: https://arxiv.org/pdf/1803.05355.pdf

Some files were taken from the original baseline code, which can be found [here](https://github.com/sheffieldnlp/fever-naacl-2018)

-Data setup
Download the dataset by running the script `download-data.sh`. Create the wikipedia db by either downloading the pages
and processing them to create a database, or download the preprocessed .db file by running `download-preprocessed.sh`

The ESIM model weights can be found [here](https://drive.google.com/drive/folders/1LLaNqWyTsskAIj_aw5UeRe5SE-vdK9rV?usp=sharing)
The NER model and MongoDB dump can be found [here](https://drive.google.com/drive/folders/1ByZIHFB5816RHSroIaRkvuullPBppBxQ?usp=sharing)

### Wikipedia
Download pre-processed Wiki Dump
```bash
bash scripts/download-processed-wiki.sh
```
### Dataset
```bash
bash scripts/download-data.sh
```

-Required modules
The following modules will be required to run some of the backend components that support the database and some of the
components involved in the document and sentence retrieval stages. In order to run, these components require
the following python modules
* fever-drqa
* tqdm
* orator
* PyMySQL

-Scoring
The code for the scoring program can be found at this repo
* scorer: https://github.com/sheffieldnlp/fever-scorer.git


The following breaks down each of the main files and briefly explains what each directory holds:

1. **pipeline**: contains the scripts that integrate the different parts of the system togther 
2. **doc_ret**: contains utility scripts and preocessing files and models related to document retrieval
3. **rte**: contains the models, training loops, utility functions and checkpoints for the RTE module
4. **scripts**: contains general setup scripts
5. **training**: contains some scripts that perform some preprocessing on the dataset in preparation for training
6. **common**: contains some baseline code scripts and database classes that are essential for running the FEVER baseline model
7. **fever-frontend**: contains a basic one-page django web application that exposes the functionality of the system through the pipeline scripts. To run the frontend, run the pipeline/start_modules.sh script which initializes the system, then start the application by runnin manage.py runserver
2. **retrieval**: contains the models for both the sentence and document retrieval stages

