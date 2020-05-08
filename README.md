# tmpFever
FEVER

All the files were taken from the original baseline code, which can be found [here](https://github.com/sheffieldnlp/fever-naacl-2018)

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
