# CWABSA
Github for: https://personal.eur.nl/frasincar/spool/icwe2020.pdf
Code for (non-)Contextualised Word embedding techniques on a hybrid approach for Aspect-Based Sentiment Analysis. 

All software is written in PYTHON3 (https://www.python.org/) and makes use of the TensorFlow framework (https://www.tensorflow.org/).

## Installation Instructions (Windows):
### Dowload required files and add them to data/externalData folder:
1. Download ontology: https://github.com/KSchouten/Heracles/tree/master/src/main/resources/externalData
2. Download SemEval2015 Datasets: http://alt.qcri.org/semeval2015/task12/index.php?id=data-and-tools
3. Download SemEval2016 Dataset: http://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools
4. Download Stanford CoreNLP parser: https://nlp.stanford.edu/software/stanford-parser-full-2018-02-27.zip
5. Download Stanford CoreNLP Language models: https://nlp.stanford.edu/software/stanford-english-corenlp-2018-02-27-models.jar
6. Download Glove Embeddings: http://nlp.stanford.edu/data/glove.42B.300d.zip
7. Download fastText Embeddings: https://fasttext.cc/docs/en/english-vectors.html
8. Download Word2vec-CBOW Embeddings https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
9. Download Word2vec-SGhttp://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_300d.txt.bz2

### Setup Environment
1. Install chocolatey (a package manager for Windows): https://chocolatey.org/install
2. Open a command prompt.
3. Install python3 by running the following command: `code(choco install python)` (http://docs.python-guide.org/en/latest/starting/install3/win/).
4. Make sure that pip is installed and use pip to install the following packages: setuptools and virtualenv (http://docs.python-guide.org/en/latest/dev/virtualenvs/#virtualenvironments-ref).
5. Create a virtual environemnt in a desired location by running the following command: `code(virtualenv ENV_NAME)`
6. Direct to the virtual environment source directory. 
7. Unzip the CWABSA_software.zip file in the virtual environment directrory. 
8. Activate the virtual environment by the following command: 'code(Scripts\activate.bat)`.
9. Install the required packages from the requirements.txt file by running the following command: `code(pip install -r requirements.txt)`.
10. Install the required space language pack by running the following command: `code(python -m spacy download en)`

### Run Software
1. Configure one of the three main files to the required configuration (main.py, main_cross.py, main_hyper.py)
2. Run the program from the command line by the following command: `code(python PROGRAM_TO_RUN.py)` (where PROGRAM_TO_RUN is main/main_cross/main_hyper) 
3. For contextualised word embedders, i.e., BERT, BERT-Large, and ELMo, the program can be run directly without loading data, the data is already in programGeneratedData (on github not the embedding matrix, as this is a too large file)
4. However, the data can also be prepared for BERT and ELMo with prepareBERT.py, prepareELMo.py;
5. One side-note for BERT_Large, an embedding matrix is added in data/programGeneratedData, this matrix has vector size 768. For vectorsize 1024, run getBERTusingColab.py on Google Colab and change line 280.
6. For non-contextualised word embedders, load data from externalData with loaddata.py

## Software explanation:
The environment contains the following main files that can be run: main.py, main_cross.py, main_hyper.py
- main.py: program to run single in-sample and out-of-sample valdition runs. Each method can be activated by setting its corresponding boolean to True e.g. to run the hybrid method, set useOntology = True, runLCRROTALT = True.
- main_cross.py: similar to main.py but runs a 10-fold cross validation procedure for each method.
- main_hyper.py: program that is able to do hyperparameter optimzation for a given space of hyperparamters for each method. To change a method change the objective and space parameters in the run_a_trial() function.

- config.py: contains parameter configurations that can be changed such as: dataset_year, batch_size, iterations.

- dataReader2016.py, loadData.py: files used to read in the raw data and transform them to the required formats to be used by one of the algorithms

- getBERTusingColab.py: Can be run in Google Colab using TPU. As input raw data and embedding vectors for each word are retrieved as matrix.
Batch-data can be used, when computation times are long, then submatrices are retrieved. These can be combined in prepareBERT.py
- prepareBERT.py: takes embedding vectors as (sub)matrices as input. A data set with unique words, i.e., context dependent words, and corresponding embedding matrix are made. 
- prepareELMo.py: prepares the train/test data for the ELMo embeddings. Returns embedding matrix and data with corresponding unique words

- lcrModelAlt.py: Tensorflow implementation for the LCR-Rot-hop algorithm
- OntologyReasoner.py: PYTHON implementation for the ontology reasoner

- att_layer.py, nn_layer.py, utils.py: programs that declare additional functions used by the machine learning algorithms.

## Directory explanation:
The following directories are necessary for the virtual environment setup: \__pycache, \Include, \Lib, \Scripts, \tcl, \venv
- data:
	- externalData: Location for the external data required by the methods
	- programGeneratedData: Location for preprocessed data that is generated by the programs
	    - cross_results_2015: Results for a k-fold cross validation process for the SemEval-2015 dataset
        - cross_results_2016: Results for a k-fold cross validation process for the SemEval-2015 dataset
	- temporaryData: Data that is temporary needed for preparing BERT and ELMo. 

## Related Work: ##
This code uses ideas and code of the following related papers:
- Wallaart, O., & Frasincar, F. (2019, June). A Hybrid Approach for Aspect-Based Sentiment Analysis Using a Lexicalized Domain Ontology and Attentional Neural Models. In European Semantic Web Conference (pp. 363-378). Springer, Cham.
- Zheng, S. and Xia, R. (2018). Left-center-right separated neural network for aspect-based sentiment analysis with rotatory attention. arXiv preprint arXiv:1802.00892.
- Schouten, K. and Frasincar, F. (2018). Ontology-driven sentiment analysis of product and service aspects. In Proceedings of the 15th Extended Semantic Web Conference (ESWC 2018). Springer. To appear
- Liu, Q., Zhang, H., Zeng, Y., Huang, Z., and Wu, Z. (2018). Content attention model for aspect based sentiment analysis. In Proceedings of the 27th International World Wide Web Conference (WWW 2018). ACM Press.
- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
- Peters, M. E., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K., & Zettlemoyer, L. (2018). Deep contextualized word representations. arXiv preprint arXiv:1802.05365.
