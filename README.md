# Master's Project, 10 ECTS

**Author: Siiri Käpyaho**

This file bundle includes the code written for the thesis *Anthropomorphization in academia: Detecting human-like descriptions of AI via prototype similarity* where the following research questions were posed:
* RQ1: Can existing anthropomorphization detection methods be improved by using open-source models, and without fine-tuning a language model or needing to use an external lexicon?
* RQ 2: Are there differences in the degree of anthropomorphization between different research domains?
* RQ 3: If a difference is observed, what are the differences between different domains like?
* RQ 4: Does the level of anthropomorphization in abstracts reflect the overall level of anthropomorphization found in the articles?

Most files function as standalone programs that have been utilized in different steps of the thesis process; in other words, there is no single command with which one could reproduce all data and analyses. However, one can follow the steps taken in the process and modify the hard-coded file paths to reproduce the results. This README briefly explains the purpose of each file. Note also that the bundle excludes code that was written for initial manual inspection of data and various (unstructured) tests.

## Contents

* Data downloads
* Preprocessing
* Experiments
* Method Data
* New method for detecting anthropomorphization via prototype similarity
* Evaluation
* Post-processing
* Analysis
* Utils
* Requirements


### Data downloads

Code for downloading the data from OpenAlex.

* `download_openalex.sh`: A script that uses the official OpenAlex CLI to download the data.
* `download_xmls.py`: As the above mentioned CLI had a bug in XML handling at the time of collecting the data ([since fixed](https://github.com/ourresearch/openalex-official/pull/1)), a custom program was written to fetch the XMLs and record status of the attemptempted files. It was run daily to download up to 100 files per day.


### Preprocessing

Code for preprocessing the data to be ready for the anthropomorphization detection method and analysis. Includes building the abstracts from inverted indices, fixing encoding errors, deduplicating, and collecting and masking sentences including target entities.

* `preprocess.py`: A program for turning the abstracts from OpenAlex from inverted indices to cleaned string representations. The pipeline includes fixing encoding errors, normalizing ambiguous characters, removing HTML tags and LaTeX commands, and removing redundant line breaks. After, empty texts and non-English texts are removed, and the most essential metadata are kept, creating a "light" version of the downloaded data.
* `dedupe.py`: A program that further removes abstracts that do not include target entities, are under 250 characters long, or contain less than 40% ASCII characters (and thus are not likely to be narrative English), as well as removes duplicated abstracts using count-based heuristics (determined after initial manual inspection).
* `inspect_xmls.py`: This Notebook inspects the success rate of the attempted XML downloads, and converts the XML files into CSV format, ready for the masking step.
* `csv_to_doc.py`: A program that converts strings into spaCy Doc objects and saves them to disk. (Current version modified to work on full-texts instead of the abstracts.)
* `mask_sentences.py`: A program that produces the masked sentences ready to be fed into the detection method, as well as records additional linguistic metadata for the masked tokens and the noun chunks they belong to. (Current version modified to work on full-texts instead of the abstracts.)


### Experiments

Includes the pre-experiment with embedding models, and the main experiment with the 1080 method variants.

* `pre-experiment-embedding-models.ipynb`: A Jupyter Notebook containing the pre-experiment conducted with the embedding models. In the experiment, 4 embedding models were evaluated in how accurately they classified 79 manually created examples of human and non-human entities given changes in input and the pooling strategy.
* `main-experiment.ipynb`: Includes the latest version of the main experiment with 1080 methods. Shows both the corrected and faulty versions of the extraction of contextual embeddings.


### Evaluation

Includes evaluation results for the 1080 experiments, and a comparison to AnthroScore (Cheng et al., 2024) and Atypical Animacy (Coll Ardanuy et al., 2020) on AnthroSet (Lonke et al., 2024), and to AnthroScore on Anthropomorphism Corpus (Shardlow et al., 2025) (answering research question 1).

* `detector_experiment_results.ipynb`: This Notebook contains the analysis of the results from the 1080 experiments on AnthroSet. (Some of the experiments needed to be re-run after detecting some problems in the experiment implementation, and this Notebook reorganizes the scattered results). It also verifies the AnthroScore results reported in Lonke et al. (2025), and recalculates performance for binary labels.
* `anthroset_results_stats.ipynb`: This Notebook essentially continues the previous one, and shows the statistical tests done on experiment results that are reported in the thesis. It also compares the final results to AnthroScore and Atypical Animacy.
* `anthropomorphism_corpus_new_method_vs_anthroscore.ipynb`: This Notebook evaluates the performance of the New method, AnthroScore, and AnthroScore with `Ġ`-prefixed tokens on Anthropomorphism Corpus.


### Method Data

Static data used for the method: target entities and prototype vector descriptions.

* `*.txt`: Entity lists, including artifact, human, and LM (incl. product name) terms.
* `prototype_descriptions.py`: Natural language descriptions used for creating prototype vectors in the final method.


### New method for detecting anthropomorphization

The final method that was used to describe the collected data. Also includes a more robust version of AnthroScore.

* `calculate_scores.py`: **The final method**. Loads the needed models from HuggingFace and calculates scores for sentences including a `[MASK]` token. As an addition from the experiments, the final method includes a mask centering function that ensures that the masked token is not cropped out from texts longer than the allowed context window.
* `get_all_anthroscores_modified.py`: A modified version of AnthroScore that saves the progress incrementally and recovers from specific errors. This version was used to calculate AnthroScores for the collected data, but the scores were not utilized in the final work after more problems surfaced from the original method. **See the file header for the original license.**


### Post-processing

Code for cleaning up the data after running the method.

* `clean-up-abstracts.ipynb`: This Notebook contains the cleanup steps taken to get rid of entities that are not likely to refer to the indented entities.
* `clean-up-fulltexts.ipynb`: This Notebook is based on the above, with minimal changes, in order to keep the results on full-texts and abstracts comparable.


### Analysis

Analyses for abstract-only (answering research questions 2 and 3) and full-text corpora (answering research question 4).

* `abstract_analysis.ipynb`: This Notebook contains the first part of the exploratory data analysis on the abstract-only data. Many of the results are not reported in the final thesis due to space constraints.
* `entity_inspection.ipynb`: This Notebook essentially continues the first analysis (as the first file started to feel crowded). Similarly, not all findings made here could be fit into the thesis.
* `fulltext_analysis.ipynb`: This Notebook contains the analysis for the full-text data.


### Utils

Helper programs for writing data from one format to another.

* `json_to_jsonl.py`: A program meant to write the individual OpenAlex metadata JSON files into a single JSONL file.
* `jsonl_to_csv.py`: A program for converting a JSONL file into a CSV (initially meant for converting the data into a format required by AnthroScore).


### Requirements

* `requirements.txt`: Includes the names of the external python packages utilized in the thesis. However, the code has been executed in different environments (locally, in Google Colab, on CSC supercomputers Puhti (decommissioned) and Roihu), and the exact versions for the packages used at the time of running the different files are not easily available.


## References

* [OpenAlex CLI](https://github.com/ourresearch/openalex-official/)
* [AnthroScore (Cheng et al., 2024)](https://aclanthology.org/2024.eacl-long.49.pdf) ([GitHub](https://github.com/myracheng/anthroscore))
* [Atypical Animacy (Coll Ardanuy et al., 2020)](https://aclanthology.org/2020.coling-main.400.pdf) ([GitHub](https://github.com/Living-with-machines/AtypicalAnimacy))
* [AnthroSet (Lonke et al., 2025)](https://aclanthology.org/2025.ommm-1.3.pdf) ([GitHub](https://github.com/doriellel/anthroset))
* [Anthropomorphism Corpus (Shardlow et al., 2025)](https://aclanthology.org/2025.findings-acl.926.pdf) ([GitHub](https://github.com/mattshardlow/Anthropomorphism_Corpus))

## License

See [LICENSE](LICENSE.txt). Note also that the file `get_all_anthroscores_modified.py` has its own license notice, available in the file header.