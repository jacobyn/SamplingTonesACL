# Characterizing Similarities and Divergences in Conversational Tones in Humans and LLMs by Sampling with People

## Abstract
Conversational tones -- the manners and attitudes in which speakers communicate -- are essential to effective communication. As Large Language Models (LLMs) become increasingly popular, it is necessary to characterize the divergences in their conversational tones relative to humans. Prior research relied on pre-existing taxonomies or text corpora, which suffer from experimenter bias and may not be representative of real-world distributions. Inspired by methods from cognitive science, we propose an iterative method for simultaneously eliciting conversational tones and sentences, where participants alternate between two tasks: (1) identify the tone of a given sentence and (2) a different participant generate a sentence based on that tone. We run 50 iterations of this process with both human participants and GPT-4 and obtain a dataset of sentences and frequent conversational tones. In an additional experiment, humans and GPT-4 annotated all sentences with all tones. With data from 1,339 participants, 33,370 human judgments, and 29,900 GPT-4 queries we show how our approach can be used to create an interpretable geometric representation of relations between tones in humans and GPT-4.  This work showcases how combining ideas from machine learning and cognitive science can address challenges in human-computer interactions.

## Code
In this section, we outline the code repository of this submission and introduce each file's function. Please also note that we did not provide a separate data submission, because the data elicited from experiment and required of for analyses are all situated in the directory already, and that it is less confusing if the relative address of data with respect to code is predefined.

### `/analysis_notebooks`
This section involves code that was used for the analysis of our Sampling with People and Quality-of-Fit rating data. For all analysis notebooks, we have yet to detailedly annotate every notebook cell with its instructions, but plan to provide a documented version of this file if asked for revision or in a coming non-anonymized preprint release.

#### `SRE_analysis.ipynb`
The analysis iPython notebook that details Sampling with People-specific statistical analyses and plotting.

#### `SRE_corpus_analysis.ipynb`
The analysis iPython notebook that details statistical analyses and plotting using both Sampling with People's and Quality-of-Fit Rating's data.

#### `alignment.py`, `embedding_sebas.py`
This python file involves a modified implementation of Ruder et al.'s
```
@inproceedings{ruder2018,
  author    = {Ruder, Sebastian  and  Cotterell, Ryan  and  Kementchedjhieva, Yova and S{\o}gaard, Anders},
  title     = {A Discriminative Latent-Variable Model for Bilingual Lexicon Induction},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  year      = {2018}
}
```
implementation for bilingual lexicon induction with latent variable model. We have also cited all required papers described in this project's [GitHub repository](https://github.com/sebastianruder/latent-variable-vecmap/tree/master).

#### `external_ot.py`
This python file involves a modified implementation of Grave et al.'s
```
@misc{grave2018unsupervised,
      title={Unsupervised Alignment of Embeddings with Wasserstein Procrustes}, 
      author={Edouard Grave and Armand Joulin and Quentin Berthet},
      year={2018},
      eprint={1805.11222},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
implementation for Gromov-Wasserstein-Procrustes alignment paradigm. We fail to find again the GitHub repository that hosts the code we have found, but after some extensive search we recognize that its code should have been located at [this repository](https://github.com/facebookresearch/MUSE). So, on top of Grave et al.'s paper per se, we have also cited all required papers noted by the repository above:

#### `get_embeddings_for_umap.py`
This python file involves the scripted procedure of obtaining the UMAP embeddings of our collected sentences and conversation tone. We first create high-dimensional sentence embeddings, then extract manifold embeddings of them using UMAP. For the usage of these methods, we have cited the following papers:
```
@article{Sanh2019DistilBERTAD,
  title={DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter},
  author={Victor Sanh and Lysandre Debut and Julien Chaumond and Thomas Wolf},
  journal={ArXiv},
  year={2019},
  volume={abs/1910.01108}
}

@misc{mcinnes2020umap,
      title={UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction}, 
      author={Leland McInnes and John Healy and James Melville},
      year={2020},
      eprint={1802.03426},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```
Additionally, the script's resulting embedding is already contained in `./data`

### `data`
This repository contains all data that was elicited from and used for analysis in this paper. Once again, note that this is the supposed content of our data as supplementary material.

#### `/alignment_benchmarking`
Bootstrapped data for cross-domain alignment that we used in our benchmarking nalysis.

#### `/human_en_fifty`, `/human_en_thirty`, `/human_en_twenty`
Data elicited from Sampling with People's human instance experiment. Files are named differently because the experiment was performed in batch. These data are anonymized.

#### `/qof-ratings`
Data elicited from Quality-of-Fit Rating's human and GPT-4 instance experiment. These data are anonymized.

#### `/similarity_judgments`
Contains both GPT-4 and Human's data from conversation tone similarity judgment experiments. These data are anonymized.

#### `/tone-feature-ratings`
Data elicited from conversation tone Rating's human and GPT-4 instance experiment. These data are anonymized.

#### `all_chains_semb_by_bert.csv`
This csv contains the `sent2vec-UMAP` sentence embeddings that we discuss in the appendix, so that the interested parties do not have to run the script for obtaining embeddings again.

#### `totalGPTData.csv`
A .csv that contains data elicited from Sampling with People's GPT-4 instance experiment.

### `gpt_experiments`
Code used for running the GPT-4 instances of our experiments.

#### `/seeds`
Some seeding files required to sample the modality, such as the initial generation of our Sampling with People paradigm.

#### `experiment.py`
The script file for running the experiments per se, equipped with options of experiment to run.

#### `paradigm.py`
The file that contains definition of experiment instances as Python objects.

### `human_experiments`
Code used for running the human instances of our experiments.
All human experiments are written using Psynet, a Python package for implementing complex online psychology experiments.
```
@misc{harrison2020gibbs,
      title={Gibbs Sampling with People}, 
      author={Peter M. C. Harrison and Raja Marjieh and Federico Adolfi and Pol van Rijn and Manuel Anglada-Tort and Ofer Tchernichovski and Pauline Larrouy-Maestri and Nori Jacoby},
      year={2020},
      eprint={2008.02595},
      archivePrefix={arXiv},
      primaryClass={q-bio.NC}
}
```

#### `/denseratingtones`
The experiment code for running quality-of-tone rating on a specified pool of sentences and conversation tones.

#### `/Sampling_With_People`
The experiment code for running Sampling with People on a specified seed of sentences and conversation tones, although the seed per se exists only to ease operation and is addressed by theory to not influence the sampling results drastically.

#### `/similarity-judgements-tone`
The experiment code for running pairwise similarity judgments on a specified pool of conversation tones.

#### `/tone-feature-rating`
The experiment code for running tone feature strength ratings on a specified pool of conversation tones.

## Declaration of Use of AI-Assistance
Following the rule set out by ACL's Ethics and AI assistance policy, we hereby declare use of the following services when completing our work:

* word-tune https://www.wordtune.com/
* Grammarly https://www.grammarly.com/
* Microsoft Copilot https://copilot.microsoft.com/
