import numpy as np
import os
import json
import pandas as pd
import gensim.downloader
import random
from sklearn.manifold import MDS
from sklearn.neighbors import KernelDensity

from nltk.stem.porter import PorterStemmer
from scipy.spatial.distance import cosine, jensenshannon, euclidean
from scipy.stats import entropy, gaussian_kde

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.patheffects as patheffects
import seaborn as sns

human_trial_fifty_df = pd.read_csv("../data/human_en_fifty/data/SRETrial.csv")
human_node_fifty_df = pd.read_csv("../data/human_en_fifty/data/SRENode.csv")
human_trial_thirty_df = pd.read_csv("../data/human_en_thirty/data/SRETrial.csv")
human_node_thirty_df = pd.read_csv("../data/human_en_thirty/data/SRENode.csv")
human_trial_twenty_df = pd.read_csv("../data/human_en_twenty/data/SRETrial.csv")
human_node_twenty_df = pd.read_csv("../data/human_en_twenty/data/SRENode.csv")

def get_compact_trial_df(human_trial_df, human_node_df):
    human_node_degrees = human_node_df[["id", "degree"]]
    human_trial_df_compact = human_trial_df.query("failed==False")[["id", "origin_id", "network_id", "previous_sample", "obtained_response", "time_taken"]]
    human_trial_df_compact = human_trial_df_compact\
        .rename(columns = {"id": "trial_id"})\
        .set_index("trial_id")
    human_trial_df_compact["previous_sample"] = human_trial_df_compact["previous_sample"].map(json.loads)
    human_trial_df_compact["provided_prompt"] = human_trial_df_compact["previous_sample"].map(lambda x: x["obtained_response"])
    human_trial_df_compact["node_mode"] = human_trial_df_compact["previous_sample"].map(lambda x: x["current_mode"])
    human_trial_df_compact = human_trial_df_compact\
                        .drop(columns=["previous_sample"])\
                        .iloc[:, [0, 1, 5, 4, 2, 3]].dropna()
    human_trial_df_compact = human_trial_df_compact.merge(human_node_degrees, left_on="origin_id", right_on="id").drop(columns="id")
    return human_trial_df_compact.query("network_id > 2")

def get_map_between_old_new(old_compact, new_comapct, degree_incr):
    old_new_network_id_map = new_comapct\
        .query("degree==0")[["network_id", "provided_prompt"]]\
        .merge(
            old_compact.query(f"degree=={degree_incr}")[["network_id", "obtained_response"]],
            left_on="provided_prompt",
            right_on="obtained_response"
        )\
        .drop_duplicates(subset=["network_id_y", "obtained_response"])[["network_id_x", "network_id_y"]]
    old_new_network_id_map = {pt[0]: pt[1] for pt in old_new_network_id_map.values}
    return old_new_network_id_map

human_fifty_compact = get_compact_trial_df(human_trial_fifty_df, human_node_fifty_df)
human_thirty_compact = get_compact_trial_df(human_trial_thirty_df, human_node_thirty_df)
old_new_network_id_map = get_map_between_old_new(human_fifty_compact, human_thirty_compact, 49)
human_thirty_compact["network_id"] = human_thirty_compact["network_id"].replace(to_replace=old_new_network_id_map)
human_thirty_compact["degree"] = human_thirty_compact["degree"] + 50
human_eighty_compact = pd.concat([human_fifty_compact, human_thirty_compact])
human_twenty_compact = get_compact_trial_df(human_trial_twenty_df, human_node_twenty_df)
old_new_network_id_map = get_map_between_old_new(human_eighty_compact, human_twenty_compact, 79)
human_twenty_compact["network_id"] = human_twenty_compact["network_id"].replace(to_replace=old_new_network_id_map)
human_twenty_compact["degree"] = human_twenty_compact["degree"] + 80
human_trial_df_compact = pd.concat([human_eighty_compact, human_twenty_compact])
human_practice_trials, human_chain_trials = human_trial_df_compact.query("network_id <= 2"), human_trial_df_compact.query("network_id > 2")
human_chain_trials_only_tones_c = human_chain_trials.query("node_mode=='c'").drop(columns="provided_prompt").rename(columns={"obtained_response": "withholding_tone"})
human_chain_trials_only_tones_s = human_chain_trials.query("node_mode=='s' and degree==0").drop(columns="obtained_response").rename(columns={"provided_prompt": "withholding_tone"})
# In the above interpretation, tones from iteration involve only tones that are sampled at the particular iteration.
human_chain_trials_only_tones = pd.concat([human_chain_trials_only_tones_c, human_chain_trials_only_tones_s])
human_chain_trials_only_tones = human_chain_trials_only_tones.sort_values(["network_id", "degree"])
human_chain_trials_only_tones["withholding_tone"] = human_chain_trials_only_tones["withholding_tone"].str.lower()
human_wanted_words = human_chain_trials_only_tones["withholding_tone"].value_counts().index
human_chain_trials_only_tones = human_chain_trials_only_tones[human_chain_trials_only_tones["withholding_tone"].isin(human_wanted_words)]

gpt_df = pd.read_csv("../data/totalGPTData.csv", sep="|", engine='python')
gpt_tones_df = gpt_df.query("node_mode == 'c'")
gpt_tones_df["node_response"] = gpt_tones_df["node_response"].str.lower()

old_gpt_df = pd.read_csv("../data/GPT_en.csv", sep="|", engine='python')
old_gpt_tones_df = old_gpt_df.query("node_mode == 'c'")
old_gpt_tones_df["node_response"] = old_gpt_tones_df["node_response"].str.lower()

from sent2vec.vectorizer import Vectorizer
from umap import UMAP
import matplotlib.cm as cm

human_chain_trials_only_sentences = human_chain_trials.query("node_mode=='s'")
gpt_df_sentences = gpt_df.query("node_mode == 's'")

temp_human_chain_trials_semb = human_chain_trials_only_sentences
temp_gpt_chain_trials_semb = gpt_df_sentences
merged_sentence = pd.concat(
    [
        temp_gpt_chain_trials_semb[["node_order", "chain_id", "node_response"]]\
            .rename(
            columns={
                "chain_id": "network_id",
                "node_order": "degree",
                "node_response": "obtained_response"
            }
            ).assign(reponse_source=["gpt" for _ in range(temp_gpt_chain_trials_semb.shape[0])]),
        temp_human_chain_trials_semb[["network_id", "degree", "obtained_response"]]\
            .assign(reponse_source=["human" for _ in range(temp_human_chain_trials_semb.shape[0])])
    ]
)

merging_vectorizer = Vectorizer()
merging_vectorizer.run(merged_sentence["obtained_response"].tolist())
merging_vectors = merging_vectorizer.vectors
merged_sembs = merged_sentence.assign(sentence_embeddings=merging_vectors)
merged_vectors_transformed = UMAP(random_state=42).fit_transform(merging_vectors)
merged_sembs = merged_sembs.assign(sentence_embeddings_umap=merged_vectors_transformed.tolist())

merged_sembs.to_csv("../export-data/all_chains_semb_by_bert.csv")
