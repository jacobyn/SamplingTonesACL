import random

from os.path import abspath
from datetime import datetime

import paradigm as SREGPT

TODAY = "-".join(str(datetime.today()).split(" "))[:20]
TODAY_STR = TODAY.replace(".", "-").replace(":", "-")
RESULT_ADDRESS = f"{abspath(__file__)}/../experiment-results/{TODAY_STR}-results"
INIT_ADDRESS = f"{abspath(__file__)}/../experiment-results/{TODAY_STR}-inits"
NUM_CHAINS = 7
with open(f"{abspath(__file__)}/../seeds/sentences.txt", "r", encoding="utf8") as f:
    sentence_list = list(map(lambda x: x.replace("\n", ""), f.readlines()))
with open(f"{abspath(__file__)}/../seeds/tones.txt", "r", encoding="utf8") as f:
    tones_list = list(map(lambda x: x.replace("\n", ""), f.readlines()))

random.shuffle(sentence_list)
random.shuffle(tones_list)

EXPERIMENT_CONFIG = {
    "num_chains": 8,
    "num_nodes_per_chain": 90 + 1, #Here is a +1 because seeds take away one slot
    "min_num_word_per_sentence": 5,
    "GPT_model": "GPT-4",
    "initiation_mode": "together",
    "aggregation_fold": 1,
    "sleep_interval": 4,
    "language": "en"
}
INIT_CONFIG = {
    "sentences": sentence_list,
    "tones": tones_list
}

CURRENT_ACT = "featuring"
assert CURRENT_ACT in ["initiating", "sampling", "rating", "featuring", "similarity"]

if CURRENT_ACT == "similarity":
    open(RESULT_ADDRESS + "-similarity.csv", "x")
    SIMILARITY_INSTANCE = SREGPT.SRESimilarityInstance(
        gpt_extraction_path = "./experiment-results/rest_of_tones.csv",
        human_extraction_path="./experiment-results/rest_of_tones.csv",
        result_address = RESULT_ADDRESS + "-similarity.csv",
        locale = "en",
        sleep_interval = 1.5,
        GPT_model= EXPERIMENT_CONFIG['GPT_model']
    )
    SIMILARITY_INSTANCE.run_rating_instance()
elif CURRENT_ACT == "sampling":
    open(RESULT_ADDRESS + "-sampling.csv", "x")
    EXPERIMENT_CONFIG["result_address"] = RESULT_ADDRESS + "-sampling.csv"
    EXPERIMENT = SREGPT.SREExperiment(
        initiation_config=INIT_CONFIG,
        **EXPERIMENT_CONFIG
    )
    experiment_log = EXPERIMENT.run_experiment()
elif CURRENT_ACT == "featuring":
    open(RESULT_ADDRESS + "-features.csv", "x")
    FEATURE_INSTANCE = SREGPT.SREFeatureExtractionInstace(
        extraction_path="./experiment-results/rest_of_tones.csv",
        result_address=RESULT_ADDRESS + "-features.csv",
        feature_names={
            "positive in valence": "Positiveness in valence means the positiveness of emotional valence, emotional pleasantness.",
            "aroused": "Aroused means the strength of emotional activation and energy observed.",
            "Informational": "Informational means the extent to which the motive of speaker focuses on giving and/or receiving accurate information.",
            "Relational": "Relational means the extent to which the motive of speaker focuses on building the relationship."
            # "positive in valence": "Positiveness in valence means the positiveness of emotional valence.",
            # "aroused": "Aroused means the amount of emotional arousal observed.",
            # "Informational": "Informational means the extent to which a speaker's motive focuses on giving and/or receiving accurate information.",
            # "Relational": "Relational means the extent to which a speaker's motive focuses on building the relationship."
        },
        sleep_interval=2,
        num_raters=5,
        GPT_model=EXPERIMENT_CONFIG['GPT_model']
    )
    FEATURE_INSTANCE.run_feature_instance()
elif CURRENT_ACT == "rating":
    open(RESULT_ADDRESS + "-rating.csv", "x")
    RATING_INSTANCE = SREGPT.SRERatingInstance(
        extraction_path = "./experiment-results/GPT-sentence-tones.csv",
        result_address = RESULT_ADDRESS + "-rating.csv",
        num_raters = 5,
        locale = "en",
        sleep_interval = 1.5,
        GPT_model= EXPERIMENT_CONFIG['GPT_model']
    )
    RATING_INSTANCE.run_rating_instance()
elif CURRENT_ACT == "initiating":
    #elicit prompts
    SREGPT.SREExperiment.extract_initiation_points(
        "c", NUM_CHAINS // 2 + NUM_CHAINS % 2, INIT_ADDRESS + "-tones.txt"
    )
    SREGPT.SREExperiment.extract_initiation_points(
        "s", NUM_CHAINS // 2, INIT_ADDRESS + "-sentences.txt"
    )

