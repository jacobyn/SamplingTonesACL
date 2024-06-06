from importlib import resources
import json

import pandas as pd
import random

import psynet.experiment
from psynet.consent import MainConsent
from psynet.page import SuccessfulEndPage
from psynet.timeline import Timeline
from psynet.trial.static import StaticTrialMaker, StaticNode
from psynet.utils import get_logger
from .paradigms.denseRating import DenseRating, DenseRatingInstruction, SREInstruction, DenseRatingRemnant, \
    DenseRatingTrial, DenseRatingRemnantTrial

logger = get_logger()
seed_address = f"./seeds/to_rate-sentence-tones.csv"
remnant_address = f"./seeds/remnants.json"
all_sentences_tones = pd.read_csv(seed_address, sep="|")
all_sentences = all_sentences_tones.query("item_type=='sentence'")["item_content"].values
all_tones = all_sentences_tones.query("item_type=='tone'")["item_content"].values
random.seed()


def get_prolific_settings():
    # with open("prolific_config.json", "r") as f:
    #     qualification = json.dumps(json.load(f))
    return {
        "recruiter": "prolific",
        # "prolific_reward_cents": 0,
        "prolific_estimated_completion_minutes": 3,
        "prolific_maximum_allowed_minutes": 10,
        # "prolific_recruitment_config": qualification,
        "base_payment": 0.0,
        "auto_recruit": False,
        "currency": "£",
        "wage_per_hour": 0.0
    }


with open("seeds/1230remnants.json", "r") as fp:
    #The above line decides the seed file. It was `remnants` becuase a few combinations were not correctly rated.
    remnant_seeds = json.load(fp)

starting_nodes = []
for sentence in remnant_seeds:
    for tone, remnant_iter in remnant_seeds[sentence]:
        starting_nodes += [
            StaticNode(
                definition={
                    "sentence": sentence,
                    "tones": [tone]
                }
            ) for _ in range(remnant_iter)
        ]


class Exp(psynet.experiment.Experiment):
    label = "Slider demo"

    @classmethod
    def extra_files(cls):
        files = super().extra_files()
        files.append((
            resources.files("denseratingtones") / "paradigms/images",
            "/static/images",
        ))
        return files

    config = {
        "initial_recruitment_size": 10,
        "title": "Write a Sentence or Feel a Conversation Tone",
        "description": "description.",
        "language": "en",
        **get_prolific_settings()
    }

    instruction_instance = SREInstruction(
        sentence_configs_path="./paradigms/practice_examples/sentence_en.json",
        tone_configs_path="./paradigms/practice_examples/tone_en.json",
        min_word_num_per_sentence=5
    )

    timeline = Timeline(
        MainConsent(),
        instruction_instance.show_instructions(),
        DenseRatingInstruction.get_scale_instructions(),
        DenseRatingInstruction.get_practice_page(5),
        # Several Trialmakers are used here because the experiment was difficult to control and debug
        
        # DenseRating(
        #     to_rate_list=all_sentences,
        #     criteria_list=all_tones,
        #     expected_trials_per_participant=12,
        #     n_sliders=5,
        #     num_rater_per_combination=5
        # ),
        # DenseRatingRemnant(
        #     remnants_path=remnant_address,
        #     expected_trials_per_participant=15,
        #     max_trials_per_participant=15,
        #     n_sliders=5,
        #     num_rater_per_combination=5
        # ),
        StaticTrialMaker(
            id_="feature_rating",
            trial_class=DenseRatingRemnantTrial,
            nodes=starting_nodes,
            expected_trials_per_participant=9,
            max_trials_per_block=9,
            allow_repeated_nodes=False,
            balance_across_nodes=True,
            check_performance_at_end=False,
            check_performance_every_trial=False,
            target_n_participants=None,
            target_trials_per_node=1,
            recruit_mode="n_trials",
            n_repeat_trials=1,
        ),
        SuccessfulEndPage(),
    )

