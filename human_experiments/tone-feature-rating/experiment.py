from importlib import resources

import pandas as pd
import random
from dallinger.config import get_config

import psynet.experiment
from psynet.trial.static import StaticTrialMaker, StaticNode
from psynet.consent import MainConsent
from psynet.page import SuccessfulEndPage
from psynet.timeline import Timeline
from psynet.utils import get_logger
from .paradigms.denseRating import DenseRatingInstruction, DenseRatingTrial, SREInstruction

logger = get_logger()
# seed_address = f"./seeds/to_rate-sentence-tones.csv"
# remnant_address = f"./seeds/remnants.json"
# all_sentences_tones = pd.read_csv(seed_address, sep="|")
# all_sentences = all_sentences_tones.query("item_type=='sentence'")["item_content"].values
# all_tones = all_sentences_tones.query("item_type=='tone'")["item_content"].values
random.seed()
human_tones = pd.read_csv("./seeds/all-human-tones.csv")["word_1"]
# gpt_tones = pd.read_csv("./seeds/all-gpt-tones.csv")["word_1"]


def get_prolific_settings():
    # with open("prolific_config.json", "r") as f:
    #     qualification = json.dumps(json.load(f))
    return {
        "recruiter": "prolific",
        "id": "scenarios-market",
        "prolific_reward_cents": 150,
        "prolific_estimated_completion_minutes": 10,
        "prolific_maximum_allowed_minutes": 100,
        # "prolific_recruitment_config": qualification,
        "base_payment": 0.0,
        "auto_recruit": False,
        "currency": "£",
        "wage_per_hour": 0.0
    }


feature_dictionary = {
            "positive in valence": "the positiveness of emotional valence, emotional pleasantness.",
            "aroused": "the strength of emotional activation and energy observed.",
            "informational": "the extent to which the motive of speaker focuses on giving and/or receiving accurate information.",
            "relational": "the extent to which the motive of speaker motive focuses on building the relationship."
}


class Exp(psynet.experiment.Experiment):

    @classmethod
    def extra_files(cls):
        files = super().extra_files()
        files.append((
            resources.files("tone-feature-rating") / "paradigms/images",
            "/static/images",
        ))
        return files

    config = {
        **get_prolific_settings(),
        "initial_recruitment_size": 5,
        "title": "Rate the relatedness of different conversation tones! (8 mins).",
        "description": """
        Hello! Scientists would like to learn more about the language you speak (English).
        You will be rating the features of converation tones for 30 times.
        The experiment should take about 8 mins with a pay of 1.2 GBP. Please use incognito mode on Google Chrome. 
        """,
        "show_bonus": False
    }

    # @classmethod
    # def extra_parameters(cls):
    #     get_config().register("cap_recruiter_auth_token", str, [], False)

    instruction_instance = SREInstruction(
        sentence_configs_path="./paradigms/practice_examples/sentence_en.json",
        tone_configs_path="./paradigms/practice_examples/tone_en.json",
        min_word_num_per_sentence=5
    )

    trial_maker = StaticTrialMaker(
        id_="feature_rating",
        trial_class=DenseRatingTrial,
        nodes=[
            StaticNode(
                definition={
                    "feature": feature,
                    "feature_description": description,
                    "tone": target_tone
                }
            ) for target_tone in human_tones for feature, description in feature_dictionary.items()
        ],
        expected_trials_per_participant=30,
        max_trials_per_block=30,
        allow_repeated_nodes=False,
        balance_across_nodes=True,
        check_performance_at_end=False,
        check_performance_every_trial=False,
        target_n_participants=None,
        target_trials_per_node=5,
        recruit_mode="n_trials",
        n_repeat_trials=1,
    )

    timeline = Timeline(
        MainConsent(),
        instruction_instance.show_instructions(),
        # DenseRatingInstruction.get_scale_instructions(),
        DenseRatingInstruction.get_practice_page(human_tones),
        # DenseRating(
        #     to_rate_list=all_sentences,
        #     criteria_list=all_tones,
        #     expected_trials_per_participant=12,
        #     n_sliders=5,
        #     num_rater_per_combination=5
        # ),
        trial_maker,
        SuccessfulEndPage(),
    )

