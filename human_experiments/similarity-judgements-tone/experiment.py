# pylint: disable=unused-import,abstract-method

import logging

import psynet.experiment
from psynet.consent import MainConsent
from psynet.modular_page import ModularPage, PushButtonControl, Prompt, TextControl, NumberControl
from psynet.page import SuccessfulEndPage, InfoPage, RejectedConsentPage
from psynet.timeline import Timeline, join, Module, conditional, CodeBlock, Page, get_template
from psynet.trial.static import StaticNode, StaticTrial, StaticTrialMaker
from psynet.prescreen import LexTaleTest
from typing import Optional
from .paradigms.paradigm_SRE import SREInstruction

import pandas as pd
from markupsafe import Markup
import json
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

########################################################################################################################
# Config
########################################################################################################################

INITIAL_RECRUIT_SIZE = 5

def get_prolific_settings():
    with open("qualification_prolific_en.json", "r") as f:
        qualification = json.dumps(json.load(f))
    return {
        "recruiter": "prolific",
        "id": "scenarios-market",
        # "prolific_reward_cents": 150,
        "prolific_estimated_completion_minutes": 10,
        "prolific_maximum_allowed_minutes": 100,
        "prolific_recruitment_config": qualification,
        "base_payment": 0.0,
        "auto_recruit": False,
        "currency": "£",
        "wage_per_hour": 0.0
    }

df = pd.read_csv("human-sim-exp-pairs.csv") # change this line's "gpt" to "human"
nodes = [
    StaticNode(
        definition={
            "id": record["id"],
            "word_1": record["word_1"],
            "word_2": record["word_2"],
            "idx_1": record["idx_1"],
            "idx_2": record["idx_2"]
        },
    )
    for record in df.to_dict("records")
]

def instructions():
    return join(
        InfoPage(Markup(
            f"""
            <p>
                Thank you for participating in our study! 
            </p>
            <p>    
                In this study, we are studying how people perceive relations between conversation tones.
            </p>
            <p>    
                <strong><strong>Your task is to rate the relatedness of different pairs of conversation tones</strong></strong>.
            </p>
            <p>
                You will have five response options, ranging from 1 ('Very Unrelated') to 5 ('Very Related'). Choose the one you think is most appropriate.
            </p>
            <p>
                <strong>Note:</strong> no prior expertise is required to complete this task, just choose
                what you intuitively think is the right answer.
            </p>
            """),
            time_estimate=5
        ),
        # InfoPage(
        #     """
        #     The quality of your responses will be automatically monitored,
        #     and you will receive a bonus at the end of the experiment
        #     in proportion to your quality score. The best way to achieve
        #     a high score is to concentrate and give each round your best attempt.
        #     """,
        #     time_estimate=5
        # )
    )

class EvaluationTrial(StaticTrial):
    time_estimate = 6.5

    def finalize_definition(self, definition, experiment, participant):
        definition["order"] = random.randint(0,1)
        return definition

    def show_trial(self, experiment, participant):
        order = self.definition["order"]
        options = [self.definition["word_1"], self.definition["word_2"]]

        return ModularPage(
            "evaluation_trial",
            Markup(f"""
            <p>
                Reminder: A <b>conversation tone</b> is the style and manner in which someone speaks.
                Sometimes, it is also referred to as the tone of a sentence. <br>
                When a sentence has a conversation tone, the speaker of a sentence has a similar attitude.
            </p>
            <p>
                How related are the following two conversation tones: 
                <strong><strong>{options[order]}, {options[1-order]}</strong></strong>?
            </p>
            <p>
                If it is difficult to choose between the options,
                don't worry, and just give what you intuitively think is the right answer.
                You only need to perform 60 comparisons, we will help you automatically end the
                experiment once 60 comparisons are completed!
            </p>
            """),
            PushButtonControl(
                choices = [i for i in range(1, 6)],
                labels = [
                    # "(0) Extremely<br>Unrelated",
                    "(1) Very<br>Different",
                    "(2) Somewhat<br>Different",
                    "(3) Neither Similar<br>nor Different",
                    "(4) Somewhat<br>Similar",
                    "(5) Very<br> Similar",
                    # "(6) Extremely<br>Related"
                    ],
                style = "min-width: 50px; margin: 10px",
                arrange_vertically = False,
                bot_response=6,
            ),
            time_estimate=self.time_estimate,
        )


trial_maker = StaticTrialMaker(
    id_="evaluations",
    trial_class=EvaluationTrial,
    nodes=nodes,
    expected_trials_per_participant=60,
    max_trials_per_block=60,
    allow_repeated_nodes=False,
    balance_across_nodes=True,
    check_performance_at_end=False,
    check_performance_every_trial=False,
    target_n_participants=None,
    target_trials_per_node=5,
    recruit_mode="n_trials",
    n_repeat_trials=1,
)


class Exp(psynet.experiment.Experiment):
    label = "Evaluating Advice"
    initial_recruitment_size = 1

    config = {
        **get_prolific_settings(),
        "initial_recruitment_size": INITIAL_RECRUIT_SIZE,
        "title": "Rate the relatedness of different conversation tones! (10 mins, 1.5 GBP, Chrome incognito).",
        "description": """
        Hello! Scientists would like to learn more about the language you speak (English).
        You will be rating the relatedness of 60 different pairs of converation tones.
        The experiment should take about 10 mins with a pay of 1.5 GBP. Please use incognito mode on Google Chrome.
        """,
        "dyno_type": "performance-l",
        "num_dynos_web": 3,
        "num_dynos_worker": 2,
        "redis_size": "premium-3",
        "host": "0.0.0.0",
        "clock_on": True,
        "heroku_python_version": "3.10.6",
        "database_url": "postgresql://postgres@localhost/dallinger",
        "database_size": "standard-2",
        # "show_bonus": False
    }

    instruction_provider = SREInstruction(
        sentence_configs_path="./paradigms/practice_examples/sentence_en.json",
        tone_configs_path="./paradigms/practice_examples/tone_en.json",
        min_word_num_per_sentence=5
    )

    timeline = Timeline(
        MainConsent(),
        instruction_provider.show_instructions(),
        instructions(),
        trial_maker,
        SuccessfulEndPage(),
    )
