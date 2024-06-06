# pylint: disable=unused-import,abstract-method,unused-argument
import random

from psynet.trial.static import StaticNode

import psynet.experiment

# from psynet.bot import Bot
from psynet.consent import MainConsent

# from psynet.experiment import is_experiment_launched
from psynet.page import InfoPage, SuccessfulEndPage
from psynet.timeline import Timeline
from psynet.utils import get_logger

from .paradigms.paradigm_SRE import SRE, SREInstruction, SREPractice, SREDirectElicit, SREDirectElicitTrial

# from dallinger.experiment import scheduled_task

logger = get_logger()
DEFAULT_MIN_WORD_PER_SENTENCE = 5
random.seed()


def get_prolific_settings():
    # with open("prolific_config.json", "r") as f:
    #     qualification = json.dumps(json.load(f))
    return {
        "recruiter": "prolific",
        "prolific_reward_cents": 30,
        "prolific_estimated_completion_minutes": 3,
        "prolific_maximum_allowed_minutes": 10,
        # "prolific_recruitment_config": qualification,
        "base_payment": 0.01,
    }


class Exp(psynet.experiment.Experiment):
    label = "Imitation chain demo"
    NUM_CHAIN = 90
    MAX_NODES_PER_CHAIN = 20
    TRIALS_PER_PARTICIPANT = 10

    config = {
        "initial_recruitment_size": 10,
        "title": "Write a Sentence or Feel a Conversation Tone",
        "description": "description.",
        "language": "en",
        "wage_per_hour": 0.01,
        **get_prolific_settings()
    }
    instruction_provider = SREInstruction(
        sentence_configs_path="./paradigms/practice_examples/sentence_en.json",
        tone_configs_path="./paradigms/practice_examples/tone_en.json",
        min_word_num_per_sentence=DEFAULT_MIN_WORD_PER_SENTENCE
    )

    timeline = Timeline(
        MainConsent(),
        # instruction_provider.show_all_instructions(), # For testing all instructions, enable this line and turn on bot
        instruction_provider.show_instructions(),
        SRE.instruction_page_before_practicing(),
        SREPractice(
            max_nodes_per_chain=int((NUM_CHAIN * MAX_NODES_PER_CHAIN // TRIALS_PER_PARTICIPANT + 1) * 1.26),
            # The +20 is set above as a buffer to the precise amount of practice nodes to prepare.
            sentence_seed_path="./paradigms/seeds/sentence_en.txt",
            tone_seed_path="./paradigms/seeds/tone_en.txt",
        ),
        SRE.instruction_page_before_starting(),
        SRE(
            sentence_seed_path="./paradigms/seeds/sentence_en.txt",
            tone_seed_path="./paradigms/seeds/tone_en.txt",
            num_chains=NUM_CHAIN,
            max_nodes_per_chain=MAX_NODES_PER_CHAIN,
            n_trials_per_participant=TRIALS_PER_PARTICIPANT,
            trials_per_node=1,
            min_word_num_per_sentence=DEFAULT_MIN_WORD_PER_SENTENCE,
            id_="SRE_chain"
        ),
        InfoPage("You finished the experiment!", time_estimate=0),
        SuccessfulEndPage(),
    )

    # def test_check_bot(self, bot: Bot, **kwargs):
    #     assert len(bot.alive_trials) == 5
    #
    # @staticmethod
    # @scheduled_task("interval", minutes=4 / 60, max_instances=1)
    # def run_bot_participant():
    #     if is_experiment_launched():
    #         bot = Bot()
    #         bot.take_experiment()
