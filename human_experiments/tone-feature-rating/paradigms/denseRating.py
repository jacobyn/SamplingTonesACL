import json
import random
from os.path import dirname

from dallinger import db
from markupsafe import Markup

from psynet.page import InfoPage
from psynet.trial.static import StaticTrial
from psynet.utils import get_logger
from psynet.modular_page import ModularPage, Slider, Control
from psynet.timeline import is_list_of, join
from psynet.trial.imitation_chain import (
    ImitationChainNode,
    ImitationChainTrial,
    ImitationChainTrialMaker,
)
from psynet.utils import NoArgumentProvided

logger = get_logger()


class DenseRatingTrial(StaticTrial):
    time_estimate = 10

    def show_trial(self, experiment, participant):
        return DenseRatingPage(
            feature=self.origin.definition["feature"],
            feature_description=self.origin.definition["feature_description"],
            tone=self.origin.definition["tone"],
            start_value=3,
            min_value=1,
            max_value=5,
            step_size=1,
        )


class DenseRatingMultiSliderControl(Control):
    macro = "dense_rating"
    external_template = "dense_rating.html"

    def __init__(
        self,
        sliders: list[Slider],
        tone_list: list[str],
        bot_response=NoArgumentProvided,
    ):
        super().__init__(bot_response)
        assert is_list_of(sliders, Slider)
        self.sliders = sliders
        self.tone_list = tone_list

    @property
    def metadata(self):
        return self.__dict__


class DenseRatingPage(ModularPage):
    page_id = 1

    def __init__(
        self,
        feature: str,
        feature_description: str,
        tone: str,
        start_value=3,
        min_value=1,
        max_value=5,
        step_size=1,
        *args,
        **kwargs
    ):
        self.feature = feature
        self.feature_description = feature_description
        self.tone = tone
        self.page_id = DenseRatingPage.page_id
        DenseRatingPage.page_id += 1
        prompt = Markup(
            """
                <h3>Preface</h3>
                A <b>conversation tone</b> is the style and manner in which someone speaks.
                Sometimes, it is also referred to as the tone of a sentence. <br>
                When a sentence has a conversation tone, the speaker of a sentence has a similar attitude. <br><br>
            """
        )
        super().__init__(
            label="slider_page",
            prompt=prompt,
            control=DenseRatingMultiSliderControl(
                sliders=[
                    Slider(
                        start_value=start_value,
                        min_value=min_value,
                        max_value=max_value,
                        step_size=step_size,
                        label=f"slider for page {self.page_id}",
                        slider_id = self.page_id
                    )
                ],
                tone_list=[self.tone]
            ),
            js_vars={
                "ratedFeature": self.feature,
                "ratedFeatureDescription": self.feature_description,
                "ratedTones": [self.tone]
            },
            time_estimate=10,
            show_next_button=False,
            *args,
            **kwargs
        )

    def format_answer(self, raw_answer, **kwargs):
        return {
            "feature": self.feature,
            "tones": self.tone,
            "answer": raw_answer
        }


class SREInstruction:
    def __init__(self, sentence_configs_path, tone_configs_path, min_word_num_per_sentence):
        self.sentence_configs_path = sentence_configs_path
        self.tone_configs_path = tone_configs_path
        self.min_word_num_per_sentence = min_word_num_per_sentence

    def show_instructions(self):
        random.seed()  # When provided no input, the current system time is used as RNG seed.
        with open(self.sentence_configs_path) as sentence_configs:
            example_configs_sentence = json.load(sentence_configs)
        with open(self.tone_configs_path) as tone_configs:
            example_configs_tone = json.load(tone_configs)
        example_config_sentence = random.choice(list(example_configs_sentence.values()))
        example_config_tone = random.choice(list(example_configs_tone.values()))
        return join(
            [
                SREInstruction.instruction_page_greetings(),
                SREInstruction.instruction_page_operationalization(),
                SREInstruction.get_sentence_example_page(example_config_sentence),
                SREInstruction.get_tone_example_page(
                    example_config_tone, self.min_word_num_per_sentence
                ),
                SREInstruction.get_tone_clarification_page()
            ]
        )

    def show_all_instructions(self):
        with open(self.sentence_configs_path) as sentence_configs:
            example_configs_sentence = json.load(sentence_configs)
        with open(self.tone_configs_path) as tone_configs:
            example_configs_tone = json.load(tone_configs)
        example_config_sentence = list(example_configs_sentence.values())
        example_config_tone = list(example_configs_tone.values())
        return join(
            sum(
                [[
                    SREInstruction.get_sentence_example_page(example_config_sentence[i]),
                    SREInstruction.get_tone_example_page(
                        example_config_tone[i], self.min_word_num_per_sentence
                    )
                ] for i in range(len(example_config_tone))], []
            )
        )

    @staticmethod
    def instruction_page_greetings():
        return InfoPage(
            Markup(
                """
                Hello, welcome to the experiment! <br>
                In this experiment, scientists would like to learn more about your language by asking you
                about sentences and conversation tones in your language.
                """
            ),
            time_estimate=5
        )

    @staticmethod
    def instruction_page_operationalization():
        return InfoPage(
            Markup(
                """
                What is a conversation tone? <br>
                A <b>conversation tone</b> is the style and manner in which someone speaks.
                Sometimes, it is also referred to as the tone of a sentence. <br>
                A few examples will be shown in the following page. <br>
                <b> Please read them carefully to avoid confusion. </b>
                """
            ),
            time_estimate=10
        )

    @staticmethod
    def get_sentence_example_page(example_config):
        return InfoPage(
            Markup(
                """
                <style>
                    .alert {
                        padding: 15px;
                        margin-bottom: 20px;
                        border: 1px solid transparent;
                        border-radius: 4px;
                        background-color: #b2c3db;
                        border-color: #cdc6e9;
                        color: #3c3c76;
                    }
                </style>
                """ +
                f"""
                    For the first example, let's look at the following sentence:
                    <div class="alert">
                        Provide an <b>adjective</b> for conversation tone in your language that you sense in
                        the following sentence:
                        <p style="margin-left: 25px;">
                            {example_config["example_sentence"]}
                        </p>
                    </div>
                    This sentence can have the following conversation tone:
                    <ul>
                        <li>
                            <b>{example_config["example_tone_1"]}</b>:
                            {example_config["example_tone_1_explanation"]}
                        </li>
                        <li>
                            <b>{example_config["example_tone_2"]}</b>:
                            {example_config["example_tone_2_explanation"]}
                        </li>
                    </ul>
                    as you can see, each sentence can have a lot of conversation tones. <br>
                    In this experiment, we only want you to choose the conversation tone you resonate the most with,
                    using an adjective.
                """
            ),
            time_estimate=20
        )

    @staticmethod
    def get_tone_example_page(example_config, min_word_num_per_sentence):
        return InfoPage(
            Markup(
                """
                <style>
                    .alert {
                        padding: 15px;
                        margin-bottom: 20px;
                        border: 1px solid transparent;
                        border-radius: 4px;
                        background-color: #b2c3db;
                        border-color: #cdc6e9;
                        color: #3c3c76;
                    }
                </style>""" +
                f"""
                        For a second example, let's look at the following conversation tone:
                        <div class="alert">
                            Provide a sentence with <b>at least {min_word_num_per_sentence} words</b>
                            in your language that has the following conversation tone:
                            <p style="margin-left: 25px;">
                                {example_config["example_tone"]}
                            </p>
                        </div>
                        Here are some example sentences:
                        <ul>
                            <li>
                                <b>Sentence</b>: "{example_config["example_sentence_1"]}"<br>
                                <b>Explanation</b>: {example_config["example_sentence_1_explanation"]}
                            </li>
                            <li>
                                <b>Sentence</b>: "{example_config["example_sentence_2"]}"<br>
                                <b>Explanation</b>: {example_config["example_sentence_2_explanation"]}
                            </li>
                        </ul>
                        as you can see, each conversation tone can have a lot of sentences. <br>
                    """
            ),
            time_estimate=20
        )

    @staticmethod
    def get_tone_clarification_page():
        return InfoPage(
            Markup(
                """
                <style>
                    .alert {
                        padding: 15px;
                        margin-bottom: 20px;
                        border: 1px solid transparent;
                        border-radius: 4px;
                        background-color: #b2c3db;
                        border-color: #cdc6e9;
                        color: #3c3c76;
                    }
                </style>
                We also make a clarification on how a sentence has a certain conversation tone. <br>
                Let us consider the sentences:
                <div class="alert">
                    <ol>
                        <li>He sounds sorry.</li>
                        <li>I am really sorry.</li>
                    </ol>
                </div>
                Sentence 1 does contain the word "sorry", <b>but the speaker does not sound apologetic</b>.
                Therefore, the conversation tone of <b>Sentence 1 is not sorry</b>; rather, the tone is more
                descriptive. <br>
                Sentence 2, on the other hand, <b>has a speaker with an apologetic attitude</b>, so the sentence
                <b>sounds sorry</b>. Therefore, Sentence 2 has an apologetic conversation tone.
                """
            ),
            time_estimate=20
        )


class DenseRatingInstruction:
    def __init__(self):
        pass

    @staticmethod
    def get_scale_instructions():
        instruction_img_path = "/static/images/scalebar_tutorial.png"
        return InfoPage(
            Markup(
                f"""
                <h3>Scale Bar Instructions</h3>
                <h5>In this experiment, scientists will learn about your language by acquiring strength of conversation
                tones on each of the provided properties.</h5>
                <br><br>
                <div class='row'>
                    <div class='col-md-6'>
                        On the right side of the screen, you can see a picture representing the scalebar interface.<br>
                        You will be rating how strong a conversation tone is on some provided property, on a scale of
                        1 to 5, with 1 being the weakest, 5 being the strongest. <br>
                        For each page, you rate the conversation tone of only the provided property on the screen. <br>
                        You may click at the position of the number on the scaling bar to rate points accordingly. <br>
                        The definition of conversation tones will appear at top of the page everytime for reference.
                    </div>
                    <div class='col-md-6'>
                        <img src="{instruction_img_path}" style="max-height: 70vh; height: auto;"/>
                    </div>
                </div>
                """
            ),
            time_estimate=10
        )

    @staticmethod
    def instruction_page_before_starting():
        return InfoPage(
            Markup(
                """
                We are going to start the experiment once you click "Next". <br>
                Have a good time!
                """
            ),
            time_estimate=3
        )

    @staticmethod
    def get_practice_page(human_tones):
        random.seed()
        before_practice = InfoPage("Now, let's warm up by doing some short practice rounds!", time_estimate=2)
        # with open(f"{dirname(__file__)}/seeds/tone_en.txt") as tone_f:
        #     example_tones = random.sample(tone_f.readlines(), k=n_sliders)
        #     example_tones = [tone[:-1] for tone in example_tones]
        feature_dictionary = {
            "positive in valence": "the positiveness of emotional valence, emotional pleasantness.",
            "aroused": "the strength of emotional activation and energy observed.",
            "informational": "the extent to which the motive of speaker focuses on giving and/or receiving accurate information.",
            "relational": "the extent to which the motive of speaker motive focuses on building the relationship."
        }
        target_feature, target_description = random.choice(list(feature_dictionary.items()))
        practice_page = DenseRatingPage(
            feature=target_feature,
            feature_description=target_description,
            tone=random.choice(human_tones)
        )
        after_practice = DenseRatingInstruction.instruction_page_before_starting()
        return join(before_practice, practice_page, after_practice)

