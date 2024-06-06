import json
import random
from os.path import dirname

from dallinger import db
from markupsafe import Markup

from psynet.page import InfoPage
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


class DenseRating(ImitationChainTrialMaker):
    DEFAULT_NUM_TRIAL_PER_PARTICIPANT = 5
    time_estimate_per_trial = 30  # 8 * n_slider
    response_timeout_sec = 1200

    def __init__(
        self,
        to_rate_list: list[str],
        criteria_list: list[str],
        num_rater_per_combination: int = 3,
        expected_trials_per_participant: int = DEFAULT_NUM_TRIAL_PER_PARTICIPANT,
        n_sliders: int = 10,
        *args,
        **kwargs
    ):
        self.choose_participant_group = None
        self.to_rate_list = to_rate_list
        self.criteria_list = criteria_list
        self.num_rater_per_combination = num_rater_per_combination
        self.unrated_combinations = {
            to_rate_item_id: {
                "criterion_ids": [criterion_id for criterion_id in range(len(criteria_list))],
                "num_resets_left": num_rater_per_combination - 1
            }
            for to_rate_item_id in range(len(to_rate_list))
        }
        self.n_sliders = n_sliders
        kwargs["id_"] = "DenseRating"
        kwargs["trial_class"] = DenseRatingTrial
        kwargs["node_class"] = DenseRatingNode
        kwargs["chain_type"] = "across"
        kwargs["max_nodes_per_chain"] = (len(self.criteria_list) // self.n_sliders + 1) * self.num_rater_per_combination
        kwargs["recruit_mode"] = "n_trials"
        kwargs["assets"] = None
        kwargs["expected_trials_per_participant"] = expected_trials_per_participant
        kwargs["max_trials_per_participant"] = expected_trials_per_participant
        kwargs["allow_revisiting_networks_in_across_chains"] = False
        kwargs["start_nodes"] = [
            DenseRatingNode(definition=self.prepare_node_definition(to_rate_item_id))
            for to_rate_item_id in range(len(to_rate_list))
        ]
        super().__init__(
            *args,
            **kwargs
        )

    def prepare_node_definition(self, to_rate_item_id):
        sentence_info = self.unrated_combinations[to_rate_item_id]
        chosen_tones = []
        for _it in range(self.n_sliders):
            if self.refill_sentence_tones(to_rate_item_id):
                random.shuffle(sentence_info["criterion_ids"])
                chosen_tone = self.provide_unique_tone(chosen_tones, sentence_info["criterion_ids"])
                if chosen_tone is False:
                    break
                else:
                    chosen_tones.append(chosen_tone)
            else:
                break
        node_definition = {
            "to_rate_item_id": to_rate_item_id,
            "criterion_ids": chosen_tones,
            "sentence": self.to_rate_list[to_rate_item_id],
            "tones": [self.criteria_list[criterion_id] for criterion_id in chosen_tones]
        }
        return node_definition

    def provide_unique_tone(self, already_chosen_tones, tones_source):
        to_fill_back = []
        to_return_tone = False
        while len(tones_source) > 0:
            current_chosen_tone = tones_source.pop()
            if current_chosen_tone in already_chosen_tones:
                to_fill_back.append(current_chosen_tone)
            else:
                to_return_tone = current_chosen_tone
                break
        for to_fill_tone in to_fill_back:
            tones_source.append(to_fill_tone)
        return to_return_tone

    def refill_sentence_tones(self, to_rate_item_id):
        sentence_info = self.unrated_combinations[to_rate_item_id]
        if len(sentence_info["criterion_ids"]) == 0:
            if sentence_info["num_resets_left"] > 0:
                sentence_info["num_resets_left"] -= 1
                sentence_info["criterion_ids"] = [criterion_id for criterion_id in range(len(self.criteria_list))]
                return True
            else:
                return False
        return True


class DenseRatingTrial(ImitationChainTrial):
    def show_trial(self, experiment, participant):
        return DenseRatingPage(
            sentence=self.origin.definition["sentence"],
            tones=self.origin.definition["tones"],
            n_sliders=self.trial_maker.n_sliders,
            start_value=3,
            min_value=1,
            max_value=5,
            step_size=1
        )

    def finalize_trial(self, answer, trial, experiment, participant):
        super().finalize_trial(answer, trial, experiment, participant)
        num_resets_left = self.trial_maker.unrated_combinations[self.origin.definition["to_rate_item_id"]]["num_resets_left"]
        if num_resets_left == 1:
            trial.network.full = True
            db.session.commit()


class DenseRatingNode(ImitationChainNode):
    def summarize_trials(self, trials: list, experiment, participant):
        return self.trial_maker.prepare_node_definition(self.definition["to_rate_item_id"])


class DenseRatingMultiSliderControl(Control):
    macro = "dense_rating"
    external_template = "dense_rating.html"

    def __init__(
        self,
        sliders: list[Slider],
        tones_list: list[str],
        bot_response=NoArgumentProvided,
    ):
        super().__init__(bot_response)
        assert is_list_of(sliders, Slider)
        self.sliders = sliders
        self.tones_list = tones_list

    @property
    def metadata(self):
        return self.__dict__


class DenseRatingPage(ModularPage):
    page_id = 1

    def __init__(
        self,
        sentence: str,
        tones: list[str],
        n_sliders: int,
        start_value=3,
        min_value=1,
        max_value=5,
        step_size=1,
        *args,
        **kwargs
    ):
        self.sentence = sentence
        self.tones = tones
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
                        label=f"slider {_it} for page {self.page_id}",
                        slider_id=self.page_id * n_sliders + _it
                    )
                    for _it in range(len(self.tones))
                ],
                tones_list=self.tones
            ),
            js_vars={
                "ratedSentence": self.sentence,
                "ratedTones": self.tones
            },
            show_next_button=False,
            time_estimate=50,
            *args,
            **kwargs
        )

    def format_answer(self, raw_answer, **kwargs):
        return {
            "sentence": self.sentence,
            "tones": self.tones,
            "answer": raw_answer
        }


class DenseRatingRemnant(ImitationChainTrialMaker):
    DEFAULT_NUM_TRIAL_PER_PARTICIPANT = 5
    time_estimate_per_trial = 30  # 8 * n_slider
    response_timeout_sec = 1200

    def __init__(
        self,
        remnants_path: str,
        num_rater_per_combination: int = 5,
        expected_trials_per_participant: int = DEFAULT_NUM_TRIAL_PER_PARTICIPANT,
        n_sliders: int = 10,
        *args,
        **kwargs
    ):
        self.remnants_path = remnants_path
        self.choose_participant_group = None
        self.num_rater_per_combination = num_rater_per_combination
        self.n_sliders = n_sliders
        self.unrated_combinations = self.get_remnants()
        kwargs["id_"] = "DenseRatingRemnant"
        kwargs["trial_class"] = DenseRatingRemnantTrial
        kwargs["node_class"] = DenseRatingRemnantNode
        kwargs["chain_type"] = "across"
        kwargs["max_nodes_per_chain"] = 60
        kwargs["recruit_mode"] = "n_trials"
        kwargs["assets"] = None
        kwargs["expected_trials_per_participant"] = expected_trials_per_participant
        kwargs["max_trials_per_participant"] = expected_trials_per_participant
        kwargs["allow_revisiting_networks_in_across_chains"] = False
        kwargs["balance_across_chains"] = True
        kwargs["start_nodes"] = [
            DenseRatingRemnantNode(
                definition=self.prepare_node_definition(sentence)
            ) for sentence in self.unrated_combinations
        ]
        super().__init__(
            *args,
            **kwargs
        )

    def get_remnants(self):
        with open(self.remnants_path, "r") as cost_f:
            cost_json = json.load(cost_f)
        remnants = {}
        for sentence, criteria in cost_json.items():
            current_sentence_remnants = {}
            for tone, num_remnant in criteria:
                current_sentence_remnants[tone] = num_remnant
            remnants[sentence] = current_sentence_remnants
        return remnants

    def prepare_node_definition(self, node_sentence):
        all_tones_of_sentence = self.unrated_combinations[node_sentence].keys()
        tones_available = [
            tone for tone in all_tones_of_sentence
            if self.unrated_combinations[node_sentence][tone] > 0
        ]
        tones_to_rate = random.sample(tones_available, min(self.n_sliders, len(tones_available)))
        db.session.commit()
        return {
            "sentence": node_sentence,
            "tones": tones_to_rate,
        }

    def finalize_trial(self, answer, trial, experiment, participant):
        super().finalize_trial(answer, trial, experiment, participant)
        remnants_of_cur_sentence = self.unrated_combinations[trial.origin.definition["sentence"]]
        for tone in answer["tones"]:
            self.unrated_combinations[answer["sentence"]][tone] -= 1
        has_tones_yet_rated = any([num_remnant > 0 for num_remnant in remnants_of_cur_sentence.values()])
        logger.info(f"ATTENTION: {remnants_of_cur_sentence.values()}")
        if not has_tones_yet_rated:
            trial.network.full = True
            db.session.commit()


class DenseRatingRemnantNode(ImitationChainNode):
    def summarize_trials(self, trials: list, experiment, participant):
        return self.trial_maker.prepare_node_definition(self.definition["sentence"])


class DenseRatingRemnantTrial(ImitationChainTrial):
    time_estimate = 60

    def show_trial(self, experiment, participant):
        return DenseRatingPage(
            sentence=self.origin.definition["sentence"],
            tones=self.origin.definition["tones"],
            n_sliders=len(self.origin.definition["tones"]),
            start_value=3,
            min_value=1,
            max_value=5,
            step_size=1
        )


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
                tones on each of the provided sentences.</h5>
                <br><br>
                <div class='row'>
                    <div class='col-md-6'>
                        On the right side of the screen, you can see a picture representing the scalebar interface.<br>
                        You will be rating the strength of a conversation tone on a scale of 1 to 5, with 1 being the
                        weakest, 5 being the strongest. <br>
                        For each page, you rate the conversation tone of only the provided sentence on the screen. <br>
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
    def get_practice_page(n_sliders):
        random.seed()
        before_practice = InfoPage("Now, let's warm up by doing some short practice rounds!", time_estimate=2)
        with open(f"{dirname(__file__)}/seeds/sentence_en.txt") as sentence_f:
            example_sentence = random.choice(sentence_f.readlines())[:-1].replace("'", "`")
        with open(f"{dirname(__file__)}/seeds/tone_en.txt") as tone_f:
            example_tones = random.sample(tone_f.readlines(), k=n_sliders)
            example_tones = [tone[:-1] for tone in example_tones]
        practice_page = DenseRatingPage(
            sentence=example_sentence,
            tones=example_tones,
            n_sliders=n_sliders
        )
        after_practice = DenseRatingInstruction.instruction_page_before_starting()
        return join(before_practice, practice_page, after_practice)

