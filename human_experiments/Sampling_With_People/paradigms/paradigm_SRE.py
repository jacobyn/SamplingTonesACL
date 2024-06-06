import datetime
import json
import random
import re
import string

import nltk

from dallinger import db
from psynet.trial.static import StaticTrialMaker, StaticTrial, StaticNode

nltk.download('wordnet')
from nltk.stem import PorterStemmer

from autocorrect import Speller
# from gingerit.gingerit import GingerIt
from markupsafe import Markup
from profanity_check import predict_prob as predict_profanity
from PyDictionary import PyDictionary

from psynet.modular_page import ModularPage, Prompt, TextControl
from psynet.page import InfoPage
from psynet.timeline import FailedValidation, join
from psynet.trial.imitation_chain import (
    ImitationChainNetwork,
    ImitationChainNode,
    ImitationChainTrial,
    ImitationChainTrialMaker,
)
from psynet.utils import get_logger

logger = get_logger()
# ginger_parse = GingerIt()
english_speller = Speller("en")
english_dictionary = PyDictionary()
english_stemmer = PorterStemmer()

# TODO: Process this prototype for Mandarin.


class SREPage(ModularPage):
    def __init__(
            self,
            prompt: str,
            current_mode: str,
            page_trial,
            time_estimate: int = 40,
            min_word_num_per_sentence: int = 5
    ):
        if current_mode == "c":
            bot_response = ''.join(self.get_random_word())
        else:
            bot_response = ' '.join([self.get_random_word() for _it in range(min_word_num_per_sentence)]) + "."
        self.current_mode = current_mode
        self.min_word_num_per_sentence = min_word_num_per_sentence
        self.page_trial = page_trial
        super().__init__(
            prompt=Prompt(Markup(prompt)),
            control=TextControl(
                block_copy_paste=True,
                bot_response=lambda bot: bot_response
            ),
            label="SREPage",
            time_estimate=time_estimate
        )

    def format_answer(self, raw_answer, **kwargs):
        if self.current_mode == "c":
            raw_answer = raw_answer.lower()
        else:
            raw_answer = raw_answer.replace("'", "`")
        return {
            "obtained_response": raw_answer,
            "current_mode": self.page_mode_switch(self.current_mode)
        }

    def validate(self, response, **kwargs):
        raw_answer = response.answer["obtained_response"]
        if self.validate_by_time():
            regex_validation = self.validate_by_regex(
                raw_answer.replace("`", "'"), self.current_mode, self.min_word_num_per_sentence
            )
            if regex_validation is not None:
                return regex_validation
            profanity_validation = self.validate_profanity_by_svm(raw_answer.replace("`", "'"))
            if profanity_validation is not None:
                return profanity_validation
            if self.current_mode == "direct_elicit":
                spelling_validation = self.validate_spelling(raw_answer)
                if spelling_validation is not None:
                    return spelling_validation
                similarity_validation = self.validate_similar_response(raw_answer, self.page_trial.participant.var.get("answers"))
                if similarity_validation is not None:
                    return similarity_validation
            elif self.current_mode == "c":
                spelling_validation = self.validate_spelling(raw_answer)
                if spelling_validation is not None:
                    return spelling_validation
                stemming_validation = self.validate_stemming(
                    self.page_trial.var.get("prompt_component"), raw_answer, self.current_mode
                )
                if stemming_validation is not None:
                    return stemming_validation
            else:
                # grammar_validation = self.validate_grammar_by_ginger(raw_answer.replace("`", "'"))
                # if grammar_validation is not None:
                #     return grammar_validation
                stemming_validation = self.validate_stemming(
                    raw_answer, self.page_trial.var.get("prompt_component"), self.current_mode
                )
                if stemming_validation is not None:
                    return stemming_validation
            return None

    def validate_by_time(self):
        time_taken = datetime.datetime.utcnow() - self.page_trial.var.get("start_time")
        logger.info(f"ATTENTION: time taken is {time_taken.seconds}.")
        if time_taken.seconds >= 300:
            logger.info("ATTENTION: trial successfully failed")
            self.page_trial.fail()
            return False
        return True

    @staticmethod
    def validate_stemming(raw_answer, unpermitted_word, current_mode):
        logger.info(re.split(r"\W", raw_answer))
        for raw_word in re.split(r"\W", raw_answer):
            logger.info(f"ATTENTION: {english_stemmer.stem(raw_word)} vs. {english_stemmer.stem(unpermitted_word)}")
            if english_stemmer.stem(raw_word) == english_stemmer.stem(unpermitted_word):
                if current_mode == "s":
                    return FailedValidation(
                        f"Your response contains the word {raw_word}, which is too similar to {unpermitted_word}."
                    )
                else:
                    return FailedValidation(
                        f"Your word is {unpermitted_word}, which is similar to {raw_word} in the given sentence."
                    )

    @staticmethod
    def validate_spelling(raw_answer):
        try:
            word_meanings = english_dictionary.meaning(raw_answer)
            if "Adjective" not in word_meanings:
                return FailedValidation(
                    f"Your word seems to be a {', '.join(word_meanings.keys())} rather than an adjective."
                )
        except (re.error, TypeError):
            addenda = ""
            autocorrected = english_speller(raw_answer)
            if raw_answer != autocorrected:
                addenda = f"But, here are some similar words our database has: {autocorrected}"
            return FailedValidation(f"We can't find the word you inputted in our dictionary. {addenda}")

    @staticmethod
    def validate_grammar_and_profanity_by_gpt(raw_answer):
        return

    @staticmethod
    def validate_profanity_by_svm(raw_answer):
        if predict_profanity([raw_answer])[0] >= 0.7:
            return FailedValidation(
                "We found out that your sentence possibly contains profanity. Do you mind changing a sentence?"
            )

    @staticmethod
    def validate_similar_response(raw_answer, other_answers):
        logger.info(f"Raw answer is {raw_answer}")
        logger.info(f"other answers are {other_answers}")
        list_other_answers = [a["obtained_response"].strip(" ").lower() for a in other_answers]
        if raw_answer.strip(" ").lower() in list_other_answers:
            return FailedValidation(
                f"We see that {raw_answer.strip(' ')} is contained in past responses {list_other_answers}, please change a word"
            )

    # @staticmethod
    # def validate_grammar_by_ginger(raw_answer):
    #     logger.info(f"ATTENTION: {raw_answer}")
    #     ginger_suggestions = ginger_parse.parse(raw_answer)
    #     if len(ginger_suggestions["corrections"]):
    #         return FailedValidation(
    #             "Our system detects a possible grammar error in your sentence. " +
    #             f"It also suggests that a revision: {ginger_suggestions['result']}"
    #         )

    @staticmethod
    def validate_by_regex(raw_answer, current_mode, min_word_num_per_sentence):
        if current_mode in ["c", "direct_elicit"]:
            pattern = re.compile(r"^[A-Za-z-]+$")
            if re.fullmatch(pattern, raw_answer) is None:
                return FailedValidation("Please enter one single adjective with no space or punctuations.")
        elif current_mode == "s":
            pattern = re.compile(
                r"^(['`\"\w\[\](){},;:\-]+?\s){" +
                str(min_word_num_per_sentence - 1) +
                r",}['`\"\w\[\](){}\-]+[.?!][`'\"\])}]?$"
            )
            if re.fullmatch(pattern, raw_answer) is None:
                return FailedValidation(
                    f"Please enter a sentence with at least {min_word_num_per_sentence} words " +
                    "with an ending punctuation."
                )

    @staticmethod
    def page_mode_switch(mode):
        return "c" if mode == "s" else "s"

    @staticmethod
    def get_random_word():
        return ''.join(random.choices(string.ascii_uppercase, k=5))


class SREDefinition:
    def __init__(
        self,
        current_mode: str,
        previous_sample: str = "",
    ):
        assert current_mode in ["c", "s"],\
            'mode of SREDefinition must be either "c" or "s".'
        self.previous_sample = previous_sample
        self.current_mode = current_mode


class SRETrial(ImitationChainTrial):
    time_estimate = 40

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.var.set("start_time", datetime.datetime.utcnow())
        self.var.set("prompt_component", self.origin.definition["previous_sample"]["obtained_response"])

    def show_trial(self, experiment, participant):
        trial_definition = self.origin.definition
        self.var.set("mode", trial_definition["current_mode"])
        resulting_page = SREPage(
            prompt=self.decide_prompt(
                current_mode=trial_definition["current_mode"],
                previous_stimulus=trial_definition["previous_sample"]
            ),
            current_mode=trial_definition["current_mode"],
            min_word_num_per_sentence=self.trial_maker.min_word_num_per_sentence,
            page_trial=self
        )
        return resulting_page

    def decide_prompt(self, current_mode, previous_stimulus):
        mode_to_demand_map = {
            "c": "an adjective for a conversation tone",
            "s": f"a sentence with at least {self.trial_maker.min_word_num_per_sentence} words"
        }
        preface_definition = """
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
            A <b>conversation tone</b> is the style and manner in which someone speaks.
            Sometimes, it is also referred to as the tone of a sentence. <br>
            When a sentence has a conversation tone, the speaker of a sentence has a similar attitude. <br><br>
        """
        entire_prompt = f"{preface_definition} Provide {mode_to_demand_map[current_mode]} in your language "
        if previous_stimulus:
            quoted_previous_stimulus = f'"{previous_stimulus["obtained_response"]}"'
            if current_mode == "c":
                entire_prompt += f"that you sense in the sentence:"
            else:
                entire_prompt += f"that has the conversation tone:"
            entire_prompt += f"""
                                <div class='alert'>
                                    <p style="margin-left: 25px;">
                                        {quoted_previous_stimulus}
                                    </p>
                                </div>
                            """
            entire_prompt += "Do not include any variation of the associated conversation tone in a shown sentence."
        return entire_prompt


class SRENode(ImitationChainNode):

    def summarize_trials(self, trials: list, experiment, participant):
        received_answers = []
        for trial in trials:
            received_answers.append(trial.answer)
        # selected_tone = max(set(received_answers), key=received_answers.count)  # This is similar to KDE peak picking
        selected_tone = random.choice(received_answers)
        self.definition["response"] = selected_tone
        return {
            "previous_sample": self.definition["response"],
            "current_mode": SREPage.page_mode_switch(self.definition["current_mode"])
        }


class SRENetwork(ImitationChainNetwork):
    pass


class SRE(ImitationChainTrialMaker):
    response_timeout_sec = 600
    check_timeout_interval_sec = 30

    def __init__(
        self,
        sentence_seed_path: str,
        tone_seed_path: str,
        num_chains: int = 10,
        n_trials_per_participant: int = 1,
        min_word_num_per_sentence: int = 5,
        *args,
        **kwargs
    ):
        self.num_chains = num_chains
        self.min_word_num_per_sentence = min_word_num_per_sentence
        self.sentence_list = self.read_seed_files(sentence_seed_path)
        self.tones_list = self.read_seed_files(tone_seed_path)
        kwargs["chain_type"] = "across"
        kwargs["recruit_mode"] = "n_trials"
        kwargs["network_class"] = SRENetwork
        kwargs["node_class"] = SRENode
        kwargs["trial_class"] = SRETrial
        kwargs["start_nodes"] = self.prepare_start_nodes()
        kwargs["balance_across_chains"] = False
        kwargs["check_performance_at_end"] = False
        kwargs["check_performance_every_trial"] = False
        kwargs["expected_trials_per_participant"] = n_trials_per_participant
        kwargs["max_trials_per_participant"] = n_trials_per_participant
        kwargs["chains_per_experiment"] = len(kwargs["start_nodes"])
        super().__init__(
            *args,
            **kwargs
        )

    def prepare_start_nodes(self):
        all_nodes = []
        num_sentence_seeds = self.num_chains // 2 + bool(self.num_chains % 2)
        seed_sentences = random.sample(self.sentence_list, k=num_sentence_seeds)
        seed_tones = random.sample(self.tones_list, k=self.num_chains - num_sentence_seeds)
        for _it in range(self.num_chains):
            if _it < self.num_chains * 0.5:
                starting_mode = "c"
                current_stimuli = seed_sentences.pop()
            else:
                starting_mode = "s"
                current_stimuli = seed_tones.pop()
            all_nodes.append(
                SRENode(
                    definition={
                        "previous_sample": {
                            "obtained_response": current_stimuli,
                            "current_mode": starting_mode
                        },
                        "current_mode": starting_mode
                    }
                )
            )
        return all_nodes

    @staticmethod
    def read_seed_files(seed_path):
        with open(seed_path, "r") as f:
            return list(map(lambda x: x.replace("\n", ""), f.readlines()))

    @staticmethod
    def instruction_page_before_practicing():
        return InfoPage(
            Markup(
                """
                Let's do some practices with the two following exercises!
                """
            ),
            time_estimate=15
        )

    @staticmethod
    def instruction_page_before_starting():
        return InfoPage(
            Markup(
                """
                We are going to start the experiment once you click "Next". <br>
                You can type your responses in the provided textbox once the experiment starts! <br>
                <b>
                    If you were asked to provide a couple of sentences and tones, please try to provide a different
                    answer every time.
                </b> <br>
                On any single page, if you linger for more than 300 seconds, our code will help you exit the experiment.
                Have a good time!
                """
            ),
            time_estimate=10
        )


class SREPractice(SRE):
    response_timeout_sec = 200
    check_timeout_interval_sec = 30

    def __init__(self, max_nodes_per_chain, *args, **kwargs):
        kwargs["id_"] = "SREPractice"
        kwargs["num_chains"] = 2
        kwargs["n_trials_per_participant"] = 2
        kwargs["max_nodes_per_chain"] = max_nodes_per_chain
        super().__init__(*args, **kwargs)


class SREDirectElicitTrial(StaticTrial):
    time_estimate = 10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.var.set("start_time", datetime.datetime.utcnow())

    def show_trial(self, experiment, participant):
        return SREPage(
            prompt=self.get_prompt(),
            current_mode="direct_elicit",
            min_word_num_per_sentence=5,
            page_trial=self
        )

    def get_prompt(self):
        preface_definition = """
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
            A <b>conversation tone</b> is the style and manner in which someone speaks.
            Sometimes, it is also referred to as the tone of a sentence. <br>
            When a sentence has a conversation tone, the speaker of a sentence has a similar attitude. <br><br>
        """
        entire_prompt = preface_definition
        if self.participant.var.get('answers'):
            entire_prompt += f"<br>You have currently provided the following conversation tones: {[ans['obtained_response'] for ans in self.participant.var.get('answers')]}, "
            entire_prompt += f"we would like you to provide {5 - len(self.participant.var.get('answers'))} more tones."
        entire_prompt += f"<br>In the following textbox, provide just one adjective for a conversation tone in English."
        return entire_prompt


class SREDirectElicitNode(StaticNode):

    def summarize_trials(self, trials: list, experiment, participant):
        received_answers = []
        for trial in trials:
            received_answers.append(trial.answer)
        # selected_tone = max(set(received_answers), key=received_answers.count)  # This is similar to KDE peak picking
        self.definition["response"] = received_answers
        db.session.commit()
        return received_answers


class SREDirectElicit(StaticTrialMaker):
    def prepare_trial(self, experiment, participant):
        trial, trial_status = super().prepare_trial(experiment, participant)
        try:
            participant.var.get("answers")
        except KeyError:
            participant.var.set("answers", [])
        return trial, trial_status

    def finalize_trial(self, answer, trial, experiment, participant):
        super().finalize_trial(answer, trial, experiment, participant)
        participant.var.set("answers", participant.var.get("answers") + [answer])
        db.session.commit()


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
                        Provide an <b>adjective</b> for a conversation tone in your language that you sense in
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
            time_estimate=30
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
            time_estimate=30
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
