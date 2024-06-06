from credentials import CREDENTIALS

import time
import random

import numpy as np
import pandas as pd

import openai
openai.organization = CREDENTIALS["organization_id"]
openai.api_key = CREDENTIALS["API_key"]

TEMPERATURE = 0.8

class SREGPTChain:
    def __init__(
        self,
        num_nodes_per_chain: int,
        starting_mode: str,
        min_num_word_per_sentence: int,
        chain_id: int,
        GPT_model: str,
        initiation_mode: str,
        aggregation_fold: int,
        lang: str,
        sleep_interval: int = 1,
        initiation_config: dict = None
    ):
        assert starting_mode in ["c", "s"], \
            "starting_mode should be either c or s"
        assert initiation_mode in ["disjoint", "together"]
        if initiation_mode == "together":
            assert "sentences" in initiation_config and "tones" in initiation_config, \
                "initiation_config not in proposed format."
        self.initiation_mode = initiation_mode
        self.sre_nodes = [None]
        self.sleep_interval = sleep_interval
        self.starting_mode = starting_mode
        self.chain_id = chain_id
        self.aggregation_fold = aggregation_fold
        self.lang = lang
        cur_node_mode = starting_mode
        for _it in range(num_nodes_per_chain):
            cur_node = SREGPTNode(
                mode = cur_node_mode,
                order = _it,
                chain_id = self.chain_id,
                min_num_word_per_sentence = min_num_word_per_sentence,
                GPT_model = GPT_model,
                prev_node = self.sre_nodes[_it],
                lang = self.lang
            )
            if initiation_mode == "together" and _it == 0:
                if cur_node_mode == "s":
                    cur_node.responses.append(initiation_config["sentences"].pop())
                elif cur_node_mode == "c":
                    cur_node.responses.append(initiation_config["tones"].pop())
                cur_node.prompt = "elicited in prior"
            self.sre_nodes.append(cur_node)
            cur_node_mode = SREGPTNode.node_mode_switch(cur_node_mode)
        self.sre_nodes.pop(0)
    
    def run_chain(self, result_address):
        chain_log = {
            "chain_id": self.chain_id,
            "chain_responses": {}
        }
        if self.chain_id % 2 == 0:
            return print("chain id mod 2 is 0")
        for cur_node in self.sre_nodes:
            if self.initiation_mode == "together" and cur_node.order == 0:
                chain_log["chain_responses"][cur_node.order] = {
                    "mode": cur_node.mode,
                    "response": cur_node.responses
                }
            else:
                if self.sleep_interval:
                    print(
                        f"On the running for chain {self.chain_id} node_id {cur_node.order}," +
                        f"we decided to sleep {self.sleep_interval} seconds.\n" +
                        f"許されよ 許されよ~"
                    )
                    time.sleep(self.sleep_interval)
                for _it in range(self.aggregation_fold):
                    cur_node.run_node_response()
                chain_log["chain_responses"][cur_node.order] = {
                    "mode": cur_node.mode,
                    "response": cur_node.responses
                }
            cur_node.record_results(result_address)
        return chain_log


class SREGPTNode:
    def __init__(
            self,
            mode: str,
            order: int,
            chain_id: int,
            min_num_word_per_sentence: int,
            GPT_model: str,
            lang: str,
            prev_node = None,
        ):
        self.mode = mode
        self.order = order
        self.chain_id = chain_id
        self.min_num_word_per_sentence = min_num_word_per_sentence
        self.responses = []
        self.GPT_model = GPT_model
        self.prev_node = prev_node
        self.lang = lang
        self.sleep_interval = 3
    
    def record_results(self, result_address):
        for response_order, response in enumerate(self.responses):
            with open(result_address, "a", encoding="utf8") as f:
                f.write(
                    f"{self.chain_id}|{self.order}|{response_order}|" +
                    f"{self.mode}|{self.prompt}|{response}\n"
                )
    
    def get_response(self):
        self.prompt = self.get_prompt(self.lang)
        if self.GPT_model is None:
            return f"This is an automatic response {random.random()} at language {self.lang}"
        else:
            try:
                API_response = openai.ChatCompletion.create(
                    model="gpt-4-0613",
                    messages=[
                        {"role": "user", "content": self.prompt}
                    ],
                    temperature=TEMPERATURE
                )
                return API_response.choices[0]["message"]["content"]
            except:
                return self.get_response()
    
    def get_prompt(self, lang):
        if lang == "en":
            if self.prev_node is None:
                prior_response = None
            else:
                prior_response = random.choice(self.prev_node.responses)
            response_prompt = ""
            if self.mode == "c":
                response_prompt = f"Provide an adjective for conversation tone in English"
                if prior_response:
                    response_prompt += f" that you sense in the following sentence: {prior_response}."
                    response_prompt += f"Respond using only an adjective"
            elif self.mode == "s":
                response_prompt = f"Provide one sentence with at least five words in English"
                if prior_response:
                    response_prompt += f" that has the conversation tone: {prior_response}"
            return response_prompt + "."
        elif lang == "zh":
            if self.prev_node is None:
                prior_response = None
            else:
                prior_response = random.choice(self.prev_node.responses)
            response_prompt = ""
            if self.mode == "c":
                response_prompt = f"請依下列句子提供一個描述該句子中語氣的形容詞: '{prior_response}'"
                response_prompt += f"請只以一個中文詞作答。"
            elif self.mode == "s":
                response_prompt = f"請撰寫一個函五個字以上，具'{prior_response}'的語氣的句子。"
            return response_prompt + "."
    
    def run_node_response(self):
        self.responses.append(self.get_response())
        return self.responses[-1]

    @staticmethod
    def node_mode_switch(mode):
        return "c" if mode == "s" else "s"


class SREExperiment:
    def __init__(
        self,
        num_chains: int,
        num_nodes_per_chain: int,
        min_num_word_per_sentence: int,
        GPT_model: str,
        result_address: str,
        initiation_mode: str,
        aggregation_fold: int,
        sleep_interval: int = 1,
        initiation_config: dict = None,
        language: str = "en"
    ):
        
        self.result_address = result_address
        self.num_chains = num_chains
        self.sre_chains = []
        self.num_nodes_per_chain = num_nodes_per_chain
        self.min_num_word_per_sentence = min_num_word_per_sentence
        self.GPT_model = GPT_model
        self.result_address
        cur_start_mode = "c"
        for _it in range(num_chains):
            self.sre_chains.append(
                SREGPTChain(
                    num_nodes_per_chain=num_nodes_per_chain,
                    starting_mode=cur_start_mode,
                    min_num_word_per_sentence=min_num_word_per_sentence,
                    chain_id = _it,
                    GPT_model = GPT_model,
                    initiation_mode = initiation_mode,
                    initiation_config = initiation_config,
                    aggregation_fold = aggregation_fold,
                    sleep_interval = sleep_interval,
                    lang = language
                )
            )
            cur_start_mode = SREGPTNode.node_mode_switch(cur_start_mode)
    
    def run_experiment(self):
        experiment_log = {
            "attr": {
                "num_chains": self.num_chains,
                "num_nodes_per_chain": self.num_nodes_per_chain,
                "min_num_word_per_sentence": self.min_num_word_per_sentence,
                "GPT_model": self.GPT_model
            },
            "experiment_responses": {}
        }
        with open(self.result_address, "w", encoding="utf8") as f:
            f.write("chain_id|node_order|response_order|node_mode|node_prompt|node_response\n")
        for cur_chain in self.sre_chains:
            experiment_log[
                "experiment_responses"
            ][cur_chain.chain_id] = cur_chain.run_chain(self.result_address)
        return experiment_log
    
    @staticmethod
    def extract_initiation_points(mode, num_responses, initiation_log_address):
        assert mode in ["c", "s"], "mode should be either 'c' or 's'."
        init_response = ""
        if mode == "c":
            init_prompt = SREExperiment.INITIATION_PROMPT_TONES
        elif mode == "s":
            init_prompt = SREExperiment.INITIATION_PROMPT_SENTENCES
        init_prompt = init_prompt.replace("|", str(num_responses))
        init_response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=[
                {"role": "user", "content": init_prompt}
            ],
            temperature=1.6
        )
        print(init_response)
        with open(initiation_log_address, "w", encoding="utf8") as f:
            f.write(init_response.choices[0]["message"]["content"])
    
    INITIATION_PROMPT_SENTENCES = """
        Provide | different sentences with at least five words in English.
        Make sentences have different conversation tones.
        Respond with one sentence per line.
    """
    INITIATION_PROMPT_TONES = """
        Provide | different adjectives for conversation tone in English.
        Respond with one word per line.
    """


class SREFeatureExtractionInstace:
    def __init__(
        self,
        extraction_path,
        result_address,
        feature_names,
        GPT_model,
        sleep_interval,
        num_raters=1
    ):
        self.extraction_path = extraction_path
        self.result_address = result_address
        self.feature_names = feature_names
        self.GPT_model = GPT_model
        self.num_raters = num_raters
        self.sleep_interval = sleep_interval
        self.extracted_df = pd.read_csv(extraction_path, sep="|")
    
    @property
    def tone_list(self):
        return np.unique(
            self.extracted_df.query("item_type == 'tone'")["item_content"]
        )
    
    def get_one_combination_prompt(self, tone, feature, explanation):
        return f"""
            A conversation tone is the style and manner in which someone speaks.
            {explanation} 
            On a scale of 1 to 5, where 5 means strongest and 1 means weakest,
            how {feature} is the conversation tone "{tone}"?
            Respond with only a number.
        """
    
    def rate_one_combination(self, tone, feature, explanation):
        rating_prompt = self.get_one_combination_prompt(tone, feature, explanation)
        if self.GPT_model is None:
            return rating_prompt
        else:
            # print(
            #     f"we decided to sleep {self.sleep_interval} seconds.\n" +
            #     f"許されよ 許されよ~"
            # )
            time.sleep(self.sleep_interval)
            API_response = openai.ChatCompletion.create(
                model="gpt-4-0613",
                messages=[
                    {"role": "user", "content": rating_prompt}
                ],
                temperature=TEMPERATURE
            )
            return API_response.choices[0]["message"]["content"]
    
    def run_feature_instance(self):
        with open(self.result_address, "a", encoding="utf8") as f:
            f.write(
                "feature|tone|rater_order|current_rating\n"
            )
        for feature in self.feature_names:
            for tone in self.tone_list:
                for rater in range(self.num_raters):
                    current_rating = self.rate_one_combination(tone, feature, self.feature_names[feature])
                    with open(self.result_address, "a", encoding="utf8") as f:
                        f.write(
                            f"{feature}|{tone}|{rater}|{current_rating}\n"
                        )


class SRERatingInstance:
    def __init__(
        self,
        extraction_path,
        result_address,
        num_raters,
        locale,
        sleep_interval,
        top_k_tones=-1,
        GPT_model=None
    ):
        self.extraction_path = extraction_path
        self.result_address = result_address
        self.locale = locale
        self.num_raters = num_raters
        self.top_k_tones = top_k_tones
        self.sleep_interval = sleep_interval
        self.GPT_model = GPT_model
        self.extracted_df = pd.read_csv(extraction_path, sep="|")

    @property
    def sentence_list(self):
        return np.unique(
            self.extracted_df.query("item_type == 'sentence'")["item_content"]
        )
    
    @property
    def tone_list(self):
        return np.unique(
            self.extracted_df.query("item_type == 'tone'")["item_content"]
        )
    
    def get_one_combination_prompt(self, sentence, tone):
        return f"""
            A conversation tone is the style and manner in which someone speaks.
            On a scale of 1 to 5, with 5 being strongest, how strong is the provided conversation tone in the following English sentence?
            Tone: {tone}
            Sentence: {sentence}
            Respond with only a number.
        """
    
    def rate_one_combination(self, sentence, tone):
        rating_prompt = self.get_one_combination_prompt(sentence, tone)
        if self.GPT_model is None:
            print(len(self.sentence_list), len(self.tone_list))
            return f"This is an automatic response for {sentence, tone}"
        else:
            # print(
            #     f"we decided to sleep {self.sleep_interval} seconds.\n" +
            #     f"許されよ 許されよ~"
            # )
            time.sleep(self.sleep_interval)
            try:
                API_response = openai.ChatCompletion.create(
                    model="gpt-4-0613",
                    messages=[
                        {"role": "user", "content": rating_prompt}
                    ],
                    temperature=TEMPERATURE
                )
                return API_response.choices[0]["message"]["content"]
            except:
                return self.rate_one_combination(sentence, tone)
    
    def run_rating_instance(self):
        with open(self.result_address, "a", encoding="utf8") as f:
            f.write(
                "sentence|tone|rater_order|current_rating\n"
            )
        for sentence in self.sentence_list:
            for tone in self.tone_list:
                for rater in range(self.num_raters):
                    current_rating = self.rate_one_combination(sentence, tone)
                    with open(self.result_address, "a", encoding="utf8") as f:
                        f.write(
                            f"{sentence}|{tone}|{rater}|{current_rating}\n"
                        )


class SRESimilarityInstance:
    def __init__(
        self,
        gpt_extraction_path,
        human_extraction_path,
        result_address,
        locale,
        sleep_interval,
        is_cross_species=False,
        GPT_model=None
    ):
        self.result_address = result_address
        self.locale = locale
        self.sleep_interval = sleep_interval
        self.GPT_model = GPT_model
        self.is_cross_species = is_cross_species
        self.gpt_extracted_df = pd.read_csv(gpt_extraction_path, sep="|")
        self.human_extracted_df = pd.read_csv(human_extraction_path, sep="|")
    
    @property
    def gpt_tone_list(self):
        return np.unique(
            self.gpt_extracted_df.query("item_type == 'tone'")["item_content"]
        )
    
    @property
    def human_tone_list(self):
        return np.unique(
            self.human_extracted_df.query("item_type == 'tone'")["item_content"]
        )
        
    def decide_prompt(self, tone_a, tone_b):
        return f"""
        A conversation tone is is the style and manner in which someone speaks, and sometimes, it is also referred to as the tone of a sentence.
        How similar are the conversation tones in each pair on a scale of 0-1 where 0 is completely dissimilar and 1 is completely similar?

        Conversation Tone 1: {tone_a}
        Conversation Tone 2: {tone_b}

        Respond only with the numerical similarity rating.
        """
    
    def rate_one_combination(self, tone_a, tone_b):
        rating_prompt = self.decide_prompt(tone_a, tone_b)
        if self.GPT_model is None:
            print(len(self.sentence_list), len(self.tone_list))
            return f"This is an automatic response for {tone_a, tone_b}"
        else:
            # print(
            #     f"we decided to sleep {self.sleep_interval} seconds.\n" +
            #     f"許されよ 許されよ~"
            # )
            time.sleep(self.sleep_interval)
            try:
                API_response = openai.ChatCompletion.create(
                    model="gpt-4-0613",
                    messages=[
                        {"role": "user", "content": rating_prompt}
                    ],
                    temperature=TEMPERATURE
                )
                return API_response.choices[0]["message"]["content"]
            except:
                return self.rate_one_combination(tone_a, tone_b)
    
    def run_rating_instance(self):
        with open(self.result_address, "a", encoding="utf8") as f:
            f.write(
                "tone_human|tone_gpt|rater_order|current_rating\n"
            )
        performed_combinations = {t: [] for t in self.human_tone_list}
        for tone_a in self.human_tone_list:
            for tone_b in self.gpt_tone_list:
                if not self.is_cross_species and (tone_b in performed_combinations[tone_a] or tone_a in performed_combinations[tone_b]):
                    continue
                for i in range(5):
                    current_rating = self.rate_one_combination(tone_a, tone_b)
                    with open(self.result_address, "a", encoding="utf8") as f:
                        f.write(
                            f"{tone_a}|{tone_b}|{i}|{current_rating}\n"
                        )
                    performed_combinations[tone_a].append(tone_b)
