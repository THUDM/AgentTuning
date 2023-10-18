import datasets
import pandas as pd


class DataLoader:
    def __init__(self, data="hotpot_qa", seed=2023):
        self.data = data
        self.seed = seed

    def load(self, sample_size=None, type="train"):
        if self.data == "hotpot_qa":
            return self.load_hotpot_qa(sample_size=sample_size, type=type)
        elif self.data == "fever":
            return self.load_fever(sample_size=sample_size, type=type)
        elif self.data == "trivia_qa":
            return self.load_trivia_qa(sample_size=sample_size, type=type)
        elif self.data == "gsm8k":
            return self.load_gsm8k(sample_size=sample_size, type=type)
        elif self.data == "physics_question":
            return self.load_physics_question(sample_size=sample_size)
        elif self.data == "disfl_qa":
            return self.load_disfl_qa(sample_size=sample_size)
        elif self.data == "sports_understanding":
            return self.load_sports_understanding(sample_size=sample_size)
        elif self.data == "strategy_qa":
            return self.load_strategy_qa(sample_size=sample_size)
        elif self.data == "sotu_qa":
            return self.load_sotu_qa(sample_size=sample_size)
        else:
            raise ValueError("Data not supported.")

    def load_hotpot_qa(self, cache_dir="data/hotpot_qa", sample_size=100, type="test"):
        assert type in ["train", "validation", "test"]
        data = datasets.load_dataset('hotpot_qa', 'fullwiki', cache_dir=cache_dir)
        df = data[type].to_pandas()
        sampled_df = df.sample(sample_size, random_state=self.seed)[["question", "answer"]].reset_index(drop=True)
        return sampled_df

    def load_fever(self, cache_dir="data/fever", sample_size=100, type="test"):
        assert type in ["train", "validation", "test"]
        data = datasets.load_dataset('copenlu/fever_gold_evidence', cache_dir=cache_dir)
        df = data[type].to_pandas()
        sampled_df = df.sample(sample_size, random_state=self.seed)[["claim", "label"]].reset_index(drop=True)
        return sampled_df

    def load_trivia_qa(self, cache_dir="data/trivia_qa", sample_size=100, type="test"):
        assert type in ["train", "validation", "test"]
        data = datasets.load_dataset('trivia_qa', 'rc.nocontext', cache_dir=cache_dir)
        df = data[type].to_pandas()
        sampled_df = df.sample(sample_size, random_state=self.seed)[["question", "answer"]].reset_index(drop=True)
        return sampled_df

    def load_gsm8k(self, cache_dir="data/gsm8k", sample_size=100, type="test"):
        assert type in ["train", "validation", "test"]
        data = datasets.load_dataset('gsm8k', name="main", cache_dir=cache_dir)
        df = data[type].to_pandas()
        sampled_df = df.sample(sample_size, random_state=self.seed)[["question", "answer"]].reset_index(drop=True)
        return sampled_df

    def load_physics_question(self, cache_dir="data/bigbench/physics_question.csv", sample_size=None):
        df = pd.read_csv(cache_dir)
        if sample_size is not None:
            sampled_df = df.sample(sample_size, random_state=self.seed)[["input", "target"]].reset_index(drop=True)
            return sampled_df
        return df

    def load_sports_understanding(self, cache_dir="data/bigbench/sports_understanding.csv", sample_size=None):
        df = pd.read_csv(cache_dir)
        if sample_size is not None:
            sampled_df = df.sample(sample_size, random_state=self.seed)[["input", "target"]].reset_index(drop=True)
            return sampled_df
        return df

    def load_disfl_qa(self, cache_dir="data/bigbench/disfl_qa.csv", sample_size=None):
        df = pd.read_csv(cache_dir)
        if sample_size is not None:
            sampled_df = df.sample(sample_size, random_state=self.seed)[["input", "target"]].reset_index(drop=True)
            return sampled_df
        return df

    def load_strategy_qa(self, cache_dir="data/bigbench/strategy_qa.csv", sample_size=None):
        df = pd.read_csv(cache_dir)
        if sample_size is not None:
            sampled_df = df.sample(sample_size, random_state=self.seed)[["input", "target"]].reset_index(drop=True)
            return sampled_df
        return df

    def load_sotu_qa(self, cache_dir="data/SOTU/SOTU_QA.csv", sample_size=None):
        df = pd.read_csv(cache_dir)
        if sample_size is not None:
            sampled_df = df.sample(sample_size, random_state=self.seed)[["question", "answer"]].reset_index(drop=True)
            return sampled_df
        return df