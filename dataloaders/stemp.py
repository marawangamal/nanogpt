import random
from dataloaders.base import AbstractDataset

from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer


class DataIterator:
    def __init__(self, num_samples, split="train", shift="in"):
        self.num_samples = num_samples
        self.split = split
        self.shift = shift

    @staticmethod
    def _clip_temp(temp: int, min_val: int = -40, max_val: int = 80) -> int:
        return max(min_val, min(max_val, temp))

    def _sample_celsius(self):
        return self._clip_temp(int(random.gauss(20, 2)))

    def __iter__(self):
        for _ in range(self.num_samples):
            temp_c = self._sample_celsius()
            temp_f = (temp_c * 9 / 5) + 32
            question = f"What is {temp_c}°C in Fahrenheit?"
            response = (
                "\nLet's solve this step by step:\n"
                "1) To convert Celsius to Fahrenheit, use the formula: °F = (°C x 9/5) + 32\n"
                f"2) Plugging in {temp_c}°C:\n   °F = ({temp_c} x 9/5) + 32\n   °F = {temp_f}\n\n####\n{temp_f}"
            )
            yield {"question": question, "answer": response}


class STemp(AbstractDataset):
    template = """[QUESTION]\n{question}\n[ANSWER]{answer}"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def parse_answer(self, generation: str) -> float:
        try:
            return (
                float(
                    generation.split("####")[-1]
                    .split(self.eos)[0]
                    .strip()
                    .split(" ")[0]
                    .split("\n")[0]
                )
                if "####" in generation
                else float("nan")
            )
        except Exception as e:
            return float("nan")

    @classmethod
    def get_sample_prompt(cls) -> str:
        return cls.template.format(
            question="What is 20°C in Fahrenheit?",
            answer="",
        )

    def format_train_example(self, example):
        return (
            self.template.format(
                question=example["question"],
                answer=example["answer"],
            )
            + "\n"
            + self.eos
        )

    def format_test_example(self, example):
        return self.template.format(
            question=example["question"],
            answer="",
        )

    def format_test_label(self, example):
        return self.parse_answer(example["answer"])

    def load_raw_data(self):
        num_train_samples = 10000
        num_test_samples = 100
        base_datasets = {
            "train": Dataset.from_generator(
                lambda: DataIterator(
                    num_train_samples, split="train", shift=self.domain_shift
                )
            ),
            "eval": Dataset.from_generator(
                lambda: DataIterator(
                    num_test_samples, split="eval", shift=self.domain_shift
                )
            ),
        }
        test_datasets = {
            "test": Dataset.from_generator(
                lambda: DataIterator(
                    num_test_samples, split="test", shift=self.domain_shift
                )
            )
        }

        for split in base_datasets:
            base_datasets[split] = self._process_train_dataset(
                base_datasets[split], self.tokenizer
            )
        for split in test_datasets:
            test_datasets[split] = self._process_test_dataset(
                test_datasets[split], self.tokenizer
            )

        return DatasetDict({**base_datasets, **test_datasets})


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    stemp = STemp(tokenizer, max_num_samples=100).load_data()

    # Print a sample from the dataset
    print("Train example:")
    print(tokenizer.decode(stemp["train"][0]["input_ids"]))

    print("Test example (input):")
    print(tokenizer.decode(stemp["test"][0]["input_ids"]))

    print("Test example (label):")
    print(stemp["test"][0]["labels"])
