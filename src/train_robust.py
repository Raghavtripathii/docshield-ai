from __future__ import annotations

from transformers import Trainer


class RobustTrainer:

    def __init__(
        self,
        model,
        train_dataset,
        eval_dataset,
        training_args,
    ):

        self.model = model

        self.train_dataset = train_dataset

        self.eval_dataset = eval_dataset

        self.training_args = training_args

    def build(self):

        return Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
        )

    def train(self):

        trainer = self.build()

        trainer.train()

        return trainer