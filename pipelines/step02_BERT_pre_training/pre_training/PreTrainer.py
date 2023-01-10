import torch
from datasets import Dataset
from transformers import Trainer, PreTrainedModel, TrainingArguments, DataCollator, PreTrainedTokenizerBase, \
    EvalPrediction, TrainerCallback, IntervalStrategy
import numpy as np

from typing import Dict, Union, Optional, Callable, List, Tuple
from torch import nn


class PreTrainer(Trainer):

    "A subclass of the Trainer class. This subclass was made so nsp and mlm losses could be displayed."

    def __init__(self, model: Union[PreTrainedModel, nn.Module] = None, args: TrainingArguments = None,
                 data_collator: Optional[DataCollator] = None, train_dataset: Optional[Dataset] = None,
                 eval_dataset: Optional[Dataset] = None, tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 model_init: Callable[[], PreTrainedModel] = None,
                 compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
                 callbacks: Optional[List[TrainerCallback]] = None,
                 optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None)):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init,
                         compute_metrics,
                         callbacks, optimizers)

        self.nsp_losses = []
        self.mlm_losses = []
        if self.args is not None and self.args.logging_steps is not None and self.args.logging_strategy == IntervalStrategy.STEPS:
            self.nsp_losses = [0] * self.args.logging_steps
            self.mlm_losses = [0] * self.args.logging_steps

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Modified to save nsp and mlm losses
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        if self.args.logging_strategy == IntervalStrategy.STEPS and self.args.logging_steps is not None:
            self.nsp_losses[self.state.global_step % self.args.logging_steps] = outputs[-1].cpu().detach().numpy()
            self.mlm_losses[self.state.global_step % self.args.logging_steps] = outputs[-2].cpu().detach().numpy()
        else:
            self.nsp_losses.append(outputs[-1].cpu().detach().numpy())
            self.mlm_losses.append(outputs[-2].cpu().detach().numpy())

        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log :obj:`logs` on the various objects watching training.

        Modified to save mlm and nsp losses

        Args:
            logs (:obj:`Dict[str, float]`):
                The values to log.
        """

        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        output["nsp_loss"] = float(np.mean(self.nsp_losses))
        output["mlm_loss"] = float(np.mean(self.mlm_losses))

        logs["nsp_loss"] = output["nsp_loss"]
        logs["mlm_loss"] = output["mlm_loss"]
        logs["step"] = output["step"]
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

        if self.args.logging_strategy != IntervalStrategy.STEPS:
            self.nsp_losses = []
            self.mlm_losses = []
