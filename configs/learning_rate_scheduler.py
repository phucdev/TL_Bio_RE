from typing import Dict, Any

from overrides import overrides
import torch
from transformers.optimization import get_linear_schedule_with_warmup
from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import LearningRateScheduler


@LearningRateScheduler.register("linear_schedule_with_warmup")
class LinearScheduleWithWarmup(LearningRateScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int
    ) -> None:
        self.lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps,
                                                            num_training_steps=num_training_steps)
        super().__init__(optimizer)

    @overrides
    def step(self, metric: float = None, epoch: int = None) -> None:
        self.lrs.step(epoch)

    @overrides
    def state_dict(self) -> Dict[str, Any]:
        return self.lr_scheduler.state_dict()

    @overrides
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.lr_scheduler.load_state_dict(state_dict)
