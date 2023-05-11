# -*- coding:utf-8 _*-

from torch.optim.lr_scheduler import _LRScheduler


class PolynomialLRDecay(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps: The number of steps to linearly increase the learning rate.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: Final learning rate to decay towards.
        power: The power of the polynomial used for decaying.
    Example:
        import torch
        v = torch.zeros(10)
        optim = torch.optim.SGD([v], lr=5e-5)
        scheduler = PolynomialLRDecay(optim, warmup_steps=3000, max_decay_steps=30000,
                                      end_learning_rate=1e-5, power=2.0)

        for epoch in range(0, 30000):
            scheduler.step()

            if epoch%1000==0:
                print(epoch, optim.param_groups[0]['lr'])
    """

    def __init__(self, optimizer, warmup_steps, max_decay_steps, end_learning_rate=1e-5, power=2.0, steps=0):
        if max_decay_steps <= 1.:
            raise ValueError('the max-decay-steps should be greater than 1.')
        self.warmup_steps = warmup_steps
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.steps = steps
        super().__init__(optimizer)


    def update_lr(self):
        if self.warmup_steps > 0 and self.steps < self.warmup_steps:
            f = self.steps / self.warmup_steps
            return [f * lr for lr in self.base_lrs]

        if self.steps >= self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        cur_decay_steps = self.max_decay_steps - self.steps
        all_decay_steps = self.max_decay_steps - self.warmup_steps
        f = (cur_decay_steps / all_decay_steps) ** self.power
        return [
            f * (lr - self.end_learning_rate) + self.end_learning_rate for lr in self.base_lrs
        ]

    def step(self):
        self.steps += 1
        cur_lrs = self.update_lr()
        for param_group, lr in zip(self.optimizer.param_groups, cur_lrs):
            param_group['lr'] = lr

