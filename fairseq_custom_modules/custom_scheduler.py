
from fairseq.optim import FairseqOptimizer
from fairseq.optim.lr_scheduler import FairseqLRScheduler, register_lr_scheduler

@register_lr_scheduler('custom_scheduler')
class CustomLRSchedule(FairseqLRScheduler):
    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)


@register_lr_scheduler('custom_anneal_scheduler')
class CustomAnnealLRSchedule(FairseqLRScheduler):
    """Decay the LR on a fixed schedule."""

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)

        # set defaults
        self.lr = args.lr[0]
        self.shrink_factor = args.shrink_factor
        self.anneal_start_after = args.anneal_start_after

        self.optimizer.set_lr(self.lr)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        parser.add_argument('--shrink-factor', default=0.5, type=float, metavar='LS', help='shrink factor for annealing, lr_new = (lr * lr_shrink)')
        parser.add_argument('--anneal-start-after', default=0, type=int, metavar='N', help='keep learning rate the same for the first N epoch')

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        if epoch >= self.anneal_start_after:
            self.optimizer.set_lr(self.shrink_factor * self.optimizer.get_lr())
        return self.optimizer.get_lr()

