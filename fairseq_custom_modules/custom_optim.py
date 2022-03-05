
import math
import torch
import torch.optim

from fairseq.optim import FairseqOptimizer, register_optimizer


@register_optimizer('custom_sgd')
class CustomSGD(FairseqOptimizer):
    def __init__(self, args, named_params):
        super().__init__(args, [param for name, param in named_params])
        self._optimizer = torch.optim.SGD([param for name, param in named_params], **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--momentum', default=0.0, type=float, metavar='M',
                            help='momentum factor')
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')
        # fmt: on

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            'lr': self.args.lr[0],
            'momentum': self.args.momentum,
            'weight_decay': self.args.weight_decay,
        }


@register_optimizer('custom_adagrad')
class CustomAdagrad(FairseqOptimizer):
    def __init__(self, args, named_params):
        super().__init__(args, [param for name, param in named_params])
        self.named_params = named_params
        self._optimizer = torch.optim.Adagrad(self.param_groups, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        pass

    @property
    def param_groups(self):
        # return [param for name, param in self.named_params]
        flinear_params = []
        other_params = []
        for name, param in self.named_params:
            if 'encoder' in name and 'fc_out' in name:
                flinear_params.append(param)
            else:
                other_params.append(param)
        groups = []
        groups.append({'params':flinear_params, 'lr':self.args.lr[0]})
        groups.append({'params':other_params, 'lr':self.args.lr[1]})
        return groups

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            'lr': self.args.lr[0],
        }

