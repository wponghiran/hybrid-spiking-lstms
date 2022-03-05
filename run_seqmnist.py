
import argparse
import math
import sys
import os
import pathlib
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import datasets, transforms

class CustomBias(nn.Module):
    def __init__(self, size):
        super(CustomBias, self).__init__()
        self.bias = nn.Parameter(torch.empty(size))
    def forward(self, inp):
        outp = inp + self.bias
        return outp

class SpikeGen(nn.Module):
    def __init__(self, soft_reset, vTh=2048):
        super(SpikeGen, self).__init__()
        self.soft_reset = soft_reset
        self.vTh = vTh
    def forward(self, inp, vMem):
        vMem += torch.abs(inp)
        is_spike = torch.gt(vMem, self.vTh)
        if self.soft_reset:
            vMem[is_spike] -= self.vTh
        else:
            vMem[is_spike] = 0
        return is_spike.float()*torch.sign(inp), vMem

class PiecewiseLinearType1F(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        outp = ((inp+2.0)/4.0).clamp(min=0.0, max=1.0)
        return outp
    @staticmethod
    def backward(ctx, grad_outp):
        inp, = ctx.saved_tensors
        grad_inp = (grad_outp.clone())/4.0
        grad_inp[inp < -2.0] = 0.0
        grad_inp[inp > 2.0] = 0.0
        return grad_inp
class PiecewiseLinearType1(nn.Module):
    def __init__(self):
        super(PiecewiseLinearType1, self).__init__()
    def forward(self, inp):
        return PiecewiseLinearType1F.apply(inp)
class IFNeuronType1(nn.Module):
    def __init__(self, size, soft_reset, vTh=1024):
        super(IFNeuronType1, self).__init__()
        self.soft_reset = soft_reset
        self.bias = nn.Parameter(torch.zeros(size))
        self.vTh = vTh
    def forward(self, inp, vMem):
        batch_size = inp.size(0)
        vMem += inp+self.bias.unsqueeze(0).repeat(batch_size, 1)
        is_spike = torch.gt(vMem, self.vTh)
        if self.soft_reset:
            vMem[is_spike] -= self.vTh
        else:
            vMem[is_spike] = 0
        return is_spike.float(), vMem

class PiecewiseLinearType2F(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        outp = (inp/2.0).clamp(min=-1.0, max=1.0)
        return outp
    @staticmethod
    def backward(ctx, grad_outp):
        inp, = ctx.saved_tensors
        grad_inp = (grad_outp.clone())/2.0
        grad_inp[inp < -2.0] = 0.0
        grad_inp[inp > 2.0] = 0.0
        return grad_inp
class PiecewiseLinearType2(nn.Module):
    def __init__(self):
        super(PiecewiseLinearType2, self).__init__()
    def forward(self, inp):
        return PiecewiseLinearType2F.apply(inp)
class IFNeuronType2(nn.Module):
    def __init__(self, size, soft_reset, vTh=512):
        super(IFNeuronType2, self).__init__()
        self.soft_reset = soft_reset
        self.bias = nn.Parameter(torch.zeros(size))
        self.vTh = vTh
    def forward(self, inp, vMem_pos, vMem_neg):
        batch_size = inp.size(0)
        vMem_pos += (inp+self.bias.unsqueeze(0).repeat(batch_size, 1))
        vMem_neg -= (inp+self.bias.unsqueeze(0).repeat(batch_size, 1))
        is_spike_pos = torch.gt(vMem_pos, self.vTh)
        is_spike_neg = torch.gt(vMem_neg, self.vTh)
        if self.soft_reset:
            vMem_pos[is_spike_pos] -= self.vTh
            vMem_neg[is_spike_pos] += self.vTh
            vMem_pos[is_spike_neg] += self.vTh
            vMem_neg[is_spike_neg] -= self.vTh
        else:
            vMem_pos[is_spike_pos] = 0 
            vMem_neg[is_spike_pos] = 0 
            vMem_pos[is_spike_neg] = 0 
            vMem_neg[is_spike_neg] = 0 
        return is_spike_pos.float()-is_spike_neg.float(), vMem_pos, vMem_neg

class SequentialModel(nn.Module):
    def __init__(self,
            lstm_cell_class,
            linear_class,
            num_timestep,
            input_size,
            num_layer, 
            hidden_size,
            dropout_prob,
            output_size,
        ):
        super(SequentialModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        # Recurrent layer
        self.list_lstm_cell = nn.ModuleList([lstm_cell_class(input_size=input_size, hidden_size=hidden_size, num_timestep=num_timestep) if layer==0 \
                else lstm_cell_class(input_size=hidden_size, hidden_size=hidden_size, num_timestep=num_timestep) for layer in range(self.num_layer)])
        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)
        # Create classifier
        self.classifier = linear_class(input_size=hidden_size, output_size=output_size, num_timestep=num_timestep)
    def forward(self, inps):
        batch_size = inps.size(0)
        seq_len = inps.size(1)
        inps = inps.transpose(0,1)    # (B,T,C) -> (T,B,C)
        finalt_hs = [] 
        finalt_cs = []
        allt_lstm_inps = inps
        # Process inputs in a layer-wise manner
        for layer in range(self.num_layer):
            h = torch.zeros(batch_size, self.hidden_size).cuda()
            c = torch.zeros(batch_size, self.hidden_size).cuda()
            # Process input in each timestep
            allt_lstm_outps = [] 
            for timestep in range(seq_len):
                next_h, next_c = self.list_lstm_cell[layer](allt_lstm_inps[timestep], (h,c))
                next_c = torch.clamp(next_c,min=-1,max=1)   # clamp to avoid overflow on Loihi
                allt_lstm_outps.append(next_h)
                h,c = next_h, next_c
            allt_lstm_outps = self.dropout(torch.stack(allt_lstm_outps, 0)) # (T, B, H)
            allt_lstm_inps = allt_lstm_outps
            finalt_hs.append(h)
            finalt_cs.append(c)
        finalt_hs = torch.stack(finalt_hs, 0)   # (L, B, H)
        finalt_cs = torch.stack(finalt_cs, 0)   # (L, B, H)
        outp = self.classifier(finalt_hs[-1])
        return outp

class BaselineLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_timestep):
        super(BaselineLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(input_size,  4*hidden_size, bias=False)
        self.linear2 = nn.Linear(hidden_size, 4*hidden_size, bias=False)
        self.bias = CustomBias(4*hidden_size)
        self.igate = nn.Sigmoid()
        self.fgate = nn.Sigmoid()
        self.cgate = nn.Tanh()
        self.ogate = nn.Sigmoid()
        self.act = nn.Tanh()
        self.reset_parameters(input_size, hidden_size)
    def reset_parameters(self, input_size, hidden_size):
        wi_i = torch.empty(hidden_size, input_size);   nn.init.orthogonal_(wi_i, gain=1.)
        wi_f = torch.empty(hidden_size, input_size);   nn.init.orthogonal_(wi_f, gain=1.)
        wi_c = torch.empty(hidden_size, input_size);   nn.init.orthogonal_(wi_c, gain=1.)
        wi_o = torch.empty(hidden_size, input_size);   nn.init.orthogonal_(wi_o, gain=1.)
        self.linear1.weight.data.copy_(torch.cat((wi_i,wi_f,wi_c,wi_o), dim=0))
        wh_i = torch.empty(hidden_size, hidden_size);   nn.init.orthogonal_(wh_i, gain=1.) 
        wh_f = torch.empty(hidden_size, hidden_size);   nn.init.orthogonal_(wh_f, gain=1.)
        wh_c = torch.empty(hidden_size, hidden_size);   nn.init.orthogonal_(wh_c, gain=1.)
        wh_o = torch.empty(hidden_size, hidden_size);   nn.init.orthogonal_(wh_o, gain=1.) 
        self.linear2.weight.data.copy_(torch.cat((wh_i,wh_f,wh_c,wh_o), dim=0))
        b_i = torch.zeros(hidden_size)
        b_f = torch.ones(hidden_size)
        b_c = torch.zeros(hidden_size)
        b_o = torch.zeros(hidden_size)
        self.bias.bias.data.copy_(torch.cat((b_i, b_f,b_c,b_o), dim=0))
    def forward(self, inp, hx):
        h, c = hx
        gate_inp = self.bias(self.linear1(inp) + self.linear2(h))
        igate_inp, fgate_inp, cgate_inp, ogate_inp = gate_inp.chunk(4,1)
        igate_outp = self.igate(igate_inp) 
        fgate_outp = self.fgate(fgate_inp) 
        cgate_outp = self.cgate(cgate_inp) 
        ogate_outp = self.ogate(ogate_inp)
        c_new = (fgate_outp*c) + (igate_outp*cgate_outp)
        h_new = ogate_outp*self.act(c_new)
        return (h_new, c_new)

class ModifiedLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_timestep):
        super(ModifiedLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(input_size,  4*hidden_size, bias=False)
        self.linear2 = nn.Linear(hidden_size, 4*hidden_size, bias=False)
        self.bias = CustomBias(4*hidden_size)
        self.igate = PiecewiseLinearType1()
        self.fgate = PiecewiseLinearType1()
        self.cgate = PiecewiseLinearType2()
        self.ogate = PiecewiseLinearType1()
        self.act = PiecewiseLinearType2()
        self.reset_parameters(input_size, hidden_size)
    def reset_parameters(self, input_size, hidden_size):
        wi_i = torch.empty(hidden_size, input_size);   nn.init.orthogonal_(wi_i, gain=1.)
        wi_f = torch.empty(hidden_size, input_size);   nn.init.orthogonal_(wi_f, gain=1.)
        wi_c = torch.empty(hidden_size, input_size);   nn.init.orthogonal_(wi_c, gain=1.)
        wi_o = torch.empty(hidden_size, input_size);   nn.init.orthogonal_(wi_o, gain=1.)
        self.linear1.weight.data.copy_(torch.cat((wi_i,wi_f,wi_c,wi_o), dim=0))
        wh_i = torch.empty(hidden_size, hidden_size);   nn.init.orthogonal_(wh_i, gain=1.) 
        wh_f = torch.empty(hidden_size, hidden_size);   nn.init.orthogonal_(wh_f, gain=1.)
        wh_c = torch.empty(hidden_size, hidden_size);   nn.init.orthogonal_(wh_c, gain=1.)
        wh_o = torch.empty(hidden_size, hidden_size);   nn.init.orthogonal_(wh_o, gain=1.) 
        self.linear2.weight.data.copy_(torch.cat((wh_i,wh_f,wh_c,wh_o), dim=0))
        b_i = torch.zeros(hidden_size)
        b_f = torch.ones(hidden_size)
        b_c = torch.zeros(hidden_size)
        b_o = torch.zeros(hidden_size)
        self.bias.bias.data.copy_(torch.cat((b_i, b_f,b_c,b_o), dim=0))
    def forward(self, inp, hx):
        h, c = hx
        gate_inp = self.bias(self.linear1(inp) + self.linear2(h))
        igate_inp, fgate_inp, cgate_inp, ogate_inp = gate_inp.chunk(4,1)
        igate_outp = self.igate(igate_inp) 
        fgate_outp = self.fgate(fgate_inp) 
        cgate_outp = self.cgate(cgate_inp) 
        ogate_outp = self.ogate(ogate_inp)
        c_new = (fgate_outp*c) + (igate_outp*cgate_outp)
        h_new = ogate_outp*self.act(c_new)
        return (h_new, c_new)

class HybridLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_timestep):
        super(HybridLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_timestep = num_timestep
        self.spikegen = SpikeGen(soft_reset=True)
        self.linear_igate = nn.Linear(input_size+hidden_size, hidden_size, bias=False)
        self.linear_fgate = nn.Linear(input_size+hidden_size, hidden_size, bias=False)
        self.linear_cgate = nn.Linear(input_size+hidden_size, hidden_size, bias=False)
        self.linear_ogate = nn.Linear(input_size+hidden_size, hidden_size, bias=False)
        self.igate = IFNeuronType1(hidden_size, soft_reset=True)
        self.fgate = IFNeuronType1(hidden_size, soft_reset=True)
        self.cgate = IFNeuronType2(hidden_size, soft_reset=True)
        self.ogate = IFNeuronType1(hidden_size, soft_reset=True)
        self.act = PiecewiseLinearType2()
    def forward(self, inp, hx):
        h, c = hx 
        batch_size = inp.size(0)
        sum_linear_inp = torch.zeros(batch_size, self.input_size+self.hidden_size).cuda()
        sum_igate_outp = torch.zeros(batch_size, self.hidden_size).cuda()
        sum_fgate_outp = torch.zeros(batch_size, self.hidden_size).cuda()
        sum_cgate_outp = torch.zeros(batch_size, self.hidden_size).cuda()
        sum_ogate_outp = torch.zeros(batch_size, self.hidden_size).cuda()
        spikegen_vmem = torch.ones(batch_size, self.input_size+self.hidden_size).cuda()
        igate_vmem = torch.ones(batch_size, self.hidden_size).cuda()
        fgate_vmem = torch.ones(batch_size, self.hidden_size).cuda()
        cgate_vmem_pos = torch.ones(batch_size, self.hidden_size).cuda()
        cgate_vmem_neg = torch.ones(batch_size, self.hidden_size).cuda()
        ogate_vmem = torch.ones(batch_size, self.hidden_size).cuda()
        for t in range(self.num_timestep):
            linear_inp, spikegen_vmem = self.spikegen(torch.round(2048*torch.cat([inp, h], dim=1)), spikegen_vmem)
            sum_linear_inp += linear_inp
            igate_inp = self.linear_igate(linear_inp)
            fgate_inp = self.linear_fgate(linear_inp)
            cgate_inp = self.linear_cgate(linear_inp)
            ogate_inp = self.linear_ogate(linear_inp)
            igate_outp,igate_vmem = self.igate(igate_inp,igate_vmem) 
            sum_igate_outp += igate_outp
            fgate_outp,fgate_vmem = self.fgate(fgate_inp,fgate_vmem) 
            sum_fgate_outp += fgate_outp
            cgate_outp,cgate_vmem_pos,cgate_vmem_neg = self.cgate(cgate_inp,cgate_vmem_pos,cgate_vmem_neg) 
            sum_cgate_outp += cgate_outp
            ogate_outp,ogate_vmem = self.ogate(ogate_inp,ogate_vmem)
            sum_ogate_outp += ogate_outp
        avg_linear_inp = sum_linear_inp/self.num_timestep
        avg_igate_outp = sum_igate_outp/self.num_timestep 
        avg_fgate_outp = sum_fgate_outp/self.num_timestep 
        avg_cgate_outp = sum_cgate_outp/self.num_timestep
        avg_ogate_outp = sum_ogate_outp/self.num_timestep
        c_new = (avg_fgate_outp*c) + ((avg_igate_outp)*avg_cgate_outp)
        h_new = (avg_ogate_outp)*self.act(c_new)
        return (h_new, c_new)

class VanillaLinear(nn.Module):
    def __init__(self, input_size, output_size, num_timestep):
        super(VanillaLinear, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.reset_parameters(input_size, output_size)
    def reset_parameters(self, input_size, output_size):
        self.linear.weight.data.uniform_(-math.sqrt(1./input_size), math.sqrt(1./input_size))
        self.linear.bias.data.uniform_(-math.sqrt(1./input_size), math.sqrt(1./input_size))
    def forward(self, inp):
        return self.linear(inp)

class MeanTracker():
    def __init__(self):
        self.reset()
    def reset(self):
        self._sum = 0
        self._tsize = 0
    def update(self, mean_inp, size_inp):
        self._sum += mean_inp*size_inp
        self._tsize += size_inp
    def mean(self):
        return self._sum/self._tsize

def accuracy(outp, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = outp.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def main():
    parser = argparse.ArgumentParser(description='Train/Test LSTM on Sequential MNIST task', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--start-epoch', default=1, type=int, help='number of training epoch')
    parser.add_argument('--end-epoch', default=50, type=int, help='number of training epoch')
    parser.add_argument('--input-size', default=28, type=int, help='input size/number of pixels to be fed at a particular step')
    parser.add_argument('--hidden-size', default=128, type=int, help='hidden size')
    parser.add_argument('--dropout', default=0.5, type=float, help='RNN dropout probability')
    parser.add_argument('--num-layer', default=1, type=int, help='number of RNN layer')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--grad-clip', default=1.0, type=float, help='maximum gradient')
    parser.add_argument('--momentum', default=0.0, type=float, help='momentum')
    parser.add_argument('--seed', default=000000, type=int, help='random seed')
    parser.add_argument('--n-timestep', default=128, type=int, help='number of timesteps to be used')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true', help='overwrite the existing log file')
    parser.add_argument('--checkpoint-dir', required=True, type=str, help='name of checkpoint directory')
    parser.add_argument('--mode', required=True, default='train', type=str, help='run mode')
    parser.add_argument('--permute', dest='permute', action='store_true', help='permute pixel')
    parser.add_argument('--lstm-cell-class', default='ModifiedLSTMCell', type=str, help='lstm cell type')

    # parse argument
    args = parser.parse_args()
   
    # set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # do not proceed if log file exists unless overwrite option is specified
    log_name = '{}_{}.log'.format(args.checkpoint_dir, args.mode)
    if not args.overwrite and os.path.isfile(log_name):
        print('| file {} exists!'.format(log_name))
        sys.exit(0)
    # open log file to write and create checkpoint directory 
    log_file = open(log_name, 'w', buffering=1)
    pathlib.Path(os.path.abspath(args.checkpoint_dir)).mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir = os.path.abspath(args.checkpoint_dir)

    # save input argument
    for arg in vars(args):
        log_file.write('| {} = {}\n'.format(arg, getattr(args, arg)))

    # create sequential model
    if args.lstm_cell_class == 'BaselineLSTMCell':
        lstm_cell_class = BaselineLSTMCell
    elif args.lstm_cell_class == 'ModifiedLSTMCell':
        lstm_cell_class = ModifiedLSTMCell
    linear_class = VanillaLinear
    model_rate = SequentialModel(
            lstm_cell_class=lstm_cell_class,
            linear_class=linear_class,
            num_timestep=args.n_timestep,
            input_size=args.input_size,
            num_layer=args.num_layer, 
            hidden_size=args.hidden_size,
            dropout_prob=args.dropout,
            output_size=10
        ).cuda()
    log_file.write('| Rate-based model\n')
    log_file.write(str(model_rate)+'\n')
    
    # create optimizer and loss function 
    optimizer = torch.optim.RMSprop(params=model_rate.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss().cuda()
    best_test_acc = 0.0
    start_epoch = args.start_epoch
    log_file.write('| use RMSprop() and CrossEntropyLoss()\n')

    # load best model
    if 'test' in args.mode:
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'checkpoint_best.tar'))
        model_rate.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_test_acc = checkpoint['acc']
        start_epoch = checkpoint['start_epoch']
    log_file.write('| load trained model from {}\n'.format(os.path.join(args.checkpoint_dir, 'checkpoint_best.tar')))
   
    # load dataset
    if args.permute:
        pixel_permutation = torch.randperm(784)
        log_file.write('| pixel_permutation = {}\n'.format(pixel_permutation.tolist()))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.8693,)),
            transforms.Lambda(lambda x: x.view(-1,args.input_size)[pixel_permutation])
            ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.8693,)),
            transforms.Lambda(lambda x: x.view(-1,args.input_size))
            ])

    dataset_path = os.path.abspath('<path_dataset>')
    train_set = datasets.MNIST(root=dataset_path, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_set = datasets.MNIST(root=dataset_path, train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # train model
    if 'train' in args.mode:
        log_file.write('######## training ########\n')
        for epoch in range(start_epoch, args.end_epoch+1):
            # initialize mean tracker for calculating mean accuracy 
            meantracker_train = MeanTracker()
            meantracker_loss = MeanTracker()
            meantracker_test = MeanTracker()
            
            # train model
            model_rate.train()
            for i, (inp, target) in enumerate(tqdm(train_loader, desc='Training rate model')):
                batch_size = inp.size(0)
                
                inp, target = inp.cuda(), target.cuda()
                outp = model_rate(inp)
                loss = criterion(outp, target)

                # measure and record accuracy
                prec1, prec5 = accuracy(outp, target, topk=(1, 5))
                meantracker_train.update(float(prec1), batch_size)
                meantracker_loss.update(float(loss), batch_size)
                
                # compute gradient and do sgd step
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model_rate.parameters(), args.grad_clip)
                optimizer.step()

                # clip weight and bias
                for name, param in model_rate.named_parameters():
                    if ('weight' in name) or ('bias' in name):
                        param.data.copy_(torch.clamp(param.data, min=-1, max=1))
            
            # test model
            model_rate.eval()
            with torch.no_grad():
                for i, (inp, target) in enumerate(tqdm(test_loader, desc='Testing rate model')):
                    batch_size = inp.size(0)
                    
                    inp, target = inp.cuda(), target.cuda()
                    outp = model_rate(inp)

                    # measure and record accuracy
                    prec1, prec5 = accuracy(outp, target, topk=(1, 5))
                    meantracker_test.update(float(prec1), batch_size)

            # save model to checkpoint
            if (epoch%10 == 0):
                torch.save({'start_epoch': epoch+1,'state_dict': model_rate.state_dict(),'optimizer': optimizer.state_dict(),'acc':best_test_acc},\
                        os.path.join(args.checkpoint_dir, 'checkpoint_{}.tar'.format(epoch)))
            if meantracker_test.mean() > best_test_acc:
                best_test_acc = meantracker_test.mean()
                torch.save({'start_epoch': epoch+1,'state_dict': model_rate.state_dict(),'optimizer': optimizer.state_dict(),'acc':best_test_acc},\
                        os.path.join(args.checkpoint_dir, 'checkpoint_best.tar'))
        
            # print stat
            log_file.write('------- epoch {}/{} --------\n'.format(epoch, args.end_epoch))
            log_file.write('training accuracy: {:.2f}, training loss: {:.5f}, test accuracy: {:.2f}\n'.format(meantracker_train.mean(), meantracker_loss.mean(), meantracker_test.mean()))
        log_file.write('best testing accuracy: {:.2f}\n'.format(best_test_acc))

    else:
        log_file.write('######## inferencing ########\n')
        meantracker_test = MeanTracker()
        model_rate.eval()
        with torch.no_grad():
            for i, (inp, target) in enumerate(tqdm(test_loader, desc='Inferencing with rate model')):
                batch_size = inp.size(0)
                inp, target = inp.cuda(), target.cuda()
                outp = model_rate(inp)

                # measure and record accuracy
                prec1, prec5 = accuracy(outp, target, topk=(1, 5))
                meantracker_test.update(float(prec1), batch_size)
            log_file.write('inference accuracy with rate model: {:.2f}\n'.format(meantracker_test.mean()))

        # create hybrid model
        lstm_cell_class = HybridLSTMCell
        linear_class = VanillaLinear
        model_hybrid = SequentialModel(
                lstm_cell_class=lstm_cell_class,
                linear_class=linear_class,
                num_timestep=args.n_timestep,
                input_size=args.input_size,
                num_layer=args.num_layer, 
                hidden_size=args.hidden_size,
                dropout_prob=args.dropout,
                output_size=10
            ).cuda()
        log_file.write('| Hybrid model\n')
        log_file.write(str(model_hybrid)+'\n')

        def scale_wgt(inp_wgt):
            return torch.clamp((torch.round(inp_wgt*128)*2),min=-254, max=254)
        def scale_bias(inp_bias, add_bias=512):
            return add_bias+torch.round(inp_bias*256)

        # copy scaled weight of RNN units from the trained model
        wgt_igate = scale_wgt(torch.cat([model_rate.list_lstm_cell[0].linear1.weight.data[0:args.hidden_size,:], model_rate.list_lstm_cell[0].linear2.weight.data[0:args.hidden_size,:]], dim=1))
        wgt_fgate = scale_wgt(torch.cat([model_rate.list_lstm_cell[0].linear1.weight.data[args.hidden_size:2*args.hidden_size,:], model_rate.list_lstm_cell[0].linear2.weight.data[args.hidden_size:2*args.hidden_size,:]], dim=1))
        wgt_cgate = scale_wgt(torch.cat([model_rate.list_lstm_cell[0].linear1.weight.data[2*args.hidden_size:3*args.hidden_size,:], model_rate.list_lstm_cell[0].linear2.weight.data[2*args.hidden_size:3*args.hidden_size,:]], dim=1))
        wgt_ogate = scale_wgt(torch.cat([model_rate.list_lstm_cell[0].linear1.weight.data[3*args.hidden_size:4*args.hidden_size,:], model_rate.list_lstm_cell[0].linear2.weight.data[3*args.hidden_size:4*args.hidden_size,:]], dim=1))
        model_hybrid.list_lstm_cell[0].linear_igate.weight.data.copy_(wgt_igate)
        model_hybrid.list_lstm_cell[0].linear_fgate.weight.data.copy_(wgt_fgate)
        model_hybrid.list_lstm_cell[0].linear_cgate.weight.data.copy_(wgt_cgate)
        model_hybrid.list_lstm_cell[0].linear_ogate.weight.data.copy_(wgt_ogate)
        # copy scaled bias of RNN units from the trained model
        bias_igate = scale_bias(model_rate.list_lstm_cell[0].bias.bias[0:args.hidden_size])
        bias_fgate = scale_bias(model_rate.list_lstm_cell[0].bias.bias[args.hidden_size:2*args.hidden_size])
        bias_cgate = scale_bias(model_rate.list_lstm_cell[0].bias.bias[2*args.hidden_size:3*args.hidden_size], add_bias=0)
        bias_ogate = scale_bias(model_rate.list_lstm_cell[0].bias.bias[3*args.hidden_size:4*args.hidden_size])
        model_hybrid.list_lstm_cell[0].igate.bias.data.copy_(bias_igate)
        model_hybrid.list_lstm_cell[0].fgate.bias.data.copy_(bias_fgate)
        model_hybrid.list_lstm_cell[0].cgate.bias.data.copy_(bias_cgate)
        model_hybrid.list_lstm_cell[0].ogate.bias.data.copy_(bias_ogate)
        # copy weight and bias of classifier from the trained model
        model_hybrid.classifier.linear.weight.data.copy_(model_rate.classifier.linear.weight.data)
        model_hybrid.classifier.linear.bias.data.copy_(model_rate.classifier.linear.bias.data)

        # test model
        meantracker_test = MeanTracker()
        model_hybrid.eval()
        with torch.no_grad():
            for i, (inp, target) in enumerate(tqdm(test_loader, desc='Inferencing with hybrid model')):
                batch_size = inp.size(0)
                inp, target = inp.cuda(), target.cuda()
                outp = model_hybrid(inp)

                # measure and record accuracy
                prec1, prec5 = accuracy(outp, target, topk=(1, 5))
                meantracker_test.update(float(prec1), batch_size)
            log_file.write('inference accuracy with hybrid model: {:.2f}\n'.format(meantracker_test.mean()))

if __name__ == '__main__':
    main()
