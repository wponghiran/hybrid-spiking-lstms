
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import FairseqEncoder, FairseqIncrementalDecoder, FairseqEncoderDecoderModel, register_model, register_model_architecture

@register_model('custom_baseline')
class CustomBaselineModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--n-timesteps', type=int, metavar='n', help='number of timesteps for simulation')
        parser.add_argument('--spike-inf', action="store_true", help='enable spiking inference')
        parser.add_argument('--cell-type', type=str, help='lstm cell type')
        parser.add_argument('--init-type', type=str, help='lstm initialization type')
        parser.add_argument('--encoder-embed-norm', action="store_true", help='normalize encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N', help='encoder embedding dimension')
        parser.add_argument('--encoder-num-layer', type=int, metavar='N', help='encoder layer')
        parser.add_argument('--encoder-hidden-size', type=int, metavar='N', help='encoder hidden size')
        parser.add_argument('--decoder-embed-norm', action="store_true", help='normalize decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N', help='decoder embedding dimension')
        parser.add_argument('--decoder-num-layer', type=int, metavar='N', help='decoder layer')
        parser.add_argument('--decoder-hidden-size', type=int, metavar='N', help='decoder hidden size')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N', help='decoder output embedding dimension')
        parser.add_argument('--encoder-dropout-in', type=float, metavar='D', help='dropout probability for encoder input embedding')
        parser.add_argument('--encoder-dropout-out', type=float, metavar='D', help='dropout probability for encoder output')
        parser.add_argument('--decoder-dropout-in', type=float, metavar='D', help='dropout probability for decoder input embedding')
        parser.add_argument('--decoder-dropout-out', type=float, metavar='D', help='dropout probability for decoder output')

    @classmethod
    def build_model(cls, args, task):
        encoder = BaseEncoder(
            dictionary=task.source_dictionary,
            n_timesteps=args.n_timesteps,
            spike_inf=args.spike_inf,
            embed_dim=args.encoder_embed_dim,
            num_layer=args.encoder_num_layer,
            embed_norm=args.encoder_embed_norm,
            hidden_size=args.encoder_hidden_size,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            cell_type=args.cell_type,
            init_type=args.init_type,
        )
        decoder = BaseDecoder(
            dictionary=task.target_dictionary,
            n_timesteps=args.n_timesteps,
            spike_inf=args.spike_inf,
            embed_dim=args.decoder_embed_dim,
            num_layer=args.decoder_num_layer,
            embed_norm=args.decoder_embed_norm,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            encoder_output_units=encoder.hidden_size,
            cell_type=args.cell_type,
            init_type=args.init_type,
        )
        return cls(encoder, decoder)

class BaseEncoder(FairseqEncoder):
    """LSTM encoder."""
    def __init__(self,
        dictionary,
        n_timesteps=128,
        spike_inf=False,
        embed_dim=256,
        num_layer=1, 
        embed_norm=True,
        hidden_size=256,
        dropout_in=0.0,
        dropout_out=0.2,
        cell_type='base',
        init_type='uniform',
    ):
        super().__init__(dictionary)
        self.hidden_size = hidden_size
        self.padding_idx = dictionary.pad()
        self.embed_norm = embed_norm
        self.num_layer = num_layer

        # Embedding layer 
        self.embedding = nn.Embedding(num_embeddings=len(dictionary), embedding_dim=embed_dim, padding_idx=self.padding_idx)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.constant_(self.embedding.weight[self.padding_idx], 0)
    
        # Dropout layer
        self.dropout_in = nn.Dropout(dropout_in) 

        # Recurrent layer
        if cell_type == 'default':
            lstm = DefaultLSTMCell
        elif cell_type == 'modified':
            lstm = HybridLSTMCell if spike_inf else ModifiedLSTMCell
        else:
            raise NotImplementedError
        self.list_lstm_cell = nn.ModuleList([lstm(input_size=embed_dim, hidden_size=hidden_size, init_type=init_type, n_timesteps=n_timesteps) if layer==0 \
                else lstm(input_size=hidden_size, hidden_size=hidden_size, init_type=init_type, n_timesteps=n_timesteps) for layer in range(self.num_layer)])

        # Dropout layer
        self.dropout_out = nn.Dropout(dropout_out)

    def forward(self, src_tokens, src_lengths):
        batch_size, seq_len = src_tokens.size()

        # Convert left-padding to right-padding
        src_tokens = utils.convert_padding_direction(src_tokens, self.padding_idx, left_to_right=True)

        # Forward through embedding layer 
        if self.training and self.embed_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=1)
        emb_inp = self.dropout_in(self.embedding(src_tokens))
        emb_inp = emb_inp.transpose(0, 1) # (B, T, C) -> (T, B, C)

        list_final_h = [] 
        list_final_c = []
        lstm_inp = emb_inp
        max_timestep = emb_inp.size(0)
        for layer in range(self.num_layer):
            h = torch.zeros(batch_size, self.hidden_size).cuda()
            c = torch.zeros(batch_size, self.hidden_size).cuda()
            lstm_outp = [] 
            for timestep in range(max_timestep):
                next_h, next_c = self.list_lstm_cell[layer](lstm_inp[timestep], (h,c))
                lstm_outp.append(next_h)
                mask = (timestep < src_lengths).float().unsqueeze(1).expand_as(next_h)
                next_h = next_h*mask + h*(1-mask)
                next_c = next_c*mask + c*(1-mask)
                h, c = next_h, next_c
            lstm_outp = self.dropout_out(torch.stack(lstm_outp, 0))
            lstm_inp = lstm_outp
            list_final_h.append(h)
            list_final_c.append(c)
        list_final_h = torch.stack(list_final_h, 0)
        list_final_c = torch.stack(list_final_c, 0)
        # Obtain padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        return {'encoder_out': (lstm_outp, list_final_h, list_final_c), \
                'encoder_padding_mask': encoder_padding_mask if encoder_padding_mask.any() else None}

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out['encoder_out'] = tuple(eo.index_select(1, new_order) for eo in encoder_out['encoder_out'])
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = encoder_out['encoder_padding_mask'].index_select(1, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number

class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_features=input_embed_dim, out_features=source_embed_dim, bias=False)
        self.linear2 = nn.Linear(in_features=input_embed_dim+source_embed_dim, out_features=output_embed_dim, bias=False)
        w = torch.empty(source_embed_dim, input_embed_dim);   nn.init.orthogonal_(w, gain=1.) 
        self.linear1.weight.data.copy_(w)
        w = torch.empty(output_embed_dim, input_embed_dim+source_embed_dim);   nn.init.orthogonal_(w, gain=1.) 
        self.linear2.weight.data.copy_(w)

    def forward(self, inp, src_hiddens, encoder_padding_mask):
        # Compute attention
        attn_scores = (src_hiddens * self.linear1(inp).unsqueeze(0)).sum(dim=2)
        if encoder_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(encoder_padding_mask, float('-inf')).type_as(attn_scores)  # FP16 support: cast to float and back
        attn_scores = F.softmax(attn_scores, dim=0)  # srclen x batch_size

        # Sum weighted sources
        outp1 = (src_hiddens * attn_scores.unsqueeze(2)).sum(dim=0)
        outp2 = torch.tanh(self.linear2(torch.cat((outp1, inp), dim=1)))

        return outp2, attn_scores

class BaseDecoder(FairseqIncrementalDecoder):
    """LSTM decoder."""
    def __init__(self,
        dictionary,
        n_timesteps=128,
        spike_inf=False,
        embed_dim=256,
        num_layer=1, 
        embed_norm=True,
        hidden_size=256,
        out_embed_dim=256,
        dropout_in=0.0,
        dropout_out=0.2,
        encoder_output_units=256, 
        cell_type='base',
        init_type='uniform',
    ):
        super().__init__(dictionary)
        self.hidden_size = hidden_size
        self.embed_norm = embed_norm
        self.num_layer = num_layer

        # Embedding layer 
        self.embedding = nn.Embedding(num_embeddings=len(dictionary), embedding_dim=embed_dim, padding_idx=dictionary.pad())
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.constant_(self.embedding.weight[dictionary.pad()], 0)

        # Dropout layer
        self.dropout_in = nn.Dropout(dropout_in) 

        # Recurrent layer + Attention
        if cell_type == 'default':
            lstm = DefaultLSTMCell
        elif cell_type == 'modified':
            lstm = HybridLSTMCell if spike_inf else ModifiedLSTMCell
        else:
            raise NotImplementedError
        self.list_lstm_cell = nn.ModuleList([lstm(input_size=hidden_size+embed_dim, hidden_size=hidden_size, init_type=init_type, n_timesteps=n_timesteps) if layer==0 else lstm(input_size=hidden_size, hidden_size=hidden_size, init_type=init_type, n_timesteps=n_timesteps) for layer in range(self.num_layer)])
        self.attention = AttentionLayer(hidden_size, encoder_output_units, hidden_size)
       
        # Dropout layer
        self.dropout_out = nn.Dropout(dropout_out) 

        # Fully connected layer 
        self.fc_out = nn.Linear(in_features=out_embed_dim, out_features=len(dictionary))
        w = torch.empty(len(dictionary), out_embed_dim);   nn.init.orthogonal_(w, gain=1.) 
        self.fc_out.weight.data.copy_(w)
        b = torch.zeros(len(dictionary))
        self.fc_out.bias.data.copy_(b)

    def forward(self, prev_output_tokens, encoder_out, incremental_state=None):
        # Get output of encoder and padding mask
        encoder_padding_mask = encoder_out['encoder_padding_mask']
        encoder_out, encoder_hiddens, encoder_cells = encoder_out['encoder_out'][:3]
        
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        batch_size, seq_len = prev_output_tokens.size()

        # Forward through embedding layer 
        if self.training and self.embed_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=1)
        prev_outps = self.embedding(prev_output_tokens)    # prev_output_token.size() = (B, T)
        prev_outps = self.dropout_in(prev_outps) 
        prev_outps = prev_outps.transpose(0, 1) # (B, T, H) -> (T, B, H)

        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            list_prev_hidden, list_prev_cell, input_feed = cached_state
        else:
            list_prev_hidden = [encoder_hiddens[i] for i in range(self.num_layer)]
            list_prev_cell = [encoder_cells[i] for i in range(self.num_layer)]
            input_feed = prev_outps.new_zeros(batch_size, self.hidden_size)

        outs = []
        for seq_idx in range(seq_len):
            # input feeding: concatenate context vector from previous time step
            inp = torch.cat((prev_outps[seq_idx, :, :], input_feed), dim=1)
            
            for layer, lstm_cell in enumerate(self.list_lstm_cell):
                # recurrent cell
                hidden, cell = self.list_lstm_cell[layer](inp, (list_prev_hidden[layer], list_prev_cell[layer]))
                # hidden state becomes the input to the next layer
                inp = self.dropout_out(hidden)

                # save state for next time step
                list_prev_hidden[layer] = hidden
                list_prev_cell[layer] = cell

            out,_ = self.attention(hidden, encoder_out, encoder_padding_mask)
            out = self.dropout_out(out)

            # input feeding
            input_feed = out

            # save final output
            outs.append(out)
        
        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(self, incremental_state, 'cached_state', (list_prev_hidden, list_prev_cell, input_feed),)

        # collect outputs across time steps
        x = torch.stack(outs)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        x = self.fc_out(x)
        return x, None

    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is None:
            return

        def reorder_state(state):
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            return state.index_select(0, new_order)

        new_state = tuple(map(reorder_state, cached_state))
        utils.set_incremental_state(self, incremental_state, 'cached_state', new_state)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e5)  # an arbitrary large number

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

class DefaultLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, init_type, n_timesteps):
        super(DefaultLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.reset_parameters(input_size, hidden_size, init_type)
        print('use DefaultLSTMCell')
    def reset_parameters(self, input_size, hidden_size, init_type):
        if init_type == 'uniform':
            print('Uniform initialization!')
            wi_i = torch.empty(hidden_size, input_size);   wi_i.uniform_(-0.1,0.1) 
            wi_f = torch.empty(hidden_size, input_size);   wi_f.uniform_(-0.1,0.1)  
            wi_c = torch.empty(hidden_size, input_size);   wi_c.uniform_(-0.1,0.1)  
            wi_o = torch.empty(hidden_size, input_size);   wi_o.uniform_(-0.1,0.1)  
            self.lstm_cell.weight_ih.data.copy_(torch.cat((wi_i,wi_f,wi_c,wi_o), dim=0))
            wh_i = torch.empty(hidden_size, hidden_size);   wh_i.uniform_(-0.1,0.1) 
            wh_f = torch.empty(hidden_size, hidden_size);   wh_f.uniform_(-0.1,0.1) 
            wh_c = torch.empty(hidden_size, hidden_size);   wh_c.uniform_(-0.1,0.1) 
            wh_o = torch.empty(hidden_size, hidden_size);   wh_o.uniform_(-0.1,0.1) 
            self.lstm_cell.weight_hh.data.copy_(torch.cat((wh_i,wh_f,wh_c,wh_o), dim=0))
        elif init_type == 'orthogonal':
            print('Orthogonal initialization!')
            wi_i = torch.empty(hidden_size, input_size);   nn.init.orthogonal_(wi_i, gain=1.)
            wi_f = torch.empty(hidden_size, input_size);   nn.init.orthogonal_(wi_f, gain=1.)
            wi_c = torch.empty(hidden_size, input_size);   nn.init.orthogonal_(wi_c, gain=1.)
            wi_o = torch.empty(hidden_size, input_size);   nn.init.orthogonal_(wi_o, gain=1.)
            self.lstm_cell.weight_ih.data.copy_(torch.cat((wi_i,wi_f,wi_c,wi_o), dim=0))
            wh_i = torch.empty(hidden_size, hidden_size);   nn.init.orthogonal_(wh_i, gain=1.) 
            wh_f = torch.empty(hidden_size, hidden_size);   nn.init.orthogonal_(wh_f, gain=1.) 
            wh_c = torch.empty(hidden_size, hidden_size);   nn.init.orthogonal_(wh_c, gain=1.) 
            wh_o = torch.empty(hidden_size, hidden_size);   nn.init.orthogonal_(wh_o, gain=1.) 
            self.lstm_cell.weight_hh.data.copy_(torch.cat((wh_i,wh_f,wh_c,wh_o), dim=0))
        else:
            raise NotImplementedError
        b_i = torch.zeros(hidden_size)
        b_f = torch.ones(hidden_size)
        b_c = torch.zeros(hidden_size)
        b_o = torch.zeros(hidden_size)
        self.lstm_cell.bias_ih.data.copy_(torch.cat((b_i, b_f,b_c,b_o), dim=0))
        b_i = torch.zeros(hidden_size)
        b_f = torch.zeros(hidden_size)
        b_c = torch.zeros(hidden_size)
        b_o = torch.zeros(hidden_size)
        self.lstm_cell.bias_hh.data.copy_(torch.cat((b_i, b_f,b_c,b_o), dim=0))
    def forward(self, inp, hx):
        return self.lstm_cell(inp, hx)

class ModifiedLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, init_type, n_timesteps):
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
        self.reset_parameters(input_size, hidden_size, init_type)
        print('use ModifiedLSTMCell')
    def reset_parameters(self, input_size, hidden_size, init_type):
        if init_type == 'uniform':
            print('Uniform initialization!')
            wi_i = torch.empty(hidden_size, input_size);   wi_i.uniform_(-0.1,0.1)  
            wi_f = torch.empty(hidden_size, input_size);   wi_f.uniform_(-0.1,0.1)  
            wi_c = torch.empty(hidden_size, input_size);   wi_c.uniform_(-0.1,0.1)  
            wi_o = torch.empty(hidden_size, input_size);   wi_o.uniform_(-0.1,0.1)  
            self.linear1.weight.data.copy_(torch.cat((wi_i, wi_f,wi_c,wi_o), dim=0))
            wh_i = torch.empty(hidden_size, hidden_size);   wh_i.uniform_(-0.1,0.1) 
            wh_f = torch.empty(hidden_size, hidden_size);   wh_f.uniform_(-0.1,0.1) 
            wh_c = torch.empty(hidden_size, hidden_size);   wh_c.uniform_(-0.1,0.1) 
            wh_o = torch.empty(hidden_size, hidden_size);   wh_o.uniform_(-0.1,0.1) 
            self.linear2.weight.data.copy_(torch.cat((wh_i, wh_f,wh_c,wh_o), dim=0))
        elif init_type == 'orthogonal':
            print('Orthogonal initialization!')
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
        else:
            raise NotImplementedError
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
    def __init__(self, input_size, hidden_size, init_type, n_timesteps):
        super(HybridLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_timestep = n_timesteps
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
        print('use HybridLSTMCell with {} ts!'.format(self.num_timestep))
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

@register_model_architecture('custom_baseline', 'custom_baseline_iwslt')
def config_model_iwslt(args):
    args.n_timesteps = getattr(args, 'n_timesteps', 128)
    args.spike_inf = getattr(args, 'spike_inf', False)
    args.encoder_embed_norm = getattr(args, 'encoder_embed_norm', True)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_num_layer = getattr(args, 'encoder_num_layer', 1)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', 256)
    args.decoder_embed_norm = getattr(args, 'decoder_embed_norm', True)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_num_layer = getattr(args, 'decoder_num_layer', 1)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 256)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', 0)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', 0)
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', 0)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', 0.2)

