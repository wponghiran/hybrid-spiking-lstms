
import argparse
import torch

parser = argparse.ArgumentParser(description='Modify checkpoint of the trained model for sequence-to-sequence translation to preserve model performance after conversion', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--src-path', required=True, type=str, help='path to source checkpoint')
parser.add_argument('--des-path', required=True, type=str, help='path to destination checkpoint')
args = parser.parse_args()

checkpoint = torch.load(args.src_path)

def scale_wgt(inp_wgt):
    return torch.clamp((torch.round(inp_wgt*128)*2),min=-254, max=254)
def scale_bias(inp_bias, add_bias=512):
    return add_bias+torch.round(inp_bias*256)

enc_linear1_wgt = checkpoint['model']['encoder.list_lstm_cell.0.linear1.weight']
enc_linear2_wgt = checkpoint['model']['encoder.list_lstm_cell.0.linear2.weight']
enc_bias = checkpoint['model']['encoder.list_lstm_cell.0.bias.bias']
hidden_size = enc_linear1_wgt.size(0)//4

checkpoint['model']['encoder.list_lstm_cell.0.linear_igate.weight'] = scale_wgt(torch.cat([enc_linear1_wgt[0:hidden_size,:], enc_linear2_wgt[0:hidden_size,:]], dim=1)).clone()
checkpoint['model']['encoder.list_lstm_cell.0.linear_fgate.weight'] = scale_wgt(torch.cat([enc_linear1_wgt[hidden_size:2*hidden_size,:], enc_linear2_wgt[hidden_size:2*hidden_size,:]], dim=1)).clone()
checkpoint['model']['encoder.list_lstm_cell.0.linear_cgate.weight'] = scale_wgt(torch.cat([enc_linear1_wgt[2*hidden_size:3*hidden_size,:], enc_linear2_wgt[2*hidden_size:3*hidden_size,:]], dim=1)).clone()
checkpoint['model']['encoder.list_lstm_cell.0.linear_ogate.weight'] = scale_wgt(torch.cat([enc_linear1_wgt[3*hidden_size:4*hidden_size,:], enc_linear2_wgt[3*hidden_size:4*hidden_size,:]], dim=1)).clone()

checkpoint['model']['encoder.list_lstm_cell.0.igate.bias'] = scale_bias(enc_bias[0:hidden_size]).clone()
checkpoint['model']['encoder.list_lstm_cell.0.fgate.bias'] = scale_bias(enc_bias[hidden_size:2*hidden_size]).clone()
checkpoint['model']['encoder.list_lstm_cell.0.cgate.bias'] = scale_bias(enc_bias[2*hidden_size:3*hidden_size], add_bias=0).clone()
checkpoint['model']['encoder.list_lstm_cell.0.ogate.bias'] = scale_bias(enc_bias[3*hidden_size:4*hidden_size]).clone()

del checkpoint['model']['encoder.list_lstm_cell.0.linear1.weight']
del checkpoint['model']['encoder.list_lstm_cell.0.linear2.weight']
del checkpoint['model']['encoder.list_lstm_cell.0.bias.bias']

dec_linear1_wgt = checkpoint['model']['decoder.list_lstm_cell.0.linear1.weight']
dec_linear2_wgt = checkpoint['model']['decoder.list_lstm_cell.0.linear2.weight']
dec_bias = checkpoint['model']['decoder.list_lstm_cell.0.bias.bias']

checkpoint['model']['decoder.list_lstm_cell.0.linear_igate.weight'] = scale_wgt(torch.cat([dec_linear1_wgt[0:hidden_size,:], dec_linear2_wgt[0:hidden_size,:]], dim=1)).clone()
checkpoint['model']['decoder.list_lstm_cell.0.linear_fgate.weight'] = scale_wgt(torch.cat([dec_linear1_wgt[hidden_size:2*hidden_size,:], dec_linear2_wgt[hidden_size:2*hidden_size,:]], dim=1)).clone()
checkpoint['model']['decoder.list_lstm_cell.0.linear_cgate.weight'] = scale_wgt(torch.cat([dec_linear1_wgt[2*hidden_size:3*hidden_size,:], dec_linear2_wgt[2*hidden_size:3*hidden_size,:]], dim=1)).clone()
checkpoint['model']['decoder.list_lstm_cell.0.linear_ogate.weight'] = scale_wgt(torch.cat([dec_linear1_wgt[3*hidden_size:4*hidden_size,:], dec_linear2_wgt[3*hidden_size:4*hidden_size,:]], dim=1)).clone()

checkpoint['model']['decoder.list_lstm_cell.0.igate.bias'] = scale_bias(dec_bias[0:hidden_size]).clone()
checkpoint['model']['decoder.list_lstm_cell.0.fgate.bias'] = scale_bias(dec_bias[hidden_size:2*hidden_size]).clone()
checkpoint['model']['decoder.list_lstm_cell.0.cgate.bias'] = scale_bias(dec_bias[2*hidden_size:3*hidden_size], add_bias=0).clone()
checkpoint['model']['decoder.list_lstm_cell.0.ogate.bias'] = scale_bias(dec_bias[3*hidden_size:4*hidden_size]).clone()

del checkpoint['model']['decoder.list_lstm_cell.0.linear1.weight']
del checkpoint['model']['decoder.list_lstm_cell.0.linear2.weight']
del checkpoint['model']['decoder.list_lstm_cell.0.bias.bias']

torch.save(checkpoint, args.des_path)
