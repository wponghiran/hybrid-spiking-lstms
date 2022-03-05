# Hybrid Analog-Spiking Long Short-Term Memory for Energy Efficient Computing on Edge Devices

This repository contains the original implementation for training LSTMs, performing LSTM-to-Hybrid LSTM conversion, and evaluating.

For more details or citation purpose, please visit [link to the paper](https://ieeexplore.ieee.org/document/9473953) 
```
@inproceedings{ponghiran2021hybrid,
  title={Hybrid Analog-Spiking Long Short-Term Memory for Energy Efficient Computing on Edge Devices},
  author={Ponghiran, Wachirawit and Roy, Kaushik},
  booktitle={2021 Design, Automation \& Test in Europe Conference \& Exhibition (DATE)},
  pages={581--586},
  year={2021},
  organization={IEEE}
}
```

## Requirements

A list of the required packages that we use in our experiment is in requirements.txt.
Some packages like fairseq would require pulling from an online repository and manual installation.
To print out requirements, run:

```
cat requirements.txt
```

To use different learning rates for different modules with fairseq, run:

```
cp ./fairseq_modified_files/__init__ <fairseq_package_installed_location>/optim/__init__.py
cp ./fairseq_modified_files/trainer.py <fairseq_package_installed_location>/trainer.py
```

## Training and Evaluating Sequential  Recognition Model

To train LSTM for sequential recognition based on rows of images, run:
```
# Sequential recognition based on rows of images
python run_seqmnist.py --checkpoint-dir <checkpoint_dir_for_baseline_lstm> --seed <seed> --mode train --lstm-cell-class BaselineLSTMCell
python run_seqmnist.py --checkpoint-dir <checkpoint_dir_for_modified_lstm> --seed <seed> --mode train --lstm-cell-class ModifiedLSTMCell

# Sequential recognition based on unpermuted pixels of images
python run_seqmnist.py --checkpoint-dir <checkpoint_dir_for_baseline_lstm> --seed <seed> --mode train --input-size 1 --lstm-cell-class BaselineLSTMCell
python run_seqmnist.py --checkpoint-dir <checkpoint_dir_for_modified_lstm> --seed <seed> --mode train --input-size 1 --lstm-cell-class ModifiedLSTMCell
```
Trained LSTM with the constraints proposed in the paper (ModifiedLSTMCell in the code) is automatically converted to hybrid LSTM during the evaluation. Specify the number of time-steps to see the impact of SNN time-steps on classification accuracy. To evaluate the model, run:
```
# Evaluate the impact of recognition based on rows of images
python run_seqmnist.py --checkpoint-dir <checkpoint_dir_for_modified_lstm> --seed <seed> --mode test --n-timestep <number_of_snn_timesteps>

# Evaluate the impact of sequential recognition based on unpermuted pixels of images
python run_seqmnist.py --checkpoint-dir <checkpoint_dir_for_modified_lstm> --seed <seed> --mode test --input-size 1 --n-timestep <number_of_snn_timesteps>
```

## Training and Evaluating NMT Model

To prepare the IWSLT14 dataset, follow instructions on [fairseq/examples/translation](https://github.com/pytorch/fairseq/tree/master/examples/translation). We use all default setting for pre-processing steps. To train LSTM for sequence-to-sequence translation using fairseq, run:

```
fairseq-train <location_of_iwslt_dataset> --lr 0.2,0.02 --clip-norm 5 --criterion cross_entropy --lr-scheduler custom_scheduler --optimizer custom_adagrad --user-dir <current_dir>/fairseq_custom_modules --max-epoch 15 --save-dir <save_baseline_dir> --arch custom_baseline_iwslt --cell-type default --init-type orthogonal --batch-size 64 --seed <seed>
fairseq-train <location_of_iwslt_dataset> --lr 0.2,0.02 --clip-norm 5 --criterion cross_entropy --lr-scheduler custom_scheduler --optimizer custom_adagrad --user-dir <current_dir>/fairseq_custom_modules --max-epoch 15 --save-dir <save_modified_dir> --arch custom_baseline_iwslt --cell-type modified --init-type orthogonal --batch-size 64 --seed <seed>
```
To properly adjust LSTM connection weight (implicit step in the LSTM-to-Hybrid LSTM conversion), run:
```
python modify_checkpoint_iwslt.py --src-path <save_dir>/checkpoint_best.pt --des-path <save_dir>/checkpoint_best_modified.pt
```
 To evaluate the model, run:
```
fairseq-generate <location_of_iwslt_dataset> --beam 5 --remove-bpe --user-dir <fairseq>/fairseq_custom_modules --quiet --path <save_dir>/checkpoint_best_modified.pt
```

## Loihi Benchmarks

Loihi benchmarking scripts contain proprietary code and run on a research chip that is not publicly available.
Please contact us if you obtain access to the research chip and are interested in replicating the results.

## Results

Average hybrid LSTM performance of sequential  recognition with rows of images is:
| SNN time-steps | 16    | 32    | 64    | 128   |
|----------------|-------|-------|-------|-------|
| Accuracy (%)   | 92.89 | 98.37 | 98.74 | 98.81 | 
Average accuracy with ModifiedLSTMCell and BaselineLSTMCell are 98.95 and 98.97 respectively.

Average hybrid LSTM performance of sequential  recognition with pixels of images is:
| SNN time-steps | 32    | 128   | 256   | 512   | 1024  |
|----------------|-------|-------|-------|-------|-------|
| Accuracy (%)   | 11.89 | 33.28 | 81.15 | 96.03 | 97.61 |
Average accuracy with ModifiedLSTMCell and BaselineLSTMCell are 97.93 and 97.71 respectively.

Average hybrid LSTM performance of sequential  recognition with pixels of images is:
| SNN time-steps | 32   | 64    | 96    | 128   | 192   | 256   |
|----------------|------|-------|-------|-------|-------|-------|
| BLEU score     | 6.72 | 17.59 | 20.85 | 22.16 | 23.13 | 23.51 |
Average BLEU score with ModifiedLSTMCell and BaselineLSTMCell are 24.15 and 25.00 respectively.

