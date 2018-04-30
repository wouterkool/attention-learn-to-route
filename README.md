# Attention Solves Your TSP

Attention based model for learning to solve the Travelling Salesman Problem. Training with REINFORCE with greedy rollout baseline.

## Paper
Please see our paper [Attention Solves Your TSP](https://arxiv.org/abs/1803.08475). 

## Dependencies

* Python>=3.5
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/)=0.3
* tqdm
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)

## Usage

For training TSP instances with 20 nodes and using rollout as REINFORCE baseline:
```bash
python run.py --graph_size 20 --baseline rollout --run_name 'tsp20_rollout'
```

By default, training will happen on all available GPUs. To disable CUDA at all, add the flag `--no_cuda`. 
Set the environment variable `CUDA_VISIBLE_DEVICES` to only use specific GPUs:
```bash
CUDA_VISIBLE_DEVICES=2,3 python run.py 
```
Note that using multiple GPUs has limited efficiency for small problem sizes (up to 50 nodes).

To evaluate a model, use the `--load_path` option to specify the model to load and add the `--eval_only` option, for example:
```bash
python run.py --graph_size 20 --eval_only --load_path 'outputs/tsp_20/tsp20_rollout_{datetime}/epoch-0.pt'
```

To load a pretrained model (single GPU only since it cannot load into `DataParallel`):
```bash
CUDA_VISIBLE_DEVICES=0 python run.py --graph_size 100 --eval_only --load_path pretrained/tsp100.pt
```
Note that the results may differ slightly from the results reported in the paper, as a different test set was used than the validation set (which depends on the random seed).

For other options and help:
```bash
python run.py -h
```
