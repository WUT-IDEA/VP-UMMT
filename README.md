# Visual Pivoting Unsupervised Multimodal Machine Translation in Low-Resource Distant Language Pairs



## Installation

Install the python package in editable mode with
```bash
pip install -e .
```

## Dependencies

- Python 3
- [NumPy](http://www.numpy.org/)
- [PyTorch](http://pytorch.org/) (currently tested on version 0.4 and 1.0)
- [fastBPE](https://github.com/facebookresearch/XLM/tree/master/tools#fastbpe) (generate and apply BPE codes)
- [Moses](https://github.com/facebookresearch/XLM/tree/master/tools#tokenizers) (scripts to clean and tokenize text only - no installation required)
- [Apex](https://github.com/nvidia/apex#quick-start) (for fp16 training)


## About the Codebase
- The codebase is a revised, improved and extended version of [XLM](https://github.com/facebookresearch/XLM).
- No changes applied to multi-GPU code, which did not work well for us. All models were trained on a single GPU.
- Code portions that are not used are stripped to make the code simpler and clearer.
- Code paths for some of the original XLM tasks such as causal LM, back-translation, auto-encoding, etc.
  are not used/tested and probably broken due to refactoring.
  
 
### Data Preparation

Text data set download [URL](https://github.com/Turghuns/MMT-datasets).

### Pre-training and Fine-tuning

We provide shell scripts under `scripts/` for pre-training and fine-tuning workflows that are
used throughout the paper. For pre-training the MLM and VMLM, we provide two
toy examples that'll do pre-training on Multi30k as a starting point:

- `mlm.sh`: Trains a MLM model on the Multi30k En-De corpus.
- `vmlm.sh`: Trains a VMLM model on the Multi30k En-De corpus.


#### Train UNMT/UMMT from scratch
You can use `[unmt|ummt]-from-scratch.sh` scripts to train UNMT/UMMT
baselines without pre-training / fine-tuning.

#### Fine-tuning VMLM on Multi30k
You can use `ummt-fintune.sh` scripts to fine-tune
existing TLM/VTLM checkpoints on NMT and MMT tasks, using the En-De Multi30k corpus.

#### Decoding NMT/MMT systems
`decode-[nmt|mmt].sh` scripts can be used on arbitrarily trained UNMT/UMMT
checkpoints, to decode translations of `val` and `test_2016_flickr` test set.
By default, it uses a beam size of 8.


# Citation

Please cite as:

``` bibtex
@inproceedings{DBLP:conf/emnlp/Tayir,
  author       = {Turghun Tayir and
                  Lin Li and
                  Xiaohui Tao and
                  Mieradilijiang Maimaiti and
                  Ming Li and 
                  Jianquan Liu},
  title        = {Visual Pivoting Unsupervised Multimodal Machine Translation in Low-Resource Distant Language Pairs},
  booktitle    = {Findings of the Association for Computational Linguistics},
  year         = {2024},

}
```

## License

See the [LICENSE](LICENSE) file for more details.
