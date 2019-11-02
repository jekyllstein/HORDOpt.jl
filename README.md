# HORDOpt

[![Build Status](https://travis-ci.org/jekyllstein/HORDOpt.jl.svg?branch=master)](https://travis-ci.org/jekyllstein/HORDOpt.jl)
[![codecov.io](https://codecov.io/gh/jekyllstein/HORDOpt.jl/branch/master/graphs/badge.svg?)](http://codecov.io/gh/jekyllstein/HORDOpt.jl)
[![Coverage Status](https://coveralls.io/repos/github/jekyllstein/HORDOpt.jl/badge.svg?branch=master)](https://coveralls.io/github/jekyllstein/HORDOpt.jl?branch=master)

Code for reproducing results published in the paper "Efficient Hyperparameter Optimization of Deep Learning Algorithms Using Deterministic RBF Surrogates" (AAAI-17) by Ilija Ilievski, Taimoor Akhtar, Jiashi Feng, and Christine Annette Shoemaker.

[arXiv](https://arxiv.org/abs/1607.08316) -- [PDF](https://arxiv.org/pdf/1607.08316)

## Installation

Within Julia REPL enter Pkg mode by typing ']' then execute

```
add https://github.com/jekyllstein/HORDOpt.jl.git
```

Ensure packaging has been installed properly by running ```Pkg.test("HORDOpt")``` 

----

## Usage

## License
HORDOpt.jl is released under the [GPLv3 license](./LICENSE.md).

## Citing the HORD algorithm
To cite the paper use the following BibTeX entry:

> ```
> @inproceedings{ilievski2017efficient,
>   title={Efficient Hyperparameter Optimization of Deep Learning Algorithms Using Deterministic RBF Surrogates},
>   author={Ilievski, Ilija and Akhtar, Taimoor and Feng, Jiashi and Shoemaker, Christine},
>   booktitle={31st AAAI Conference on Artificial Intelligence (AAAI-17)},
>   year={2017}
> }
> ```