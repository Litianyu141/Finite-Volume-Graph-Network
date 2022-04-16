# GQSAT 

Can Q-learning with Graph Networks learn a Generalizable Branching Heuristic for a SAT solver?

## How to add metadata for evaluation

```python3 add_metadata.py --eval-problems-paths <path_to_folder_with_cnf>```

## How to train

```./train.sh```

## How to evaluate 

* add the path to the model to the script first
* choose the evaluation dataset
* ```./evaluate.sh```


## How to build a solver (you need this only if you changed the c++ code)

Run `make && make python-wrap` in the minisat folder.

## How to build swig code (if you changed minisat-python interface, e.g. in GymSolver.i)

Go to minisat/minisat/gym, run `swig -fastdispatch -c++ -python3 GymSolver.i` and then repeat the building procedure from the previous step.

## Individual Contributor License Agreement

Please fill out the following CLA and email to sgodil@nvidia.com:  https://www.apache.org/licenses/icla.pdf

## Cite

```
@inproceedings{kurin2019improving,
  title={Can Q-Learning with Graph Networks Learn a Generalizable Branching Heuristic for a SAT Solver?},
  author={Kurin, Vitaly and Godil, Saad and Whiteson, Shimon and Catanzaro, Bryan},
  booktitle = {Advances in Neural Information Processing Systems 32},
  year={2020}
}
```

## Acknowledgements

We would like to thank [Fei Wang](https://github.com/feiwang3311/minisat) whose initial implementation of the environment we used as a start, and the creators of [Minisat](https://github.com/niklasso/minisat) on which it is based on.
We would also like to thank the creators of [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric) whose 
MetaLayer and [Graph Nets](https://arxiv.org/abs/1806.01261) implementation we built upon. 