# Overview
This is the modified repo from https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets, which made meshgraphnet can predict pressure field， plot Cl Cd and write to tecplot. You can find more info from the original repo. We also add single edge encode method support.

# Download dataset
Datasets can be downloaded using the script download_dataset.sh. They contain a metadata file describing the available fields and their shape, and tfrecord datasets for train, valid and test splits. Dataset names match the naming in the paper. This repo is only suit for incompressible dataset.

# Env
For users with RTX 30\40\A100 series graphics cards, it would be advisable to install a specific version of TensorFlow via pip to run the code in this repository successfully. This can be done as shown below:
## 1.install tensorflow1.1x wheel index
~~~py
pip install nvidia-pyindex
~~~

## 2.install tensorflow1.1x-gpu version
~~~py
pip install nvidia-tensorflow
~~~
## 3.install other module specified in requirements.txt
For users with other types of graphics cards, you are able to use the environment specified in the original Meshgraphnets repository.
# Train&Test
you can modify such hyperparameters in all .sh files:
~~~py
batch_size=2
num_training_steps=10000000
num_rollouts=100
dual_edge=true # dual_edge=true means the model will encode undirected graph and causing high demand of memory.
save_tec=false
plot_boundary=true
~~~
Then set your dataset path in the sh file and run .sh file.

~~~sh
sh MGN-src/main/run_model.py 2>&1 | tee training_log.txt
~~~

# Contact
Email: lty1040808318@163.com if you have any questions.

By Tianyu Li
