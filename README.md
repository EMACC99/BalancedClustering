# BalancedClustering

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Repo for my balanced clustering experiments

## Required libraries

* numpy
* matplotlib
* pymoo
* cython
* pandas
* pstats
* Cprofile
* threading
* multiprocessing

## Cython Setup

Clone the repo with `git clone --recursive` to clone the submodules

Give permitions to the cython script with

```sh
chmod +x build_cython.sh
```

and excecute it with

```sh
./build_cython.sh
```

## Run

To run the script

```sh
python main.py run <args>
```

### Args

| Arg    | description | Default Value | Required|
|--------|-------------|---------------|---------|
|`-k`| Number of centroids | `N\A` | `True` |
|`--data`| The dataset to use | `N\A` | `True` |
|`--hard`| The archive hard limit | `75` | `False` |
|`--soft`| The archive soft limit | `150`| `False` |
|`--gamma`| The gamma to multiply the archve | `2` | `False` |
|`--climb`| Hill climbing iterations | `2500` | `False` |
|`--itemp`| SA initial temperature | `500` | `False` |
|`--ftemp`| SA final temperature | `1e-7` | `False` |
|`--iter`|Annealing iterations | `2500` | `False` |
|`--cool`| Cooling factor | `0.9` | `False` |
|`--win`| PHY-based early-termination window size | `10` | `False` |
|`--seed`| The seed for the RNG | `N\A` | `False` |
|`--alpha`| Step modifier for the hill climber | `1` | `False` |

## Visualizing the profiler data

To get a view of the profiler in an iteractive way, install the snakeviz python package, this can be done with

```sh
conda install -c conda-forge snakeviz
```

with anaconda.

To visualize the profiling data run

```sh
snakeviz profiler_<...>.prof
```
