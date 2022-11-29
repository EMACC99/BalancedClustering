# BalancedClustering

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
