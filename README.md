# Single-View Graph Contrastive Learning with Soft Neighborhood Awareness (SIGNA)

This repository contains the source code for SIGNA. 

## Run

Transductive tasks
```shell
python main.py --dataset photo 
```

Inductive tasks
```shell
python ppi_main.py --dataset ppi
```

Important args:
* `--dataset` wikics, photo, computers, cs, physics, flickr, ppi

All datasets will be downloaded directly from publicly available sources, and thus we do not upload them to this repo.

## The results on Amazon Photo should be close to:

<img width="600" src="https://github.com/sunisfighting/NETON/blob/main/example_photo.jpg"/>

