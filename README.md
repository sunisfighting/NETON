### Do Not Expect Too Much from Your Neighbors in Contrastive Learning of Node Representation

This repository contains the source code for NETON. 

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

# The results on Amazon Photo dataset should be close to:


![example_photo](https://github.com/sunisfighting/NETON/blob/main/example_photo.jpg)
