# Decoder Choice Network for Meta-Learning

This repo contains code we used. It includes code for running the few-shot learning domain experiments, including Oracle classification.

### Dependencies
This code requires the following:
* python 3.\*
* Pytorch 0.4.1+

### Data
For the Oracle data, see the usage instructions in `pre_data/oracle_resized.py`. We resize the images to 28*28 to decrease the parameters in the model to do 200-way training.

To generate training data, see command in `pre_data/entire_image_generate.py` and `pre_data/npz_generate.py`.

### Oracle
Generate few-shot learning tasks.
```bash
$ cd Oracle
$ python oracle_make_task.py
```
To run the code, see examples Oracle/oracle_fmeta.py.

Automated training and test fuzzymeta4
```bash
$ python oracle_command.py
```

Or you can use specific commands like

```bash
CUDA_VISIBLE_DEVICES=2 python oracle_experiment.py --train=True --picshow=False --metatrain_itr=60000 --img_num=21 --meta_batch_size=16 --num_classes=20 --k_shot=1 --data_source=entire --lossf=cross_entropy --k_query=1 --inner_num=1 --model_name=fuzzymeta4
```