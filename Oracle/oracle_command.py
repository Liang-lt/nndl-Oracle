import os

shot = 1
nc = 5
model_name = 'fuzzymeta4'
lossf = 'cross_entropy'
inner_num = 1

query = shot
ms = 32
# ms = 32 if nc == 5 else 16
''''''

# origin data
if os.system('CUDA_VISIBLE_DEVICES=0 python oracle_experiment.py --train=True --picshow=False --metatrain_itr=60000 --fd_shot=3 --img_num=3 --meta_batch_size=16 --num_classes=20 --k_shot=1 --data_source=origin --lossf=cross_entropy --k_query=1 --inner_num=1 --model_name=fuzzymeta4'):
    raise ValueError('Here1!')

if os.system('CUDA_VISIBLE_DEVICES=1 python oracle_experiment.py --train=True --picshow=False --metatrain_itr=60000 --fd_shot=5 --img_num=5 --meta_batch_size=16 --num_classes=20 --k_shot=1 --data_source=origin --lossf=cross_entropy --k_query=1 --inner_num=1 --model_name=fuzzymeta4'):
    raise ValueError('Here1!')


# entire data

if os.system('CUDA_VISIBLE_DEVICES=2 python oracle_experiment.py --train=True --picshow=False --metatrain_itr=60000 --img_num=21 --meta_batch_size=16 --num_classes=20 --k_shot=1 --data_source=entire --lossf=cross_entropy --k_query=1 --inner_num=1 --model_name=fuzzymeta4'):
    raise ValueError('Here1!')

if os.system('CUDA_VISIBLE_DEVICES=3 python oracle_experiment.py --train=True --picshow=False --metatrain_itr=60000 --img_num=63 --meta_batch_size=16 --num_classes=20 --k_shot=3 --data_source=entire --lossf=cross_entropy --k_query=3 --inner_num=1 --model_name=fuzzymeta4'):
    raise ValueError('Here1!')

if os.system('CUDA_VISIBLE_DEVICES=4 python oracle_experiment.py --train=True --picshow=False --metatrain_itr=60000 --img_num=105 --meta_batch_size=16 --num_classes=20 --k_shot=5 --data_source=entire --lossf=cross_entropy --k_query=5 --inner_num=1 --model_name=fuzzymeta4'):
    raise ValueError('Here1!')

# npz data

if os.system('CUDA_VISIBLE_DEVICES=5 python oracle_experiment.py --train=True --picshow=False --metatrain_itr=60000 --img_num=41 --meta_batch_size=16 --num_classes=20 --k_shot=1 --data_source=npz --lossf=cross_entropy --k_query=1 --inner_num=1 --model_name=fuzzymeta4'):
    raise ValueError('Here1!')

if os.system('CUDA_VISIBLE_DEVICES=6 python oracle_experiment.py --train=True --picshow=False --metatrain_itr=60000 --img_num=123 --meta_batch_size=16 --num_classes=20 --k_shot=3 --data_source=npz --lossf=cross_entropy --k_query=3 --inner_num=1 --model_name=fuzzymeta4'):
    raise ValueError('Here1!')

if os.system('CUDA_VISIBLE_DEVICES=7 python oracle_experiment.py --train=True --picshow=False --metatrain_itr=60000 --img_num=205 --meta_batch_size=16 --num_classes=20 --k_shot=5 --data_source=npz --lossf=cross_entropy --k_query=5 --inner_num=1 --model_name=fuzzymeta4'):
    raise ValueError('Here1!')



# npz data 200ways

if os.system('CUDA_VISIBLE_DEVICES=0 python oracle_experiment.py --train=True --picshow=False --metatrain_itr=60000 --img_num=41 --meta_batch_size=1 --num_classes=200 --k_shot=1 --data_source=npz --lossf=cross_entropy --k_query=1 --inner_num=1 --model_name=fuzzymeta4'):
    raise ValueError('Here1!')

if os.system('CUDA_VISIBLE_DEVICES=1 python oracle_experiment.py --train=True --picshow=False --metatrain_itr=60000 --img_num=123 --meta_batch_size=4 --num_classes=200 --k_shot=3 --data_source=npz --lossf=cross_entropy --k_query=3 --inner_num=1 --model_name=fuzzymeta4'):
    raise ValueError('Here1!')

if os.system('CUDA_VISIBLE_DEVICES=2 python oracle_experiment.py --train=True --picshow=False --metatrain_itr=60000 --img_num=205 --meta_batch_size=4 --num_classes=200 --k_shot=5 --data_source=npz --lossf=cross_entropy --k_query=5 --inner_num=1 --model_name=fuzzymeta4'):
    raise ValueError('Here1!')