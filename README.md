# vit-cifar10
A correctness test for ViT in Cifar10. Just type below commands to run it.

```shell
export DATA=<absolute path where you store cifar10 data>
torchrun --nproc_per_node=4 --master_port=19198 trainv1.py --config config.py
```
