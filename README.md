# Fully-Integerized-End-to-End-LIC

## Pytorch
Codes for the proposed fully-integeried method on two state-of-the-art end-to-end LIC models, which are the scale hyperprior model and the autoregressive model.
This part includes three steps, which are named as pre_int32 (quantization and activation equalization), int32 (integerization and internal bit width increment) and OCS.

## TVM
Fully-integerized LIC model implemented with TVM Relay and built results are saved in so file format.

## Tensorrt
Fully-integerized LIC model deployed based on TensorRT.
