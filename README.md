source /cvmfs/sft.cern.ch/lcg/views/LCG_100cuda/x86_64-centos7-gcc8-opt/setup.sh


# Issue faced

## First issue

```bash
2022-02-08 22:30:07.239716: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libcuda.so.1
2022-02-08 22:30:07.302914: E tensorflow/stream_executor/cuda/cuda_driver.cc:314] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
2022-02-08 22:30:07.303018: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (lxslc706.ihep.ac.cn): /proc/driver/nvidia/version does not exist
2022-02-08 22:30:07.306799: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-02-08 22:30:07.341548: I tensorflow/core/platform/profile_utils/cpu_utils.cc:104] CPU Frequency: 1997545000 Hz
2022-02-08 22:30:07.344088: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0xdd2cc20 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2022-02-08 22:30:07.344165: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2022-02-08 22:30:07.352060: F tensorflow/core/platform/default/env.cc:72] Check failed: ret == 0 (11 vs. 0)Thread creation via pthread_create() failed.
 *** Break *** abort
 *** Break *** segmentation violation
Segmentation fault (core dumped)
```
