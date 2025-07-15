## CoQMoE: Co-Designed Quantization and Computation Orchestration for Mixture-of-Experts Vision Transformer on FPGA

## overview
This repository contains the implementation of Python codes for our paper *[CoQMoE: Co-Designed Quantization and Computation Orchestration for Mixture-of-Experts Vision Transformer on FPGA](https://arxiv.org/abs/2506.08496)*, which focuses on the quantization and computation orchestration of Mixture-of-Experts (MoE) Vision Transformers (ViTs) on FPGA. 

This work is based on our previous work *[UbiMoE: a Ubiquitous Mixture-of-Experts Vision Transformer Accelerator with Hybrid Computation Pattern on FPGA](https://arxiv.org/abs/2502.05602)*, which implements a fully streaming attention kernel optimized for latency and a reusable linear kernel optimized for resource efficiency.

Compared to UbiMoE, CoQMoE introduces a co-designed quantization and computation orchestration strategy that achieves a better trade-off between performance and resource utilization.

## Environment
- **Ubuntu 20.04**
- **Vitis**, **XRT** (Xilinx Runtime) and **XCU280 platform** 2022.2 [link](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/alveo/u280.html) 
- **Model and Dataset** : We use the [$M^3ViT$](https://github.com/VITA-Group/M3ViT) as our model and evaluate on [Cityscape Dataset](https://www.cityscapes-dataset.com/)

## Compile and Run
For this part, please refer to the latest fix16 version we provided in [UbiMoE](https://github.com/DJ000011/UbiMoE), which includes the complete end-to-end execution process, covering both the host program and the actual hardware code, as well as the compiled xclbin file that can be run directly.
