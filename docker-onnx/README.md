# ABot-Recon torch-free ONNX 建图服务

onnxruntime + numpy(无 torch/无 open3d),镜像 ~2.2GB(torch 版 17.8GB)。与 torch 版数值对齐
(camera_poses cos=0.999997,world_points cos=0.9967)。

## 模型
导出图(tools/export_onnx_full.py)固定窗口 N,滑窗 + numpy 位姿组合推理:
- `abot_recon_n32.onnx` — CPU 部署用(窗大、窗数少);GPU 上单张注意力 34GB 跑不了。
- `abot_recon_n15.onnx` — **GPU 部署用**:注意力元素数 <2^31,绕过 ORT CUDA 的 int32/显存限制。
配套 `.onnx.data`(4GB 权重)+ `.json`(归一化 mean/std sidecar)。经 ModelScope
`zheqiushui/abot-recon-onnx` 中转。

## 跑法
CPU(稳,~140s/窗):
    ABOT_MODEL_ID=/app/model/abot_recon_n32.onnx ABOT_WARMUP=24 ORT_PROVIDERS=CPUExecutionProvider
GPU(GB10,~8s/窗,~17× 快;需自编译 sm_121 wheel,见 wheels/):
    ABOT_MODEL_ID=/app/model/abot_recon_n15.onnx ABOT_WARMUP=12 \
    ORT_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider ORT_OPT_LEVEL=basic ORT_CUDA_MEM_GB=90 \
    LD_LIBRARY_PATH=.../nvidia/cu13/lib:.../nvidia/cudnn/lib   (--gpus all)

## GPU 依赖(wheels/)
官方 onnxruntime-gpu aarch64 轮子在 GB10(sm_121)上要 JIT 且撞 cudnn/int32 限制。
自编译单架构 wheel:build.sh --use_cuda --cuda_home /usr/local/cuda-13.0
--cudnn_home <cudnn> --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=121。
把产物放 docker-onnx/wheels/ 后 build 会自动烤进镜像。
