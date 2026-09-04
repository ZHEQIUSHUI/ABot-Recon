#!/usr/bin/env bash
# ABot-Recon torch-free ONNX 服务入口:FastAPI(建图 API)+ viser(网页 3D)。
# 模型 = 导出的 abot_recon ONNX 图(挂载进来,ABOT_MODEL_ID 指向 .onnx 或其目录)。
set -e
cd /app/docker-onnx

echo "== ABot-Recon ONNX 建图服务(torch-free)=="
echo "   模型 : ${ABOT_MODEL_ID:-/app/model/abot_recon.onnx}"
echo "   EP   : ${ORT_PROVIDERS:-CPUExecutionProvider}"
echo "   API  : http://0.0.0.0:${MAP_PORT:-8000}   (POST /jobs 提交视频)"
echo "   3D网页: http://0.0.0.0:${VISER_PORT:-8080}"
exec uvicorn service:app --host 0.0.0.0 --port "${MAP_PORT:-8000}"
