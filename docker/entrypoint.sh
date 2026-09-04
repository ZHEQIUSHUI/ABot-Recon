#!/usr/bin/env bash
# ABot-Recon 服务入口:一个进程内同时起 FastAPI(建图 API)与 viser(网页 3D 可视化)。
# 模型(acvlab/ABot-Recon)首次用时自动从 HF 下载并热加载,无需挂载本地权重。
set -e
cd /app/docker

echo "== ABot-Recon 建图服务(热加载)=="
echo "   模型 : ${ABOT_MODEL_ID:-acvlab/ABot-Recon}  (首次用时自动下载到 HF 缓存)"
echo "   API  : http://0.0.0.0:${MAP_PORT:-8000}   (POST /jobs 提交视频;GET /status 看模型)"
echo "   3D网页: http://0.0.0.0:${VISER_PORT:-8080}  (加载进度/在线状态/看点云)"
echo "   空闲 ${MAP_IDLE_UNLOAD:-1800}s 自动卸载释放显存"
exec uvicorn service:app --host 0.0.0.0 --port "${MAP_PORT:-8000}"
