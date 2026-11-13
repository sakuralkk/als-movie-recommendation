#!/bin/bash
# 把 MovieLens 1M 数据从 Ubuntu 上传到 HDFS
# 前置：movie 数据已解压到 ~/ml-1m/
set -euo pipefail

DATA_DIR="${HOME}/ml-1m"
HDFS_DIR="/user/root/movielens"

if [ ! -d "$DATA_DIR" ]; then
  echo "未找到 $DATA_DIR，请先下载并解压 ml-1m.zip"
  exit 1
fi

echo "== 复制数据到 master 容器 =="
docker cp "$DATA_DIR/." master:/root/ml-1m/

echo "== 创建 HDFS 目录 =="
docker exec master hdfs dfs -mkdir -p "$HDFS_DIR"

echo "== 上传三个数据文件 =="
for f in ratings.dat movies.dat users.dat; do
  docker exec master hdfs dfs -put -f "/root/ml-1m/$f" "$HDFS_DIR/"
  echo "  $f 上传完成"
done

echo "== HDFS 目录内容 =="
docker exec master hdfs dfs -ls -h "$HDFS_DIR"
