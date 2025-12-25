#!/bin/bash
# 调试模式启动脚本

# 设置调试环境变量
export DEBUG_MODE=true
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# 创建调试日志目录
DEBUG_LOG_DIR="debug_logs"
mkdir -p $DEBUG_LOG_DIR

# 生成调试日志文件名
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEBUG_LOG_FILE="${DEBUG_LOG_DIR}/debug_${TIMESTAMP}.log"

echo "================================"
echo "  金融分析系统 - 调试模式"
echo "================================"
echo "调试日志: $DEBUG_LOG_FILE"
echo "调试模式: 已启用"
echo "================================"
echo ""

# 运行程序并同时输出到文件和控制台
python src/main.py "$@" 2>&1 | tee $DEBUG_LOG_FILE

echo ""
echo "================================"
echo "调试日志已保存到: $DEBUG_LOG_FILE"
echo "================================"
