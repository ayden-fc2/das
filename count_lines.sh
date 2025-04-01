#!/bin/bash

# 代码行数统计脚本
# 排除构建目录、依赖目录和配置文件

# 定义颜色代码
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查是否安装了cloc
if ! command -v cloc &> /dev/null; then
    echo -e "${RED}错误：cloc 未安装。${NC}"
    echo -e "请使用以下命令安装："
    echo -e "  macOS: ${YELLOW}brew install cloc${NC}"
    echo -e "  Linux: ${YELLOW}sudo apt-get install cloc${NC}"
    echo -e "  Windows(Chocolatey): ${YELLOW}choco install cloc${NC}"
    exit 1
fi

# 目标目录（默认为当前目录）
TARGET_DIR=${1:-.}

echo -e "${GREEN}开始统计代码行数...${NC}"
echo -e "目标目录: ${YELLOW}${TARGET_DIR}${NC}"
echo -e "排除目录: target, .next, node_modules, static, chunks, dist, compiled"
echo -e "排除文件类型: .json, .lock"

# 执行cloc命令
cloc \
    --exclude-dir=target,.next,node_modules,static,chunks,dist,compiled \
    --exclude-ext=json,lock \
    --fullpath \
    --not-match-f="(\.next|\.swc|\.cache|\.git)" \
    "${TARGET_DIR}"

# 检查cloc执行结果
if [ $? -eq 0 ]; then
    echo -e "${GREEN}统计完成！${NC}"
else
    echo -e "${RED}统计过程中出现错误。${NC}"
    exit 1
fi
