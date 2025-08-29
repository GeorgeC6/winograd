#!/usr/bin/env bash

# --- 颜色定义 ---
# 检查终端是否支持颜色（是否是交互式终端 TTY）
if [ -t 1 ]; then
    RESET="\e[0m"; BOLD="\e[1m";
    GREEN="\e[32m"; RED="\e[31m"; YELLOW="\e[33m"; CYAN="\e[36m";
else
    # 如果不是交互式终端（例如重定向到文件），则不使用颜色
    RESET=""; BOLD="";
    GREEN=""; RED=""; YELLOW=""; CYAN=""
fi

# --- 用法说明 ---
usage() {
    printf "👉 ${BOLD}Usage:${RESET} $0 <dest_dir>\n"
    printf "💡 ${CYAN}dest_dir${RESET}: Directory name to create and copy logs to\n"
    exit 1
}

# --- 解析命令行参数 ---
if [ $# -ne 1 ]; then
    printf "${BOLD}${RED}ERROR:${RESET} Please provide exactly one argument (destination directory)\n"
    usage
fi

DEST_DIR="$1"

# --- 验证参数 ---
if [ -z "$DEST_DIR" ]; then
    printf "${BOLD}${RED}ERROR:${RESET} Destination directory cannot be empty\n"
    usage
fi

# --- 主逻辑 ---
REMOTE_HOST="hpc101"
REMOTE_PATH="~/winograd"

# 在本地创建目标目录（如果它不存在的话）
printf "${BOLD}${CYAN}==>${RESET}${BOLD} Creating local directory: ${YELLOW}%s${RESET}\n" "$DEST_DIR"
mkdir -p "$DEST_DIR"

# 拷贝日志文件
printf "${BOLD}${CYAN}==>${RESET}${BOLD} Copying log files...${RESET}\n"

# 使用数组来存储文件模式，便于管理
FILE_PATTERNS=("*.log" "*.out" "*.err")

for pattern in "${FILE_PATTERNS[@]}"; do
    printf "  Copying ${pattern}...\n"
    scp -q "${REMOTE_HOST}:${REMOTE_PATH}/${pattern}" "$DEST_DIR/" 2>/dev/null || printf "    ⚠️${YELLOW}Warning:${RESET} No ${pattern} files found\n"
done

echo

printf "✅ ${BOLD}${GREEN}Copy complete!${RESET} Check the files in '${YELLOW}%s${RESET}'\n\n" "$DEST_DIR"
