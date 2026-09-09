#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd -- "$APP_DIR/.." && pwd)"

if [[ -f "$APP_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$APP_DIR/.env"
  set +a
fi

if [[ -x "$APP_DIR/.venv/bin/python" ]]; then
  PYTHON="$APP_DIR/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

export PYTHONPATH="$APP_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

run_assistant() {
  (cd "$APP_DIR" && "$PYTHON" -m personal_assistant "$@")
}

pause_screen() {
  printf '\n按 Enter 返回主菜单...'
  read -r
}

clear_screen() {
  command -v clear >/dev/null 2>&1 && clear || true
}

chat_once() {
  local message
  read -r -p '请输入问题: ' message
  [[ -z "$message" ]] && return
  run_assistant chat "$message" || true
  pause_screen
}

route_query() {
  local message
  read -r -p '请输入要分析的问题: ' message
  [[ -z "$message" ]] && return
  run_assistant route "$message" || true
  pause_screen
}

index_file() {
  local path
  read -r -e -p '请输入 Markdown/TXT/PDF 文件路径: ' path
  [[ -z "$path" ]] && return
  if [[ ! -f "$path" ]]; then
    echo "文件不存在: $path"
  else
    run_assistant index "$path" || true
  fi
  pause_screen
}

show_menu() {
  clear_screen
  cat <<'EOF'
========================================
       Personal Assistant Bash UI
========================================
1) 单次对话（使用 .env 配置）
2) 进入终端 REPL
3) 查看问题路由
4) 导入文档到 Milvus
5) 检查 PostgreSQL / Milvus / Neo4j
6) 初始化开发数据库
7) 查看系统架构
0) 退出
========================================
EOF
}

main() {
  local choice
  while true; do
    show_menu
    read -r -p '请选择 [0-7]: ' choice
    case "$choice" in
      1) chat_once ;;
      2) run_assistant repl || true; pause_screen ;;
      3) route_query ;;
      4) index_file ;;
      5) run_assistant health || true; pause_screen ;;
      6) run_assistant db-init || true; pause_screen ;;
      7) run_assistant architecture; pause_screen ;;
      0) echo '再见。'; exit 0 ;;
      *) echo '无效选项。'; sleep 1 ;;
    esac
  done
}

cd "$PROJECT_ROOT"
main "$@"
