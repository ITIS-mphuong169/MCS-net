#!/usr/bin/env bash
# Buoc 1: in thong tin may. Chup lai ket qua gui di neu co loi.
echo "== He dieu hanh"; uname -sr; grep PRETTY_NAME /etc/os-release 2>/dev/null
echo; echo "== GPU"
if command -v nvidia-smi >/dev/null; then nvidia-smi
elif command -v rocm-smi >/dev/null; then rocm-smi
else echo "!! Khong thay nvidia-smi hay rocm-smi (may khong co GPU, hoac chua cai driver)"; fi
echo; echo "== RAM"; free -h
echo; echo "== O dia (can ~65GB trong)"; df -h "$HOME"
echo; echo "== So CPU"; nproc
echo; echo "== Cong cu"
for t in python3 conda git tmux screen unzip curl; do printf "%-8s " "$t"; command -v "$t" || echo "(khong co)"; done
python3 --version 2>/dev/null
echo; echo "== Internet"
curl -sI --max-time 10 https://github.com >/dev/null && echo "github: OK" || echo "github: KHONG truy cap duoc"
curl -sI --max-time 10 https://www.kaggle.com >/dev/null && echo "kaggle: OK" || echo "kaggle: KHONG truy cap duoc"
curl -sI --max-time 10 https://download.pytorch.org >/dev/null && echo "pytorch: OK" || echo "pytorch: KHONG truy cap duoc"
