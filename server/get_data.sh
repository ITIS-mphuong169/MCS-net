#!/usr/bin/env bash
# Buoc 3: tai WikiArt tu Kaggle (cung bo steubk/wikiart -> cung tap test voi cac run truoc).
# Can file ~/.kaggle/kaggle.json (kaggle.com -> Settings -> API -> Create New Token).
set -e
ENV_DIR=${ENV_DIR:-$HOME/mcs-venv}
DATA_ROOT=${DATA_ROOT:-$HOME/data}
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "!! Thieu ~/.kaggle/kaggle.json. Lam:  mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
  exit 1
fi
mkdir -p "$DATA_ROOT" && cd "$DATA_ROOT"
if [ -z "$(find "$DATA_ROOT" -maxdepth 4 -type d -name Impressionism 2>/dev/null | head -1)" ]; then
  [ -f wikiart.zip ] || "$ENV_DIR/bin/kaggle" datasets download -d steubk/wikiart
  echo ">> Giai nen (lau)..."
  unzip -q -o wikiart.zip -d wikiart && rm -f wikiart.zip
fi
D=$(dirname "$(find "$DATA_ROOT" -maxdepth 4 -type d -name Impressionism | head -1)")
N=$(find "$D" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "$D" > "$HOME/.mcs_data_dir"
echo ">> DATA_DIR = $D  ($N thu muc phong cach, phai la 27)"
