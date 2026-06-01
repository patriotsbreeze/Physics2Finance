#!/usr/bin/env bash
# Download all required datasets for Physics2Finance.
#
# Usage:
#   bash scripts/download_data.sh [--pdearena] [--jhtdb] [--fi2010] [--binance]
#
# Flags default to ALL datasets if none specified.

set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"
PDEARENA="${PDEARENA:-0}"
JHTDB="${JHTDB:-0}"
FI2010="${FI2010:-0}"
BINANCE="${BINANCE:-0}"

# Parse flags
ALL=1
for arg in "$@"; do
  case $arg in
    --pdearena) PDEARENA=1; ALL=0 ;;
    --jhtdb)    JHTDB=1;    ALL=0 ;;
    --fi2010)   FI2010=1;   ALL=0 ;;
    --binance)  BINANCE=1;  ALL=0 ;;
  esac
done

if [ "$ALL" -eq 1 ]; then
  PDEARENA=1; JHTDB=1; FI2010=1; BINANCE=1
fi

mkdir -p "$DATA_DIR"/{pdearena,jhtdb,financial/{binance,fi2010}}

# ─── PDEArena ───────────────────────────────────────────────────────────────
if [ "$PDEARENA" -eq 1 ]; then
  echo "Downloading PDEArena datasets..."
  PDEARENA_DIR="$DATA_DIR/pdearena"

  # NavierStokes2D forced turbulence (primary training set)
  NS2D_URL="https://huggingface.co/datasets/pdearena/NavierStokes-2D/resolve/main"
  for split in train valid test; do
    mkdir -p "$PDEARENA_DIR/NavierStokes2D-ForcedTurbulence/$split"
    # Files are hosted on HuggingFace datasets
    echo "  Fetching NS2D $split split..."
    # Use huggingface-cli if available
    if command -v huggingface-cli &>/dev/null; then
      huggingface-cli download pdearena/NavierStokes-2D \
        --repo-type dataset \
        --local-dir "$PDEARENA_DIR/NavierStokes2D-ForcedTurbulence"
      break
    else
      echo "  Install huggingface-cli: pip install huggingface_hub[cli]"
      echo "  Then re-run: huggingface-cli download pdearena/NavierStokes-2D --repo-type dataset --local-dir $PDEARENA_DIR/NavierStokes2D-ForcedTurbulence"
      break
    fi
  done
fi

# ─── JHTDB ──────────────────────────────────────────────────────────────────
if [ "$JHTDB" -eq 1 ]; then
  echo "Downloading JHTDB isotropic turbulence cutout..."
  JHTDB_DIR="$DATA_DIR/jhtdb/isotropic1024coarse"
  mkdir -p "$JHTDB_DIR"

  # JHTDB cutout files require registration. This script downloads
  # the publicly available demo cutout.
  JHTDB_DEMO="http://turbulence.pha.jhu.edu/cutouts/sample_iso1024.h5"
  if [ ! -f "$JHTDB_DIR/sample_iso1024.h5" ]; then
    curl -L -o "$JHTDB_DIR/sample_iso1024.h5" "$JHTDB_DEMO" || \
      echo "  Failed to download JHTDB sample. Register at http://turbulence.pha.jhu.edu for full access."
  else
    echo "  JHTDB sample already downloaded."
  fi
fi

# ─── FI-2010 ────────────────────────────────────────────────────────────────
if [ "$FI2010" -eq 1 ]; then
  echo "Downloading FI-2010 LOB dataset..."
  FI2010_DIR="$DATA_DIR/financial/fi2010"
  mkdir -p "$FI2010_DIR"

  # FI-2010 is hosted on UCI ML Repository / the original paper's supplementary
  FI2010_URL="https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649"
  echo "  FI-2010 requires manual download from:"
  echo "  $FI2010_URL"
  echo "  Place the .txt files in: $FI2010_DIR/"
  echo ""
  echo "  Alternatively, use the Kaggle mirror:"
  echo "  kaggle datasets download -d mczielinski/bitcoin-historical-data"
fi

# ─── Binance ─────────────────────────────────────────────────────────────────
if [ "$BINANCE" -eq 1 ]; then
  echo "Downloading Binance BTC/USDT LOB data..."
  python3 -c "
from src.data.financial.binance_loader import BinanceDataDownloader
dl = BinanceDataDownloader('$DATA_DIR/financial/binance')
print('Downloading aggTrades (2023-01-01 to 2023-03-31)...')
dl.download_agg_trades('BTCUSDT', '2023-01-01', '2023-03-31')
print('Done.')
"
fi

echo ""
echo "Dataset download complete. Data directory: $DATA_DIR"
echo "Next step: python -m src.training.pretrain_fluid --config configs/pretrain_config.yaml"
