#!/usr/bin/env bash
set -e

PIPELINE_DIR="pipelines"

echo "🚀 Running all pipelines WITHOUT provenance"
echo "==========================================="

for file in ${PIPELINE_DIR}/*.py; do
  module=$(basename "$file" .py)

  if [[ "$module" == "__init__" ]]; then
    continue
  fi

  echo ""
  echo "▶ python -m pipelines.${module}"
  python -m pipelines.${module}
done

echo ""
echo "🧬 Running all pipelines WITH provenance"
echo "========================================"

for file in ${PIPELINE_DIR}/*.py; do
  module=$(basename "$file" .py)

  if [[ "$module" == "__init__" ]]; then
    continue
  fi

  echo ""
  echo "▶ python -m pipelines.${module} --track-provenance"
  python -m pipelines.${module} --track-provenance
done

echo ""
echo "✅ All pipelines finished successfully"
chmod +x run_all_pipelines.sh