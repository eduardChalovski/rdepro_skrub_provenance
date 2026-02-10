$PipelineDir = "pipelines"

Write-Host "🚀 Running all pipelines WITHOUT provenance"
Write-Host "==========================================="

Get-ChildItem "$PipelineDir\*.py" | ForEach-Object {
    $module = $_.BaseName
    if ($module -eq "__init__") { return }

    Write-Host ""
    Write-Host "▶ python -m pipelines.$module"
    python -m pipelines.$module
}

Write-Host ""
Write-Host "🧬 Running all pipelines WITH provenance"
Write-Host "========================================"

Get-ChildItem "$PipelineDir\*.py" | ForEach-Object {
    $module = $_.BaseName
    if ($module -eq "__init__") { return }

    Write-Host ""
    Write-Host "▶ python -m pipelines.$module --track-provenance"
    python -m pipelines.$module --track-provenance
}

Write-Host ""
Write-Host "✅ All pipelines finished successfully"