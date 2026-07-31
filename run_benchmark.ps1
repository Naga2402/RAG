# JISR — full Ragas benchmark (Agentic vs Naive) over the complete golden set.
#
#   .\run_benchmark.ps1
#
# Expect roughly 3-4 hours. It is fully unattended and safe to leave running;
# progress is written to results\benchmark_full.log as it goes.
# Outputs:
#   results\benchmark.csv            aggregate metrics per system
#   results\per_question_*.csv       per-question Ragas scores
#   results\raw_answers.json         every answer + latency (dissertation appendix)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "== JISR full benchmark ==" -ForegroundColor Cyan

# 1. vector database
Write-Host "[1/3] starting pgvector container..."
docker compose up -d | Out-Null
$ok = $false
foreach ($i in 1..30) {
    $s = (docker inspect -f '{{.State.Health.Status}}' jisr-pg 2>$null)
    if ($s -eq 'healthy') { $ok = $true; break }
    Start-Sleep -Seconds 3
}
if (-not $ok) { Write-Host "  pgvector did not become healthy - is Docker Desktop running?" -ForegroundColor Red; exit 1 }
Write-Host "  pgvector healthy." -ForegroundColor Green

# 2. models
Write-Host "[2/3] checking Ollama models..."
$ollama = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
$have = & $ollama list
foreach ($m in @('llama3:8b', 'jais-chat', 'nomic-embed-text')) {
    if ($have -notmatch [regex]::Escape($m)) { Write-Host "  MISSING model: $m" -ForegroundColor Red; exit 1 }
}
Write-Host "  models present." -ForegroundColor Green

# 3. benchmark
Write-Host "[3/3] running benchmark (this takes ~3-4 hours)..." -ForegroundColor Yellow
$env:PYTHONIOENCODING = "utf-8"
$start = Get-Date
& ".\.venv\Scripts\python.exe" -m src.eval.run *>&1 |
    Tee-Object -FilePath "results\benchmark_full.log"
$mins = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)

Write-Host "`n== finished in $mins min ==" -ForegroundColor Cyan
if (Test-Path "results\benchmark.csv") {
    Write-Host "`nRESULTS:" -ForegroundColor Green
    Get-Content "results\benchmark.csv"
} else {
    Write-Host "No benchmark.csv - check results\benchmark_full.log" -ForegroundColor Red
}
