# start.ps1 — Arranca Pentamodal de forma estable e idempotente.
# Si ya esta corriendo, lo reporta; si no, libera el puerto 8000, arranca y verifica.
$ErrorActionPreference = "SilentlyContinue"
$base = "http://127.0.0.1:8000"

# 1. Ya esta corriendo?
try {
    $r = Invoke-WebRequest -Uri "$base/api/health" -TimeoutSec 3 -UseBasicParsing
    Write-Output "Ya estaba corriendo: HTTP $($r.StatusCode) en $base"
    exit 0
} catch {}

# 2. Liberar el puerto 8000
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 1

# 3. Arrancar (en su carpeta, con logs)
Set-Location "C:\Users\vi_ch\pentamodal-app"
if (-not (Test-Path "logs")) { New-Item -ItemType Directory "logs" | Out-Null }
Start-Process -FilePath "python" -ArgumentList "app.py" -NoNewWindow `
    -RedirectStandardOutput "logs\out.txt" -RedirectStandardError "logs\err.txt"

# 4. Esperar y verificar (hasta ~45s; el lifespan inicia 7 BBDD + autodetecta Ollama)
for ($i = 1; $i -le 45; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-WebRequest -Uri "$base/api/health" -TimeoutSec 3 -UseBasicParsing
        Write-Output "Arrancado OK: HTTP $($r.StatusCode) en $base"
        exit 0
    } catch {}
}
Write-Output "No respondio tras 45s. Revisa logs\err.txt"
exit 1
