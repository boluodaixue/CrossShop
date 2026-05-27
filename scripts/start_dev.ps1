param(
    [int]$Port = 8000,
    [switch]$SkipFrontend,
    [switch]$NoInstall
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$OutputDir = Join-Path $Root "output"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "找不到虚拟环境 Python：$Python"
}

if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "端口 $Port 已被占用，请先停止已有服务或改用 -Port 参数"
}

function Stop-ProcessTree {
    param($Process)
    if ($null -eq $Process -or $Process.HasExited) {
        return
    }
    & taskkill.exe /PID $Process.Id /T /F 2>$null | Out-Null
}

$BackendLog = Join-Path $OutputDir "backend.stdout.log"
$BackendErr = Join-Path $OutputDir "backend.stderr.log"

Write-Host "== 启动后端 =="
$backend = Start-Process `
    -FilePath $Python `
    -ArgumentList @("-m", "uvicorn", "globex_agent.presentation.server:app", "--port", "$Port") `
    -WorkingDirectory $Root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $BackendLog `
    -RedirectStandardError $BackendErr `
    -PassThru

$health = $null
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 1
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2
        break
    } catch {
        # 服务还没就绪，继续等待
    }
}

if ($null -eq $health) {
    Write-Host "后端启动失败，最近日志："
    if (Test-Path $BackendLog) {
        Get-Content $BackendLog -Tail 30
    }
    if (Test-Path $BackendErr) {
        Get-Content $BackendErr -Tail 30
    }
    Stop-ProcessTree $backend
    exit 1
}

Write-Host "后端已就绪：http://127.0.0.1:$Port/health"

$frontend = $null
$FrontendPkg = Join-Path $Root "frontend\package.json"
if (-not $SkipFrontend -and (Test-Path $FrontendPkg)) {
    $FrontendDir = Join-Path $Root "frontend"
    $NodeModules = Join-Path $FrontendDir "node_modules"
    if (-not $NoInstall -and -not (Test-Path $NodeModules)) {
        Write-Host "== 安装前端依赖 =="
        Push-Location $FrontendDir
        try {
            & npm.cmd install --no-audit --no-fund
            if ($LASTEXITCODE -ne 0) {
                throw "npm install 失败"
            }
        } finally {
            Pop-Location
        }
    }

    Write-Host "== 启动前端 =="
    $FrontendLog = Join-Path $OutputDir "frontend.stdout.log"
    $FrontendErr = Join-Path $OutputDir "frontend.stderr.log"
    $frontend = Start-Process `
        -FilePath "npm.cmd" `
        -ArgumentList @("run", "dev") `
        -WorkingDirectory $FrontendDir `
        -WindowStyle Hidden `
        -RedirectStandardOutput $FrontendLog `
        -RedirectStandardError $FrontendErr `
        -PassThru
    Start-Sleep -Seconds 5
    Write-Host "前端已启动：http://localhost:5173"
}

Write-Host ""
Write-Host "后端 PID：$($backend.Id)"
if ($frontend) {
    Write-Host "前端 PID：$($frontend.Id)"
}
Write-Host "日志目录：$OutputDir"
Write-Host "按 Ctrl+C 停止全部服务..."

try {
    while ($true) {
        Start-Sleep -Seconds 2
        if ($backend.HasExited) {
            Write-Host "后端进程已退出，停止脚本。"
            break
        }
    }
} finally {
    Stop-ProcessTree $frontend
    Stop-ProcessTree $backend
    Write-Host "服务已停止。"
}
