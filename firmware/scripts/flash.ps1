param(
    [Parameter(Mandatory = $true)]
    [string]$Port,
    [string]$ProjectRoot = (Join-Path $PSScriptRoot "..\esp32-csi-fingerprint-node")
)

$ErrorActionPreference = "Stop"

if (-not $env:IDF_PATH) {
    throw "Please load the ESP-IDF environment before running this script."
}

$ProjectRoot = (Resolve-Path $ProjectRoot).Path
Push-Location $ProjectRoot
try {
    & idf.py -p $Port flash
}
finally {
    Pop-Location
}

