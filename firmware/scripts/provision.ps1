param(
    [Parameter(Mandatory = $true)]
    [string]$Port,
    [string]$ConfigPath = (Join-Path $PSScriptRoot "..\..\Config\system_config.json"),
    [string]$NodeLabel,
    [int]$NodeId,
    [string]$WifiSsid,
    [string]$WifiPassword = "",
    [string]$TargetIp,
    [int]$TargetPort = 5005,
    [int]$WifiChannel = 6,
    [string]$ProjectRoot = (Join-Path $PSScriptRoot "..\esp32-csi-fingerprint-node")
)

$ErrorActionPreference = "Stop"

if (-not $env:IDF_PATH) {
    throw "Please load the ESP-IDF environment before running this script."
}

$resolvedNodeId = $NodeId
$resolvedWifiSsid = $WifiSsid
$resolvedWifiPassword = $WifiPassword
$resolvedTargetIp = $TargetIp
$resolvedTargetPort = $TargetPort
$resolvedWifiChannel = $WifiChannel

if ($ConfigPath -and (Test-Path $ConfigPath)) {
    $config = Get-Content $ConfigPath -Raw | ConvertFrom-Json

    if (-not $PSBoundParameters.ContainsKey('TargetIp') -and $config.host.target_ip) {
        $resolvedTargetIp = [string]$config.host.target_ip
    }

    if (-not $PSBoundParameters.ContainsKey('TargetPort') -and $null -ne $config.host.udp_port) {
        $resolvedTargetPort = [int]$config.host.udp_port
    }

    $selectedNode = $null
    if ($PSBoundParameters.ContainsKey('NodeId')) {
        $selectedNode = $config.nodes | Where-Object { [int]$_.node_id -eq $NodeId } | Select-Object -First 1
    } elseif ($PSBoundParameters.ContainsKey('NodeLabel')) {
        $selectedNode = $config.nodes | Where-Object { [string]$_.label -eq $NodeLabel } | Select-Object -First 1
    }

    if ($null -ne $selectedNode) {
        if (-not $PSBoundParameters.ContainsKey('NodeId')) {
            $resolvedNodeId = [int]$selectedNode.node_id
        }
        if (-not $PSBoundParameters.ContainsKey('WifiSsid') -and $selectedNode.wifi_ssid) {
            $resolvedWifiSsid = [string]$selectedNode.wifi_ssid
        }
        if (-not $PSBoundParameters.ContainsKey('WifiPassword') -and $selectedNode.wifi_password) {
            $resolvedWifiPassword = [string]$selectedNode.wifi_password
        }
        if (-not $PSBoundParameters.ContainsKey('WifiChannel') -and $null -ne $selectedNode.wifi_channel) {
            $resolvedWifiChannel = [int]$selectedNode.wifi_channel
        }
        if (-not $PSBoundParameters.ContainsKey('TargetIp') -and $config.host.target_ip) {
            $resolvedTargetIp = [string]$config.host.target_ip
        }
        if (-not $PSBoundParameters.ContainsKey('TargetPort') -and $null -ne $config.host.udp_port) {
            $resolvedTargetPort = [int]$config.host.udp_port
        }
    }
}

$missing = @()
if ([string]::IsNullOrWhiteSpace($resolvedWifiSsid)) { $missing += "WifiSsid" }
if ([string]::IsNullOrWhiteSpace($resolvedTargetIp)) { $missing += "TargetIp" }
if ($missing.Count -gt 0) {
    $missingList = $missing -join ", "
    throw "Missing required provisioning values: $missingList. Provide them manually or via -ConfigPath."
}

$ProjectRoot = (Resolve-Path $ProjectRoot).Path
$ProvisionDir = Join-Path $ProjectRoot "build\provisioning"
$CsvPath = Join-Path $ProvisionDir "csi_cfg.csv"
$BinPath = Join-Path $ProvisionDir "csi_cfg.bin"
$Generator = Join-Path $env:IDF_PATH "components\nvs_flash\nvs_partition_generator\nvs_partition_gen.py"

New-Item -ItemType Directory -Force -Path $ProvisionDir | Out-Null

$csv = @"
key,type,encoding,value
csi_cfg,namespace,,
ssid,data,string,$resolvedWifiSsid
password,data,string,$resolvedWifiPassword
target_ip,data,string,$resolvedTargetIp
target_port,data,u16,$resolvedTargetPort
node_id,data,u8,$resolvedNodeId
wifi_channel,data,u8,$resolvedWifiChannel
"@

Set-Content -Path $CsvPath -Value $csv -NoNewline

python $Generator generate --version 2 $CsvPath $BinPath 0x6000

Push-Location $ProjectRoot
try {
    python -m esptool --chip esp32s3 --port $Port --baud 460800 `
        write_flash --flash-mode dio --flash-freq 80m --flash-size 4MB `
        0x9000 $BinPath
}
finally {
    Pop-Location
}
