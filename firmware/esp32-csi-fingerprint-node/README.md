# ESP32 CSI Fingerprinting Node

This is a clean ESP-IDF firmware package for ESP32-S3-DevKitC-1. It connects
to WiFi, captures CSI, and streams ADR-018 frames over UDP for fingerprinting
and later multi-node expansion.

The project ships with `sdkconfig.defaults` so CSI support and the custom
partition table are enabled by default.

## Default behavior

- Target host: set at provisioning time
- UDP port: `5005`
- CSI frame format: ADR-018, unchanged from the existing node firmware
- Node ID: `1` by default, but configurable per device

## Build

From this directory, after loading the ESP-IDF environment:

```powershell
idf.py set-target esp32s3 build
```

## Flash

Use the helper script from `firmware/scripts`, or flash directly:

```powershell
idf.py -p COM7 flash
```

## Provision

Provisioning writes a small NVS image with WiFi credentials and the UDP target
for the current node. The firmware reads those values first and falls back to
Kconfig defaults when NVS is empty.

```powershell
..\scripts\provision.ps1 -Port COM7 -ConfigPath "..\..\Config\system_config.json" `
  -NodeId 1
..\scripts\provision.ps1 -Port COM7 -ConfigPath "..\..\Config\system_config.json" `
  -NodeLabel "ESP 1"
..\scripts\provision.ps1 -Port COM7 -WifiSsid "MyWiFi" -WifiPassword "secret" `
  -TargetIp "192.168.1.20" -TargetPort 5005 -NodeId 1 -WifiChannel 6
```

If you want a one-step firmware refresh, run build, flash, and provision in that
order. The scripts are intentionally separate so you can re-provision nodes
without rebuilding the app binary.

When `-ConfigPath` is used, the script pulls `host.target_ip`, `host.udp_port`,
and the selected node's `wifi_ssid`, `wifi_password`, and `wifi_channel` from
the shared JSON. You can select the node with either `-NodeId` or `-NodeLabel`,
and any explicit manual argument still wins over config values.

## Frame format

The firmware emits ADR-018 frames with:

- 4 byte magic
- 1 byte node ID
- 1 byte antenna count
- 2 byte subcarrier count
- 4 byte frequency in MHz
- 4 byte sequence number
- 1 byte RSSI
- 1 byte noise floor
- 2 reserved bytes
- interleaved I/Q CSI payload
