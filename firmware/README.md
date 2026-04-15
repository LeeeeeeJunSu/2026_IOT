# Firmware

This subtree contains the ESP32-S3 CSI fingerprinting firmware package and the
host-side helper scripts used to build, flash, and provision each node.

- `esp32-csi-fingerprint-node/` - ESP-IDF project for ESP32-S3-DevKitC-1
- `scripts/` - PowerShell helpers for build, flash, and NVS provisioning

The firmware keeps the ADR-018 UDP CSI frame format and sends to UDP port
`5005` by default. The desktop app can listen on the same port without any
extra firmware changes.

The provisioning script can also read shared values from
`Config/system_config.json` when you pass `-ConfigPath`. That lets you source
the desktop host IP/port and per-node WiFi settings from the same config that
drives the app.
