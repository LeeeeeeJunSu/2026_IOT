# CSI Fingerprinting Workspace

This workspace is a clean, separated codebase for estimating a person's room
location from ESP32 CSI using a grid-based fingerprinting workflow.

## Structure

- `Config/` - Shared settings for host IP, UDP port, grid size, fingerprinting,
  simulation, and node provisioning
- `firmware/` - ESP32-S3 firmware plus build, flash, and provision helpers
- `app/` - Tkinter desktop app for cell learning and probability visualization
- `simulator/` - ESP32 CSI traffic simulator for hardware-free testing

## Single Source Of Truth

Edit `Config/system_config.json`
first when you want to change:

- the UDP host IP and port
- the grid size
- capture defaults
- simulator behavior
- per-node IDs, WiFi, and COM port mapping

The goal is to avoid scattering initial setup values across multiple scripts.
