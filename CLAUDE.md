# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

This repository contains Netplan network configuration files for NVIDIA DGX servers. The configurations set up static IPs on QSFP network interfaces for NCCL (NVIDIA Collective Communications Library) GPU-to-GPU interconnects.

## Project Structure

- `10-qsfp.yaml` - DGX 1 network config (static IP: 10.0.0.1/24)
- `10-qsfp-dgx2.yaml` - DGX 2 network config (static IP: 10.0.0.2/24)

## Usage

These are Netplan YAML configuration files. To apply:
```bash
sudo netplan apply
```

To validate before applying:
```bash
sudo netplan generate
```