#!/usr/bin/env bash
# Adds or updates a NetworkManager Wi-Fi profile so the Pi auto-connects to a
# phone hotspot. Requires Raspberry Pi OS Bookworm (or any OS using
# NetworkManager / nmcli).  For Bullseye or earlier see README.
#
# Usage:
#   sudo bash setup/configure_hotspot.sh "<SSID>" "<password>"
#   sudo bash setup/configure_hotspot.sh "<SSID>" "<password>" --interface wlan1
#   sudo bash setup/configure_hotspot.sh "<SSID>" "<password>" --static-ip 192.168.43.50/24
#   sudo bash setup/configure_hotspot.sh "<SSID>" "<password>" --static-ip 192.168.43.50/24 --gateway 192.168.43.1
#
# By default the script autodetects the first NetworkManager-managed wifi
# interface (handles Pis with a USB wifi adapter where the built-in wlan0 is
# left unmanaged). Override with --interface if you have multiple managed
# wifi interfaces and want a specific one.

set -euo pipefail

CON_NAME="lightmeter-hotspot"
STATIC_IP=""
GW=""
IFACE=""

usage() {
    echo "Usage: $0 <SSID> <password> [--interface <wlanN>] [--static-ip <addr/prefix>] [--gateway <ip>]"
    exit 1
}

[[ $# -lt 2 ]] && usage

SSID="$1"
PASS="$2"
shift 2

while [[ $# -gt 0 ]]; do
    case "$1" in
        --interface) IFACE="$2";    shift 2 ;;
        --static-ip) STATIC_IP="$2"; shift 2 ;;
        --gateway)   GW="$2";        shift 2 ;;
        *) usage ;;
    esac
done

# ── sanity checks ─────────────────────────────────────────────────────────────

if [[ $EUID -ne 0 ]]; then
    echo "Run as root:  sudo bash $0 \"$SSID\" \"$PASS\""
    exit 1
fi

if ! command -v nmcli &>/dev/null; then
    echo "ERROR: nmcli not found. Is NetworkManager installed?"
    echo "  Raspberry Pi OS Bookworm uses NetworkManager by default."
    echo "  For Bullseye/Buster, add the hotspot to /etc/wpa_supplicant/wpa_supplicant.conf instead."
    exit 1
fi

# ── pick a managed wifi interface ────────────────────────────────────────────

if [[ -z "$IFACE" ]]; then
    # First wifi device whose STATE is NOT 'unmanaged'.  Catches the common
    # case where wlan0 (built-in) is unmanaged and a USB wifi adapter on
    # wlan1 is the active radio.
    IFACE=$(nmcli -t -f DEVICE,TYPE,STATE device \
        | awk -F: '$2=="wifi" && $3!="unmanaged" {print $1; exit}')
    if [[ -z "$IFACE" ]]; then
        echo "ERROR: no NetworkManager-managed wifi interface found."
        echo "       Available wifi devices and their state:"
        nmcli -f DEVICE,TYPE,STATE device | grep -E "DEVICE|wifi"
        echo "       Pass --interface <name> if you know which to use."
        exit 1
    fi
    echo "Detected managed wifi interface: $IFACE"
else
    echo "Using requested interface: $IFACE"
fi

# ── create or update profile ──────────────────────────────────────────────────

if nmcli connection show "$CON_NAME" &>/dev/null; then
    echo "Updating existing profile '$CON_NAME' ..."
    nmcli connection modify "$CON_NAME" connection.interface-name "$IFACE"
    nmcli connection modify "$CON_NAME" 802-11-wireless.ssid "$SSID"
    nmcli connection modify "$CON_NAME" wifi-sec.key-mgmt wpa-psk
    nmcli connection modify "$CON_NAME" wifi-sec.psk "$PASS"
else
    echo "Creating new profile '$CON_NAME' on $IFACE ..."
    nmcli connection add \
        type wifi \
        ifname "$IFACE" \
        con-name "$CON_NAME" \
        ssid "$SSID" \
        wifi-sec.key-mgmt wpa-psk \
        wifi-sec.psk "$PASS"
fi

nmcli connection modify "$CON_NAME" connection.autoconnect yes
nmcli connection modify "$CON_NAME" connection.autoconnect-priority 50

# ── static IP (optional) ──────────────────────────────────────────────────────

if [[ -n "$STATIC_IP" ]]; then
    # Derive a sensible gateway default from the address if not given.
    if [[ -z "$GW" ]]; then
        GW=$(echo "$STATIC_IP" | sed 's|\.[0-9]*/.*|.1|')
    fi
    nmcli connection modify "$CON_NAME" ipv4.method manual
    nmcli connection modify "$CON_NAME" ipv4.addresses "$STATIC_IP"
    nmcli connection modify "$CON_NAME" ipv4.gateway "$GW"
    nmcli connection modify "$CON_NAME" ipv4.dns "8.8.8.8"
    echo "Static IP:  $STATIC_IP  (gateway: $GW)"
else
    nmcli connection modify "$CON_NAME" ipv4.method auto
fi

# ── summary ───────────────────────────────────────────────────────────────────

HOSTNAME=$(hostname)
echo ""
echo "Profile '$CON_NAME' is ready."
echo "The Pi will auto-connect when '$SSID' is in range (priority 50)."
echo ""
echo "Reach Grafana from your phone once connected:"
echo "  http://${HOSTNAME}.local:3000    (mDNS — preferred)"
if [[ -n "$STATIC_IP" ]]; then
    BARE_IP=$(echo "$STATIC_IP" | cut -d/ -f1)
    echo "  http://${BARE_IP}:3000          (static IP fallback)"
fi
echo ""
echo "Note: 'raspberrypi.local' may not resolve on older Android devices."
echo "      Use the static IP in that case (pass --static-ip to this script)."
echo ""
echo "The measurement script keeps writing to local InfluxDB regardless of"
echo "hotspot state — the hotspot is only needed for live Grafana access."
