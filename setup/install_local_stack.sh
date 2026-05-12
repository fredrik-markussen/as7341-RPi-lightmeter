#!/usr/bin/env bash
# Installs InfluxDB 1.12.4 + Grafana on the Pi, creates the 'lightmeter'
# database with a 90-day retention policy, and provisions the Grafana
# datasource. Safe to re-run — all steps are idempotent.
#
# Usage: sudo bash setup/install_local_stack.sh
#
# InfluxDB is installed from the upstream .deb release on dl.influxdata.com
# (no apt repo) — avoids the v1/v2 package-name churn in repos.influxdata.com.
# Grafana uses the official apt.grafana.com repo.

set -euo pipefail

INFLUX_VER="1.12.4"

# ── pre-flight ────────────────────────────────────────────────────────────────

if [[ $EUID -ne 0 ]]; then
    echo "Run as root:  sudo bash $0"
    exit 1
fi

if ! grep -qi "debian\|raspbian" /etc/os-release 2>/dev/null; then
    echo "This script targets Raspberry Pi OS / Debian. Aborting."
    exit 1
fi

ARCH=$(dpkg --print-architecture)
case "$ARCH" in
    arm64|amd64) ;;
    armhf)
        echo "ERROR: 32-bit ARM (armhf) is not supported by InfluxDB upstream .deb releases."
        echo "       Use 64-bit Raspberry Pi OS (Bookworm 64-bit). The 64-bit image works on"
        echo "       Pi 3 / Pi 4 / Pi 5 / Pi Zero 2 W."
        exit 1
        ;;
    *) echo "Unsupported architecture: $ARCH (need arm64 or amd64)"; exit 1 ;;
esac

APT_UPDATED=false
apt_update_once() {
    wait_for_apt
    if [[ $APT_UPDATED == false ]]; then
        apt-get update -q
        APT_UPDATED=true
    fi
}

# Block until no other apt/dpkg process is holding a lock.  Common offender on
# a fresh Pi is unattended-upgrades or apt.systemd.daily running in the
# background.  Times out after 10 min so a permanently-stuck lock is loud
# rather than silent.
wait_for_apt() {
    local max=600
    local waited=0
    while sudo fuser \
            /var/lib/dpkg/lock-frontend \
            /var/lib/dpkg/lock \
            /var/lib/apt/lists/lock \
            &>/dev/null; do
        if [[ $waited -eq 0 ]]; then
            echo "    Waiting for another apt/dpkg process to finish (e.g. unattended-upgrades) ..."
        fi
        if [[ $waited -ge $max ]]; then
            echo "ERROR: apt lock still held after ${max}s. Check:  ps aux | grep -E 'apt|dpkg'"
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
    if [[ $waited -gt 0 ]]; then
        echo "    apt lock released after ${waited}s, continuing."
    fi
}

# Clean up artefacts from earlier failed runs that used the apt-repo path.
rm -f /etc/apt/sources.list.d/influxdata.list \
      /etc/apt/sources.list.d/influxdb.list \
      /etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg

# ── InfluxDB ${INFLUX_VER} (direct .deb) ──────────────────────────────────────

echo "==> Installing InfluxDB ${INFLUX_VER} (${ARCH}) ..."

if ! dpkg -s influxdb &>/dev/null; then
    wait_for_apt
    apt-get install -y wget

    DEB_NAME="influxdb_${INFLUX_VER}-1_${ARCH}.deb"
    DEB_URL="https://dl.influxdata.com/influxdb/releases/v${INFLUX_VER}/${DEB_NAME}"

    echo "    Downloading ${DEB_URL}"
    wget -q --show-progress -O "/tmp/${DEB_NAME}" "${DEB_URL}"

    echo "    Installing /tmp/${DEB_NAME}"
    wait_for_apt
    dpkg -i "/tmp/${DEB_NAME}" || { wait_for_apt; apt-get install -fy; }

    rm -f "/tmp/${DEB_NAME}"
else
    echo "    influxdb already installed: $(dpkg -s influxdb | grep '^Version' | awk '{print $2}')"
fi

systemctl enable influxdb
systemctl start  influxdb

# Wait for the HTTP API to become reachable (up to 30 s).
echo "    Waiting for InfluxDB to be ready ..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8086/ping -o /dev/null; then
        echo "    InfluxDB is up."
        break
    fi
    sleep 1
    if [[ $i -eq 30 ]]; then
        echo "ERROR: InfluxDB did not start within 30 s. Check: journalctl -u influxdb"
        exit 1
    fi
done

# ── database and retention policy ─────────────────────────────────────────────

echo "==> Configuring 'lightmeter' database ..."

DB_EXISTS=$(influx -execute "SHOW DATABASES" 2>/dev/null | grep -c "^lightmeter$" || true)
if [[ $DB_EXISTS -eq 0 ]]; then
    influx -execute "CREATE DATABASE lightmeter"
    echo "    Created database 'lightmeter'."
else
    echo "    Database 'lightmeter' already exists."
fi

# Set 90-day default retention (adjust with ALTER RETENTION POLICY if needed).
RP_EXISTS=$(influx -database lightmeter -execute "SHOW RETENTION POLICIES" 2>/dev/null \
    | grep -c "^autogen" || true)
if [[ $RP_EXISTS -gt 0 ]]; then
    influx -execute "ALTER RETENTION POLICY autogen ON lightmeter DURATION 90d REPLICATION 1 DEFAULT" 2>/dev/null || true
    echo "    Retention policy: autogen 90 days."
fi

# ── Grafana (apt.grafana.com) ─────────────────────────────────────────────────

echo "==> Installing Grafana ..."

# Clean up artefacts from earlier failed runs.
rm -f /etc/apt/sources.list.d/grafana.list \
      /usr/share/keyrings/grafana.key \
      /usr/share/keyrings/grafana.gpg \
      /etc/apt/keyrings/grafana.gpg

if ! dpkg -s grafana &>/dev/null; then
    apt_update_once
    wait_for_apt
    # wget and gpg are usually pre-installed; install is a no-op if so.
    # apt-transport-https and software-properties-common are not needed —
    # we write sources.list.d directly and use only HTTPS for the repo.
    apt-get install -y wget gpg

    mkdir -p /etc/apt/keyrings
    wget -qO- https://apt.grafana.com/gpg.key \
        | gpg --dearmor > /etc/apt/keyrings/grafana.gpg
    chmod 644 /etc/apt/keyrings/grafana.gpg

    echo "deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com stable main" \
        > /etc/apt/sources.list.d/grafana.list

    wait_for_apt
    apt-get update -q
    wait_for_apt
    apt-get install -y grafana
else
    echo "    Grafana already installed: $(dpkg -s grafana | grep '^Version' | awk '{print $2}')"
fi

# ── Grafana datasource provisioning ──────────────────────────────────────────

echo "==> Provisioning Grafana datasource ..."

PROV_DIR="/etc/grafana/provisioning/datasources"
mkdir -p "$PROV_DIR"

cat > "$PROV_DIR/lightmeter.yml" <<'YAML'
apiVersion: 1
datasources:
  - name: lightmeter
    type: influxdb
    access: proxy
    url: http://localhost:8086
    database: lightmeter
    isDefault: true
    editable: true
YAML

echo "    Datasource config written to $PROV_DIR/lightmeter.yml"

systemctl enable grafana-server
systemctl restart grafana-server

# ── port notes ────────────────────────────────────────────────────────────────

echo ""
echo "==> Port summary (no firewall rules added by this script):"
echo "    InfluxDB  — localhost:8086  (line protocol write endpoint)"
echo "    Grafana   — 0.0.0.0:3000   (reachable on any interface, including hotspot)"
echo ""
echo "==> Done.  Access Grafana at:"
echo "    http://$(hostname).local:3000  (mDNS — works from phones on the same network)"
echo "    http://localhost:3000          (from the Pi itself)"
echo ""
echo "    First login: admin / admin  (Grafana will prompt you to change it)"
echo "    Datasource 'lightmeter' (InfluxDB, localhost:8086) is pre-configured."
echo ""
echo "==> InfluxDB write endpoint for .env:"
echo "    INFLUX_ENDPOINTS=[[\"127.0.0.1\",8086,\"lightmeter\"]]"
