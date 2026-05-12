#!/usr/bin/env bash
# Installs InfluxDB 1.8 + Grafana on the Pi, creates the 'lightmeter'
# database with a 90-day retention policy, and provisions the Grafana
# datasource. Safe to re-run — all steps are idempotent.
#
# Usage: sudo bash setup/install_local_stack.sh

set -euo pipefail

# ── pre-flight ────────────────────────────────────────────────────────────────

if [[ $EUID -ne 0 ]]; then
    echo "Run as root:  sudo bash $0"
    exit 1
fi

if ! grep -qi "debian\|raspbian" /etc/os-release 2>/dev/null; then
    echo "This script targets Raspberry Pi OS / Debian. Aborting."
    exit 1
fi

APT_UPDATED=false
apt_update_once() {
    if [[ $APT_UPDATED == false ]]; then
        apt-get update -q
        APT_UPDATED=true
    fi
}

# ── InfluxDB 1.8 ──────────────────────────────────────────────────────────────

echo "==> Installing InfluxDB 1.8 ..."

if ! dpkg -s influxdb &>/dev/null; then
    apt_update_once
    apt-get install -y curl gnupg apt-transport-https

    curl -fsSL https://repos.influxdata.com/influxdata-archive_compat.key \
        | gpg --dearmor -o /etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg

    echo "deb https://repos.influxdata.com/debian stable main" \
        > /etc/apt/sources.list.d/influxdata.list

    apt-get update -q
    # Pin to 1.8.x — the influxdb2 package is a separate name in this repo.
    apt-get install -y "influxdb=1.8.*"
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

# ── Grafana ───────────────────────────────────────────────────────────────────

echo "==> Installing Grafana ..."

if ! dpkg -s grafana &>/dev/null; then
    apt_update_once
    apt-get install -y wget software-properties-common

    wget -qO /usr/share/keyrings/grafana.key https://apt.grafana.com/gpg.key

    echo "deb [signed-by=/usr/share/keyrings/grafana.key] https://apt.grafana.com stable main" \
        > /etc/apt/sources.list.d/grafana.list

    apt-get update -q
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
