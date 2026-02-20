#!/usr/bin/env bash
# =============================================================================
# Generate self-signed CA, server, and client certs for Mosquitto mTLS
# =============================================================================
# Output: docker/mosquitto/certs/
#   - ca.crt, ca.key       CA (keep ca.key private)
#   - server.crt, server.key  Broker (mount into Mosquitto container)
#   - client.crt, client.key  Client (for MQTT clients using mTLS)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CERT_DIR="${CERT_DIR:-$REPO_ROOT/docker/mosquitto/certs}"
DAYS="${DAYS:-3650}"

echo "=== PDM Pilot — MQTT certificate generation ==="
echo "Output directory: $CERT_DIR"
echo "Validity: $DAYS days"
echo ""

mkdir -p "$CERT_DIR"
cd "$CERT_DIR"

# -----------------------------------------------------------------------------
# 1. CA (Certificate Authority)
# -----------------------------------------------------------------------------
echo "[1/3] Generating CA..."
openssl genrsa -out ca.key 4096
openssl req -x509 -new -nodes -key ca.key -sha256 -days "$DAYS" -out ca.crt \
  -subj "/C=US/ST=State/L=City/O=PDM-Pilot/OU=MQTT/CN=PDM-MQTT-CA"

# -----------------------------------------------------------------------------
# 2. Server certificate (Mosquitto broker) with SAN
# -----------------------------------------------------------------------------
echo "[2/3] Generating server certificate..."
openssl genrsa -out server.key 2048
openssl req -new -key server.key -out server.csr \
  -subj "/C=US/ST=State/L=City/O=PDM-Pilot/OU=MQTT/CN=mosquitto"

# Sign server CSR with SAN for broker hostnames
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
  -out server.crt -days "$DAYS" -sha256 \
  -extfile <(printf "subjectAltName=DNS:localhost,DNS:mosquitto,DNS:*.mosquitto,IP:127.0.0.1\nextendedKeyUsage=serverAuth")

rm -f server.csr

# -----------------------------------------------------------------------------
# 3. Client certificate (for MQTT clients, mTLS)
# -----------------------------------------------------------------------------
echo "[3/3] Generating client certificate..."
openssl genrsa -out client.key 2048
openssl req -new -key client.key -out client.csr \
  -subj "/C=US/ST=State/L=City/O=PDM-Pilot/OU=MQTT/CN=pdm-mqtt-client"

openssl x509 -req -in client.csr -CA ca.crt -CAkey ca.key \
  -out client.crt -days "$DAYS" -sha256 \
  -extfile <(printf "extendedKeyUsage=clientAuth\n")

rm -f client.csr

# -----------------------------------------------------------------------------
# Permissions and cleanup
# -----------------------------------------------------------------------------
chmod 600 ca.key server.key client.key
chmod 644 ca.crt server.crt client.crt
rm -f ca.srl 2>/dev/null || true

echo ""
echo "=== Done ==="
echo "Broker (Mosquitto):  server.crt, server.key, ca.crt"
echo "Clients (mTLS):      client.crt, client.key, ca.crt"
echo ""
echo "TLS ports (configure in mosquitto.conf): 8883 (MQTT), 8884 (WebSocket)"
echo "Connect with client cert: mosquitto_sub -h localhost -p 8883 --cafile ca.crt --cert client.crt --key client.key -t 'test'"
echo ""
