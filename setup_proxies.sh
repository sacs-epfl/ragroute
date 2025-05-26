#!/bin/bash

LATENCY=20
BANDWIDTH=$((100 * 1024 * 1024 / 8))

## Server → Router
#../toxiproxy-cli create --listen 127.0.0.1:5055 --upstream 127.0.0.1:5555 server-router
#../toxiproxy-cli toxic add --type latency --attribute latency=$LATENCY server-router
#../toxiproxy-cli toxic add --type bandwidth --attribute rate=$BANDWIDTH server-router
#
## Router → Server
#../toxiproxy-cli create --listen 127.0.0.1:5056 --upstream 127.0.0.1:5556 router-server
#../toxiproxy-cli toxic add --type latency --attribute latency=$LATENCY router-server
#../toxiproxy-cli toxic add --type bandwidth --attribute rate=$BANDWIDTH router-server

# Clients
for i in {0..3}; do
  s_port=$((6100 + i))  # proxy listen port for server to client[i]
  r_port=$((6000 + i))  # actual client receive port
  ../toxiproxy-cli create --listen 127.0.0.1:$s_port --upstream 127.0.0.1:$r_port s2c$i
  ../toxiproxy-cli toxic add --type latency --attribute latency=$LATENCY --downstream s2c$i
  ../toxiproxy-cli toxic add --type bandwidth --attribute rate=$BANDWIDTH --downstream s2c$i

  s_port=$((7600 + i))  # proxy listen port for client[i] to server
  r_port=$((7500 + i))  # actual server receive port
  ../toxiproxy-cli create --listen 127.0.0.1:$s_port --upstream 127.0.0.1:$r_port c2s$i
  ../toxiproxy-cli toxic add --type latency --attribute latency=$LATENCY --upstream c2s$i
  ../toxiproxy-cli toxic add --type bandwidth --attribute rate=$BANDWIDTH --upstream c2s$i
done
