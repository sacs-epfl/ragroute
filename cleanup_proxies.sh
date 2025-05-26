#!/bin/bash
#for name in server-router router-server s2c0 s2c1 s2c2 s2c3 c2s0 c2s1 c2s2 c2s3; do
for name in s2c0 s2c1 s2c2 s2c3 c2s0 c2s1 c2s2 c2s3; do
  ../toxiproxy-cli delete $name
done

