#!/bin/bash
set -e

# create swap file
echo "sudo fallocate -l 4G /swapfile"
sudo fallocate -l 4G /swapfile
echo "sudo chmod 600 /swapfile"
sudo chmod 600 /swapfile

# enabling the swap file
echo "sudo mkswap /swapfile"
sudo mkswap /swapfile
echo "sudo swapon /swapfile"
sudo swapon /swapfile

# check the swapfile
echo "sudo swapon --show"
sudo swapon --show
echo "free -h"
free -h