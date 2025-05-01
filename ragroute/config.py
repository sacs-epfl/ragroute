"""Configuration settings for the federated search system."""

# ZMQ Communication ports
SERVER_ROUTER_PORT = 5555
ROUTER_SERVER_PORT = 5556
SERVER_CLIENT_BASE_PORT = 6000
CLIENT_SERVER_BASE_PORT = 7500

# HTTP server settings
HTTP_HOST = "127.0.0.1"
HTTP_PORT = 8000

# Router queue settings
MAX_QUEUE_SIZE = 100

# For loading the models and data
USR_DIR = "/Users/mdevos"
ONLINE = True  # Whether we compute everything from scratch or use the precomputed information
