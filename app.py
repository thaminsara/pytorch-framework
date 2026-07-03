import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from gui.routes import app
from src.utils import ensure_dir, load_config

config = load_config(os.path.join(os.path.dirname(__file__), "config", "default.yaml"))
ensure_dir(config.get("paths", {}).get("data_dir", "./data"))
ensure_dir(config.get("paths", {}).get("models_dir", "./models"))
ensure_dir(config.get("paths", {}).get("experiments_dir", "./experiments"))
ensure_dir(config.get("paths", {}).get("plots_dir", "./gui/static/plots"))


def main():
    parser = argparse.ArgumentParser(description="PyTorch Training Framework")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    print(f"Starting PyTorch Training Framework on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
