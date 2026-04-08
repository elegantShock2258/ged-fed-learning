import http.server
import socketserver
import threading
import os
import logging
import json
import functools

log = logging.getLogger(__name__)

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        # Prevent browser caching of JSON so the visual is instant
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

def create_handler_class(serve_directory):
    class VisualizerHandler(CORSHTTPRequestHandler):
        def translate_path(self, path):
            # Normalize path securely within serve_directory
            path = path.split("?", 1)[0].split("#", 1)[0]
            path = os.path.normpath(path)
            assert not path.startswith("..")
            # Strip initial slash
            if path.startswith("/"):
                path = path[1:]
            return os.path.join(serve_directory, path)
    return VisualizerHandler

def run_bridge(port=8080):
    serve_dir = os.path.join(os.path.dirname(__file__), "..", "saved_models")
    os.makedirs(serve_dir, exist_ok=True)
    
    # Initialize empty state if missing
    state_file = os.path.join(serve_dir, "realtime_state.json")
    if not os.path.exists(state_file):
        with open(state_file, "w") as f:
            json.dump({"nodes": {}, "connections": {}}, f)
            
    # Serve the directory securely through translate_path instead of chdir
    Handler = create_handler_class(serve_dir)
    try:
        with socketserver.TCPServer(("", port), Handler) as httpd:
            log.info(f"Visualizer Bridge streaming data from {serve_dir} on port {port}")
            httpd.serve_forever()
    except Exception as e:
        log.warning(f"Bridge server failed to start on port {port}: {e}")

def start_visualizer_bridge_daemon(port=8080):
    t = threading.Thread(target=run_bridge, args=(port,), daemon=True)
    t.start()
    return t

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_bridge()
