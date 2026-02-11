#!/usr/bin/env python3
"""Simple HTTP server for the GLB animation viewer."""

import http.server
import json
import os
import sys
from pathlib import Path
from urllib.parse import unquote

PORT = 8080
PROJECT_ROOT = Path(__file__).parent

# Paths - point to latest output and source video
OUTPUT_DIR = PROJECT_ROOT / "test_output_glb"
VIDEO_FILE = Path(r"C:\Sam3DBodyToOpenSim\videos\aitor_garden_walk.mp4")


class ViewerHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)

    def do_GET(self):
        # API: get latest GLB model
        if self.path == "/api/latest-model":
            glb_files = sorted(OUTPUT_DIR.glob("*.glb"), key=os.path.getmtime)
            if glb_files:
                latest = glb_files[-1]
                self.send_json({"file": latest.name})
            else:
                self.send_json({"file": None})
            return

        # API: get video path
        if self.path == "/api/video-path":
            if VIDEO_FILE.exists():
                self.send_json({"path": "/video/" + VIDEO_FILE.name})
            else:
                self.send_json({"path": None})
            return

        # Serve video file
        if self.path.startswith("/video/"):
            if VIDEO_FILE.exists():
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(VIDEO_FILE.stat().st_size))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                with open(VIDEO_FILE, "rb") as f:
                    self.wfile.write(f.read())
                return
            self.send_error(404, "Video not found")
            return

        # Serve GLB model files from output dir
        if self.path.startswith("/output/"):
            filename = unquote(self.path.split("/")[-1])
            file_path = OUTPUT_DIR / filename
            if file_path.exists():
                if filename.endswith('.glb'):
                    content_type = "model/gltf-binary"
                else:
                    content_type = "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(file_path.stat().st_size))
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            self.send_error(404, f"File not found: {filename}")
            return

        # Default file serving
        super().do_GET()

    def send_json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, format, *args):
        if args and (str(args[1]) != "200" or "/api/" in str(args[0])):
            super().log_message(format, *args)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    server = http.server.HTTPServer(("", port), ViewerHandler)
    print(f"Viewer: http://localhost:{port}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Video:  {VIDEO_FILE}")
    print(f"Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
