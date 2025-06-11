#!/usr/bin/env python3
"""
Simple HTTP server to serve the analysis files locally.
This avoids CORS issues when loading benchmark_results.json.
"""

import http.server
import socketserver
import os
import webbrowser
from threading import Timer

PORT = 8000

def open_browser():
    webbrowser.open(f'http://localhost:{PORT}/analyse.html')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

Handler = http.server.SimpleHTTPRequestHandler

print(f"Starting server at http://localhost:{PORT}")
print(f"Open http://localhost:{PORT}/analyse.html in your browser")
print("Press Ctrl+C to stop the server")

# Open browser after 1 second
Timer(1, open_browser).start()

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()