#!/usr/bin/env python3
"""
Simple HTTP server to run the HTML/CSS/JS chatbot locally.
This serves the static files and avoids CORS issues.
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path

# Change to the web directory
web_dir = Path(__file__).parent
os.chdir(web_dir)

PORT = 8000

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add permissive CORS headers for static assets
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    # Handle CORS preflight for any static path (not strictly necessary but nice to have)
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def main():
    handler = MyHTTPRequestHandler
    
    print(f"🚀 Starting UnB Chatbot server...")
    print(f"📁 Serving files from: {web_dir}")
    print(f"🌐 Server running at: http://localhost:{PORT}")
    print(f"📱 Open http://localhost:{PORT} in your browser")
    print(f"⚠️  Make sure the mock server is running on port 8001 if you want to test locally")
    print(f"    Run: python mock_chatbot_server.py")
    print(f"\n Press Ctrl+C to stop the server")
    
    try:
        with socketserver.TCPServer(("", PORT), handler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n⏹️  Server stopped")
        sys.exit(0)
    except OSError as e:
        if e.errno == 98:  # Address already in use
            print(f"\n❌ Error: Port {PORT} is already in use!")
            print(f"   Try closing other applications or use a different port")
        else:
            raise

if __name__ == "__main__":
    main()
