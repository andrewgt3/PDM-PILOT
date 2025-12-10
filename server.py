import http.server
import socketserver
import os

PORT = 8080
DIRECTORY = "frontend/dist"

class SPAHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # 1. Check if the path actually exists (e.g., /assets/foo.js)
        # 2. If valid file, serve it.
        # 3. If NOT valid file, serve index.html (SPA Fallback)
        
        abs_path = os.path.join(os.getcwd(), DIRECTORY, self.path.lstrip('/'))
        
        # If it's a directory (like /assets/) but doesn't have an index.html, 
        # normally it shows a listing. We want to avoid that for routes.
        # But wait, /assets IS a real directory in dist.
        # React routes (like /audit, /plant) are NOT directories.
        
        if os.path.exists(abs_path) and os.path.isfile(abs_path):
            # It's a real file (css, js, png), act normal
            super().do_GET()
        else:
            # It's a route (or a missing file). Redirect to index.html
            self.path = '/'
            super().do_GET()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

if __name__ == "__main__":
    # Allow address reuse
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), SPAHandler) as httpd:
        print(f"serving SPA at http://localhost:{PORT}")
        httpd.serve_forever()
