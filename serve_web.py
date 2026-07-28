# Serves web/ over HTTPS on port 8443 using the self-signed cert.
# Phone URL: https://192.168.1.16:8443  (accept the security warning once)
import ssl, os
from http.server import HTTPServer, SimpleHTTPRequestHandler

os.chdir(os.path.join(os.path.dirname(__file__), "web"))

ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.load_cert_chain(
    certfile=os.path.join(os.path.dirname(__file__), "cert.pem"),
    keyfile=os.path.join(os.path.dirname(__file__),  "key.pem"),
)

server = HTTPServer(("0.0.0.0", 8443), SimpleHTTPRequestHandler)
server.socket = ctx.wrap_socket(server.socket, server_side=True)
print("Serving web/ over HTTPS -> https://192.168.1.16:8443")
server.serve_forever()
