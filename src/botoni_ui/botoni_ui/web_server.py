#!/usr/bin/env python3
import os, ssl, subprocess, functools, threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

CERT_FILE = '/tmp/botoni_cert.pem'
KEY_FILE  = '/tmp/botoni_key.pem'

def ensure_cert():
    if os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE):
        return
    subprocess.run([
        'openssl','req','-x509','-newkey','rsa:2048',
        '-keyout',KEY_FILE,'-out',CERT_FILE,
        '-days','365','-nodes','-subj','/CN=botoni'
    ], check=True, capture_output=True)

class WebServerNode(Node):
    def __init__(self):
        super().__init__('botoni_web_server')
        self.declare_parameter('port', 5173)
        self.declare_parameter('address', '0.0.0.0')
        port    = self.get_parameter('port').value
        address = self.get_parameter('address').value

        share   = get_package_share_directory('botoni_ui')
        web_dir = os.path.join(share, 'web')

        try:
            ensure_cert()
            use_ssl = False
        except Exception as e:
            self.get_logger().warn(f'SSL falló, usando HTTP: {e}')
            use_ssl = False

        class SilentHandler(SimpleHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # silencia todos los logs HTTP
        handler = functools.partial(SilentHandler, directory=web_dir)
        self.httpd = ThreadingHTTPServer((address, port), handler)

        if use_ssl:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            ctx.load_cert_chain(certfile=CERT_FILE, keyfile=KEY_FILE)
            self.httpd.socket = ctx.wrap_socket(
                self.httpd.socket, server_side=True)
            proto = 'https'
        else:
            proto = 'http'

        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.get_logger().info(f'Botoni UI en {proto}://{address}:{port}  web_dir={web_dir}')
        if use_ssl:
            self.get_logger().info('HTTPS activo — acepta advertencia del browser.')

    def destroy_node(self):
        try: self.httpd.shutdown()
        except: pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = WebServerNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()