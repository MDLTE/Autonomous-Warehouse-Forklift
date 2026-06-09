#!/usr/bin/env python3
import subprocess, time, urllib.request, json
import rclpy
from rclpy.node import Node

class NgrokNode(Node):
    def __init__(self):
        super().__init__('ngrok_node')
        subprocess.Popen(['ngrok', 'http', '5173'],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
        try:
            res = urllib.request.urlopen('http://localhost:4040/api/tunnels')
            data = json.loads(res.read())
            url = data['tunnels'][0]['public_url']
            self.get_logger().info(f'╔══════════════════════════════════════╗')
            self.get_logger().info(f'║  NGROK: {url}')
            self.get_logger().info(f'╚══════════════════════════════════════╝')
        except Exception as e:
            self.get_logger().warn(f'ngrok URL no disponible: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = NgrokNode()
    rclpy.spin(node)

if __name__ == '__main__':
    main()