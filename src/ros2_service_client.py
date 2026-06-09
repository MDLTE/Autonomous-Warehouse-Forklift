"""
ros2_service_client.py
Cliente de servicio ROS2 (Python) para AddTwoInts.

Uso:
  1. Asegurarse de que el servidor esté corriendo:
       ros2 run <paquete> add_two_ints_server
  2. Ejecutar: python3 ros2_service_client.py 3 5
     (o: ros2 run <paquete> add_two_ints_client 3 5)
"""

import sys
import rclpy
from rclpy.node import Node

# Importar la interfaz generada automáticamente por rosidl
from my_interfaces.srv import AddTwoInts


class AddTwoIntsClient(Node):
    """Nodo ROS2 que actúa como cliente del servicio 'add_two_ints'."""

    def __init__(self):
        super().__init__("add_two_ints_client")

        # Crear el cliente del servicio
        self.client = self.create_client(AddTwoInts, "add_two_ints")

        # Esperar a que el servicio esté disponible
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Esperando al servicio 'add_two_ints'...")

    def send_request(self, a, b):
        """Enviar una solicitud asíncrona al servicio."""
        request = AddTwoInts.Request()
        request.a = a
        request.b = b

        # Llamada asíncrona
        future = self.client.call_async(request)
        return future


def main(args=None):
    rclpy.init(args=args)
    node = AddTwoIntsClient()

    # Leer argumentos de línea de comandos
    a = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    future = node.send_request(a, b)

    # Esperar la respuesta
    rclpy.spin_until_future_complete(node, future)
    response = future.result()
    node.get_logger().info(f"Resultado: {a} + {b} = {response.sum}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
