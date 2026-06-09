"""
ros2_service_server.py
Servidor de servicio ROS2 (Python) para AddTwoInts.

Pasos previos:
  1. Crear el workspace:  mkdir -p ~/ros2_ws/src
  2. Copiar my_interfaces/ dentro de ~/ros2_ws/src/
  3. Compilar:
       cd ~/ros2_ws
       colcon build --packages-select my_interfaces
       source install/setup.bash
  4. Ejecutar: python3 ros2_service_server.py
     (o mejor, crear un paquete Python con entry_points)
"""

import rclpy
from rclpy.node import Node

# Importar la interfaz generada automáticamente por rosidl
from my_interfaces.srv import AddTwoInts


class AddTwoIntsServer(Node):
    """Nodo ROS2 que ofrece el servicio 'add_two_ints'."""

    def __init__(self):
        super().__init__("add_two_ints_server")

        # Crear el servicio: tipo de interfaz, nombre, callback
        self.srv = self.create_service(
            AddTwoInts, "add_two_ints", self.add_callback
        )
        self.get_logger().info("Servicio 'add_two_ints' listo.")

    def add_callback(self, request, response):
        """Callback que se ejecuta al recibir una solicitud."""
        response.sum = request.a + request.b
        self.get_logger().info(
            f"Request: a={request.a}, b={request.b} -> sum={response.sum}"
        )
        return response


def main(args=None):
    rclpy.init(args=args)
    node = AddTwoIntsServer()

    try:
        rclpy.spin(node)  # Mantener el nodo activo
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
