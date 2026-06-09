import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import String


WAYPOINTS = [
    (3.23, 1.43),
    (3.26, 4.08),
    (0.83, 4.24),
    (0.91, 0.97),
]


class MissionManager(Node):
    """
    Sends goals to nav_node_sim one at a time.
    Sends the next waypoint whenever the robot is IDLE.
    Assumes IDLE = arrived at previous goal (or startup).
    """

    def __init__(self):
        super().__init__('mission_manager')

        self.goal_pub = self.create_publisher(Point, '/nav_goal_pose', 10)
        self.state_sub = self.create_subscription(
            String, '/nav_state', self.state_cb, 10)

        self.nav_state  = ''
        self.wp_idx     = 0
        self.waiting    = False   # True while robot is executing a goal

        self.create_timer(0.2, self.mission_loop)

        self.get_logger().info(
            f'Mission manager ready — {len(WAYPOINTS)} waypoints: {WAYPOINTS}')

    def state_cb(self, msg: String):
        self.nav_state = msg.data.strip()

    def send_waypoint(self):
        wx, wy = WAYPOINTS[self.wp_idx]
        msg   = Point()
        msg.x = float(wx)
        msg.y = float(wy)
        msg.z = 0.0
        self.goal_pub.publish(msg)
        self.get_logger().info(
            f'Waypoint {self.wp_idx + 1}/{len(WAYPOINTS)} sent: ({wx:.2f}, {wy:.2f})')
        self.wp_idx  += 1
        self.waiting  = True

    def mission_loop(self):
        if self.wp_idx >= len(WAYPOINTS):
            return

        if self.nav_state == 'IDLE':
            self.waiting = False

        if not self.waiting and self.nav_state == 'IDLE':
            self.send_waypoint()


def main(args=None):
    rclpy.init(args=args)
    node = MissionManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()