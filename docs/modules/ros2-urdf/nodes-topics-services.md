---
sidebar_position: 2
---

# ROS 2 Nodes, Topics, and Services

This section covers the fundamental communication patterns in ROS 2: nodes, topics, and services. Understanding these concepts is crucial for building distributed robotic systems.

## Learning Objectives

By the end of this section, you will be able to:
- Create and run ROS 2 nodes in Python and C++
- Implement publishers and subscribers for topic-based communication
- Create and use services for request/response communication
- Understand the differences between topics and services
- Debug communication issues in ROS 2 systems

## ROS 2 Nodes

A node is a process that performs computation. Nodes are the fundamental building blocks of a ROS 2 program. Multiple nodes are typically used to build a robotic system, each responsible for different functionality.

### Node Characteristics

- **Unique name**: Each node must have a unique name within the ROS graph
- **Communication interface**: Nodes communicate with other nodes through topics, services, and actions
- **Lifecycle**: Nodes have a well-defined lifecycle with states (unconfigured, inactive, active, finalized)
- **Parameters**: Nodes can have configurable parameters that can be set at runtime

### Creating a Node in Python

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_node')
        self.get_logger().info('Minimal node created')

def main(args=None):
    rclpy.init(args=args)
    node = MinimalNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Creating a Node in C++

```cpp
#include "rclcpp/rclcpp.hpp"

class MinimalNode : public rclcpp::Node
{
public:
    MinimalNode() : Node("minimal_node")
    {
        RCLCPP_INFO(this->get_logger(), "Minimal node created");
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MinimalNode>());
    rclcpp::shutdown();
    return 0;
}
```

## Topics and Message Passing

Topics enable asynchronous, decoupled communication between nodes using a publish/subscribe pattern. Multiple publishers can publish to the same topic, and multiple subscribers can subscribe to the same topic.

### Topic Characteristics

- **Unidirectional**: Data flows from publisher to subscriber
- **Asynchronous**: Publishers and subscribers don't need to be synchronized
- **Many-to-many**: Multiple publishers can send to multiple subscribers
- **Message types**: Each topic has a specific message type that defines the data structure

### Publisher Example

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Subscriber Example

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Services

Services provide synchronous, bidirectional communication between nodes using a request/response pattern. A service client sends a request and waits for a response from a service server.

### Service Characteristics

- **Synchronous**: Client waits for response before continuing
- **Bidirectional**: Request and response data flow
- **One-to-one**: One client communicates with one server at a time
- **Service types**: Each service has a specific type defining request and response structure

### Service Server Example

```python
#!/usr/bin/env python3

from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning {response.sum}')
        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()
    rclpy.spin(minimal_service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Service Client Example

```python
#!/usr/bin/env python3

from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalClientAsync(Node):
    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    minimal_client = MinimalClientAsync()
    response = minimal_client.send_request(1, 2)
    minimal_client.get_logger().info(f'Result of add_two_ints: {response.sum}')
    minimal_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Quality of Service (QoS) Settings

QoS settings control how messages are delivered between publishers and subscribers, affecting reliability, durability, and performance.

### Common QoS Profiles

- **Default**: Balances performance and reliability
- **Reliable**: Guarantees message delivery
- **Best effort**: Prioritizes performance over reliability
- **Sensor data**: Optimized for sensor data streams
- **Services**: Default settings for services

### Example with Custom QoS

```python
from rclpy.qos import QoSProfile

# Create a custom QoS profile
qos_profile = QoSProfile(depth=10)
qos_profile.reliability = rclpy.qos.ReliabilityPolicy.RELIABLE

# Use it when creating a publisher
publisher = self.create_publisher(String, 'topic', qos_profile)
```

## Practical Example: Robot Command Interface

Let's create a practical example that combines nodes, topics, and services for robot control:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from example_interfaces.srv import Trigger

class RobotCommander(Node):
    def __init__(self):
        super().__init__('robot_commander')

        # Publisher for robot commands
        self.cmd_publisher = self.create_publisher(String, 'robot_commands', 10)

        # Service for robot status
        self.status_service = self.create_service(
            Trigger,
            'get_robot_status',
            self.get_status_callback
        )

        # Timer for periodic status updates
        self.timer = self.create_timer(1.0, self.status_timer_callback)

        self.robot_state = "idle"
        self.get_logger().info("Robot Commander node initialized")

    def get_status_callback(self, request, response):
        response.success = True
        response.message = f"Robot is {self.robot_state}"
        self.get_logger().info(f"Status requested: {response.message}")
        return response

    def status_timer_callback(self):
        msg = String()
        msg.data = f"Robot status: {self.robot_state}"
        self.cmd_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = RobotCommander()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Debugging ROS 2 Communication

### Useful Commands

- `ros2 node list` - List all active nodes
- `ros2 topic list` - List all active topics
- `ros2 service list` - List all active services
- `ros2 topic echo <topic_name>` - Monitor messages on a topic
- `ros2 topic info <topic_name>` - Get information about a topic
- `ros2 node info <node_name>` - Get information about a node

### Common Issues

- **Node name conflicts**: Use unique node names or namespaces
- **Message type mismatches**: Ensure publisher and subscriber use same message type
- **Network configuration**: Verify ROS_DOMAIN_ID and RMW settings
- **Permissions**: Ensure proper permissions for DDS communication

## Key Takeaways

- Nodes are the fundamental computational units in ROS 2
- Topics enable asynchronous, many-to-many communication
- Services provide synchronous, request/response communication
- QoS settings control message delivery characteristics
- Proper debugging techniques help identify communication issues

In the next section, we'll explore URDF (Unified Robot Description Format) for describing robot models.