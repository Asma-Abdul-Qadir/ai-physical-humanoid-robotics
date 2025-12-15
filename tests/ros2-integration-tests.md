# ROS 2 Integration Tests

This document outlines integration tests to validate the complete ROS 2 examples in simulated environments. These tests ensure that all components work together correctly as a system.

## Test Suite Overview

The ROS 2 integration tests validate:
- Publisher-subscriber communication
- Service client-server interaction
- URDF model loading and visualization
- Launch file functionality
- Action server-client interaction
- Parameter management

## Test Environment Setup

### Prerequisites
- ROS 2 Humble Hawksbill installed
- Gazebo Garden installed
- All required packages built and sourced
- Robot simulation environment ready

### Setup Commands
```bash
# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Source workspace
cd ~/robotics_ws
source install/setup.bash

# Verify ROS 2 installation
ros2 topic list
ros2 node list
```

## Test 1: Basic Communication Integration

### Objective
Validate that publisher and subscriber nodes can communicate successfully in a simulated environment.

### Test Steps
1. Launch the basic publisher-subscriber system:
   ```bash
   ros2 launch exercise_1_pub_sub system.launch.py
   ```

2. Verify communication:
   ```bash
   # In another terminal, echo the topic
   ros2 topic echo /chatter std_msgs/msg/String
   ```

3. Check for continuous message flow between nodes.

### Expected Results
- Publisher sends messages at regular intervals
- Subscriber receives messages without errors
- No message loss or communication failures

### Validation Criteria
- [ ] Publisher and subscriber nodes start successfully
- [ ] Messages are transmitted continuously
- [ ] No communication errors occur
- [ ] Message rate is consistent

## Test 2: Service Communication Integration

### Objective
Validate that service client and server can communicate and exchange data successfully.

### Test Steps
1. Launch the service system:
   ```bash
   ros2 launch exercise_2_service system.launch.py
   ```

2. Call the service from command line:
   ```bash
   ros2 service call /add_two_ints example_interfaces/srv/AddTwoInts "{a: 5, b: 3}"
   ```

3. Verify the service returns the correct sum.

### Expected Results
- Service server responds to requests
- Correct sum is returned (8 for the example above)
- No errors during service call

### Validation Criteria
- [ ] Service server starts successfully
- [ ] Service call completes without errors
- [ ] Correct result is returned
- [ ] Multiple service calls work reliably

## Test 3: URDF Model Integration

### Objective
Validate that URDF robot models load correctly and can be visualized in simulation.

### Test Steps
1. Load the URDF in robot state publisher:
   ```bash
   ros2 run robot_state_publisher robot_state_publisher __params:=path/to/robot_config.yaml
   ```

2. Launch RViz for visualization:
   ```bash
   ros2 run rviz2 rviz2
   ```

3. Add robot model display in RViz:
   - Add "RobotModel" display type
   - Set "Robot Description" to the appropriate parameter name

4. Verify all links and joints are displayed correctly.

### Expected Results
- Robot model appears in RViz
- All links are visible with correct geometry
- Joint axes are displayed properly

### Validation Criteria
- [ ] URDF loads without errors
- [ ] All robot links are visible in RViz
- [ ] Joint connections are correct
- [ ] No missing or malformed components

## Test 4: Launch File Integration

### Objective
Validate that launch files start all specified nodes with correct parameters.

### Test Steps
1. Create a comprehensive launch file that includes:
   - Robot state publisher
   - Joint state publisher
   - Navigation nodes
   - Visualization tools

2. Launch the system:
   ```bash
   ros2 launch robot_system comprehensive.launch.py
   ```

3. Verify all nodes start successfully:
   ```bash
   ros2 node list
   ros2 param list /node_name
   ```

4. Check parameter values are set correctly.

### Expected Results
- All specified nodes start without errors
- Parameters are loaded from YAML files
- No conflicting node names
- System runs stably

### Validation Criteria
- [ ] All nodes start successfully
- [ ] Parameters are loaded correctly
- [ ] No node name conflicts
- [ ] System remains stable for 5+ minutes

## Test 5: Action Server Integration

### Objective
Validate that action servers and clients can coordinate complex tasks with feedback.

### Test Steps
1. Launch the action server:
   ```bash
   ros2 run exercise_5_action navigation_action_server
   ```

2. Send an action goal from client:
   ```bash
   # Using command line tools or a client script
   ros2 action send_goal /navigate_to_pose example_interfaces/action/Fibonacci "{order: 5}"
   ```

3. Monitor feedback during execution.

### Expected Results
- Action goal is accepted by server
- Feedback is provided during execution
- Result is returned upon completion
- Cancellation is handled properly

### Validation Criteria
- [ ] Action server accepts goals
- [ ] Feedback is published during execution
- [ ] Result is returned correctly
- [ ] Cancellation is handled gracefully

## Test 6: Parameter Management Integration

### Objective
Validate that parameters can be set, changed, and monitored across multiple nodes.

### Test Steps
1. Launch nodes with parameter configuration:
   ```bash
   ros2 launch parameter_system system.launch.py
   ```

2. Change parameters at runtime:
   ```bash
   ros2 param set /parameter_manager_node max_linear_velocity 2.0
   ```

3. Verify parameter changes are applied:
   ```bash
   ros2 param get /parameter_manager_node max_linear_velocity
   ```

4. Test parameter validation with invalid values.

### Expected Results
- Parameters are set correctly at startup
- Runtime parameter changes are applied
- Invalid values are rejected
- Nodes respond appropriately to parameter changes

### Validation Criteria
- [ ] Parameters load correctly at startup
- [ ] Runtime changes are applied successfully
- [ ] Invalid values are rejected with proper error messages
- [ ] Nodes function correctly with new parameter values

## Test 7: Simulation Integration

### Objective
Validate that all ROS 2 components work together in a simulated environment.

### Test Steps
1. Launch Gazebo simulation with ROS 2 bridge:
   ```bash
   ros2 launch gazebo_ros empty_world.launch.py
   ```

2. Spawn robot model in simulation:
   ```bash
   ros2 run gazebo_ros spawn_entity.py -entity my_robot -file path/to/robot.urdf
   ```

3. Launch robot controllers:
   ```bash
   ros2 launch robot_control launch_controllers.launch.py
   ```

4. Send commands to robot and verify response.

### Expected Results
- Robot spawns correctly in Gazebo
- Controllers connect to simulation
- Robot responds to commands
- Sensor data is published correctly

### Validation Criteria
- [ ] Robot model loads in Gazebo
- [ ] Controllers connect without errors
- [ ] Robot moves according to commands
- [ ] Sensor data is published and accurate

## Automated Test Scripts

### Basic Communication Test Script
```bash
#!/bin/bash
# test_basic_communication.sh

# Launch publisher
ros2 run exercise_1_pub_sub publisher &
PUB_PID=$!

# Give publisher time to start
sleep 2

# Launch subscriber in background to capture messages
ros2 run exercise_1_pub_sub subscriber > /tmp/subscriber_output &
SUB_PID=$!

# Let them communicate for a few seconds
sleep 5

# Check if messages were received
if grep -q "I heard:" /tmp/subscriber_output; then
    echo "PASS: Communication test successful"
    RESULT=0
else
    echo "FAIL: No messages received"
    RESULT=1
fi

# Cleanup
kill $PUB_PID $SUB_PID 2>/dev/null
rm /tmp/subscriber_output 2>/dev/null

exit $RESULT
```

### Parameter Test Script
```bash
#!/bin/bash
# test_parameters.sh

# Launch parameter node
ros2 run exercise_6_params parameter_manager &
NODE_PID=$!

# Wait for node to start
sleep 3

# Test parameter setting
ros2 param set /parameter_manager_node max_linear_velocity 2.5
sleep 1

# Verify parameter was set
CURRENT_VAL=$(ros2 param get /parameter_manager_node max_linear_velocity | awk '{print $2}')

if [ "$CURRENT_VAL" = "2.5" ]; then
    echo "PASS: Parameter set successfully"
    RESULT=0
else
    echo "FAIL: Parameter not set correctly, got $CURRENT_VAL"
    RESULT=1
fi

# Cleanup
kill $NODE_PID 2>/dev/null

exit $RESULT
```

## Test Execution Matrix

| Test | Manual | Automated | Simulation | Priority |
|------|--------|-----------|------------|----------|
| Basic Communication | ✓ | ✓ | - | High |
| Service Communication | ✓ | ✓ | - | High |
| URDF Model | ✓ | - | ✓ | High |
| Launch Files | ✓ | ✓ | - | High |
| Actions | ✓ | ✓ | - | High |
| Parameters | ✓ | ✓ | - | High |
| Full Simulation | ✓ | - | ✓ | Critical |

## Success Criteria

### Individual Test Success
- Each test must pass 10 consecutive runs
- No memory leaks or resource issues
- Response times within acceptable limits
- Error handling works correctly

### Integration Success
- 95%+ of tests pass in automated suite
- No critical failures in simulation
- System stability over 30-minute test period
- All components work together without conflicts

## Failure Analysis

### Common Failure Modes
1. **Node startup failures**: Check dependencies and parameter files
2. **Communication timeouts**: Verify network configuration and QoS settings
3. **Parameter validation errors**: Check parameter types and ranges
4. **Simulation instability**: Verify URDF model and physics properties

### Debugging Commands
```bash
# Check all active nodes
ros2 node list

# Check all active topics
ros2 topic list

# Check node status
ros2 lifecycle list node_name

# Check parameter values
ros2 param list node_name
ros2 param get node_name param_name

# Monitor topic messages
ros2 topic echo /topic_name

# Check service availability
ros2 service list
```

## Performance Benchmarks

### Communication Performance
- **Message latency**: < 10ms for local communication
- **Message rate**: Support up to 1000 Hz for critical topics
- **Bandwidth usage**: Optimize large message types (images, point clouds)

### Parameter Performance
- **Update rate**: Parameters should update within 100ms
- **Memory usage**: Minimal overhead for parameter management
- **Validation time**: Parameter validation should be < 1ms

## Continuous Integration

### CI Pipeline Steps
1. Build all packages
2. Run unit tests
3. Run integration tests
4. Run simulation tests
5. Generate test reports
6. Archive test results

### Test Report Format
```yaml
test_suite: ros2_integration
timestamp: 2025-01-10T10:00:00Z
results:
  basic_communication:
    status: PASS
    duration: 15.2s
    details: "All 10 runs successful"
  service_communication:
    status: PASS
    duration: 12.8s
    details: "Service calls responded correctly"
  # ... additional test results
summary:
  total_tests: 7
  passed: 7
  failed: 0
  success_rate: 100%
```

## Key Takeaways

- Integration tests validate that components work together as a system
- Automated tests ensure consistent validation across builds
- Simulation integration tests validate real-world scenarios
- Performance benchmarks ensure system meets requirements
- Failure analysis helps identify and resolve issues quickly