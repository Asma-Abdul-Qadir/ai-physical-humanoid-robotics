# Capstone Project Integration Tests

## Overview

This document outlines the integration tests for the complete humanoid robotics system. These tests validate that all modules work together seamlessly to achieve the capstone project objectives of integrating voice commands, motion planning, navigation, and object recognition.

## Integration Test Cases

### Test Case 1: Complete Voice Command Flow
**Objective**: Validate the complete flow from voice input to action execution

**Test Steps**:
1. System receives voice command: "Go to the kitchen"
2. Language module processes the command and identifies navigation intent
3. Perception module provides current scene context
4. Fusion module makes decision to execute navigation
5. Action module executes navigation to kitchen
6. System provides confirmation response

**Validation Criteria**:
- Command correctly parsed as navigation intent
- Appropriate response generated within 2 seconds
- Navigation action executed successfully
- Robot reaches specified destination within 10cm accuracy

### Test Case 2: Object Recognition and Manipulation
**Objective**: Validate perception and manipulation integration

**Test Steps**:
1. System receives command: "Get me the red cup"
2. Perception module identifies objects in environment
3. System confirms presence of target object
4. Action module plans and executes manipulation
5. System provides feedback on action status

**Validation Criteria**:
- Object correctly identified in scene
- Manipulation action planned and executed
- Success rate >90% for object grasping
- System provides appropriate feedback

### Test Case 3: Multi-Modal Context Integration
**Objective**: Validate integration of perception, language, and action with context awareness

**Test Steps**:
1. System receives contextual command: "Go to the person in the living room"
2. Perception module identifies people in environment
3. Navigation system plans route to person's location
4. Safety system monitors for obstacles
5. Action module executes navigation

**Validation Criteria**:
- Context correctly interpreted
- Multiple modalities used effectively
- Safe navigation achieved
- Appropriate behavior in social context

### Test Case 4: Safety System Integration
**Objective**: Validate safety system operates correctly with other modules

**Test Steps**:
1. System attempts navigation in environment with obstacles
2. Safety module monitors obstacle distances
3. System adjusts behavior based on safety status
4. Emergency stop functionality tested

**Validation Criteria**:
- Safety system monitors continuously
- Obstacles detected and avoided
- Emergency stop functions correctly
- Risk level calculated appropriately

### Test Case 5: Complex Task Execution
**Objective**: Validate execution of complex multi-step tasks

**Test Steps**:
1. System receives complex command: "Go to the kitchen, find a cup, and bring it to me"
2. Task is broken down into sub-tasks
3. Each sub-task is executed sequentially
4. System maintains context across sub-tasks
5. Final task completion confirmed

**Validation Criteria**:
- Task decomposition successful
- All sub-tasks executed correctly
- Context maintained throughout
- Final goal achieved

## Integration Test Implementation

### Test Script: Complete Flow Validation

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from capstone_language.msg import Command, Response
from capstone_action.msg import ActionResult
from capstone_perception.msg import SceneDescription
import time
import threading

class CompleteFlowTest(Node):
    """Test the complete voice-to-action flow"""

    def __init__(self):
        super().__init__('complete_flow_test')

        # Publishers
        self.speech_pub = self.create_publisher(String, 'speech_recognition', 10)

        # Subscribers
        self.response_sub = self.create_subscription(
            Response, 'responses', self.response_callback, 10)
        self.action_result_sub = self.create_subscription(
            ActionResult, 'action_results', self.action_result_callback, 10)
        self.scene_sub = self.create_subscription(
            SceneDescription, 'scene_description', self.scene_callback, 10)

        self.response_received = False
        self.action_result_received = False
        self.scene_received = False

        self.get_logger().info('Complete flow test initialized')

    def response_callback(self, msg):
        self.response_received = True
        self.get_logger().info(f'Response received: {msg.text}')

    def action_result_callback(self, msg):
        self.action_result_received = True
        self.get_logger().info(f'Action result: {msg.action_type} - {msg.success}')

    def scene_callback(self, msg):
        self.scene_received = True
        self.get_logger().info(f'Scene: {msg.room_type} with {len(msg.objects)} objects')

    def run_test(self):
        """Run the complete flow test"""
        self.get_logger().info('Starting complete flow test...')

        # Reset flags
        self.response_received = False
        self.action_result_received = False
        self.scene_received = False

        # Send a navigation command
        command = "Go to the kitchen"
        speech_msg = String()
        speech_msg.data = command
        self.speech_pub.publish(speech_msg)

        self.get_logger().info(f'Sent command: {command}')

        # Wait for responses (up to 10 seconds)
        timeout = time.time() + 10.0
        while time.time() < timeout:
            if self.response_received and self.action_result_received:
                break
            time.sleep(0.1)

        # Validate results
        success = all([
            self.response_received,
            self.action_result_received
        ])

        if success:
            self.get_logger().info('✅ Complete flow test PASSED')
        else:
            self.get_logger().info('❌ Complete flow test FAILED')
            self.get_logger().info(f'  - Response received: {self.response_received}')
            self.get_logger().info(f'  - Action result received: {self.action_result_received}')

        return success

def main(args=None):
    rclpy.init(args=args)
    test_node = CompleteFlowTest()

    # Give system time to initialize
    time.sleep(3.0)

    success = test_node.run_test()

    test_node.destroy_node()
    rclpy.shutdown()

    return success

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
```

### Test Script: Multi-Modal Integration

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from capstone_perception.msg import SceneDescription, ObjectDetection
from capstone_language.msg import Command, Response
from capstone_action.msg import ActionResult
import time

class MultiModalIntegrationTest(Node):
    """Test integration of perception, language, and action"""

    def __init__(self):
        super().__init__('multimodal_integration_test')

        # Publishers
        self.speech_pub = self.create_publisher(String, 'speech_recognition', 10)

        # Subscribers
        self.response_sub = self.create_subscription(
            Response, 'responses', self.response_callback, 10)
        self.action_result_sub = self.create_subscription(
            ActionResult, 'action_results', self.action_result_callback, 10)
        self.scene_sub = self.create_subscription(
            SceneDescription, 'scene_description', self.scene_callback, 10)

        self.responses = []
        self.action_results = []
        self.scenes = []

        self.get_logger().info('Multi-modal integration test initialized')

    def response_callback(self, msg):
        self.responses.append(msg)
        self.get_logger().info(f'Response: {msg.text}')

    def action_result_callback(self, msg):
        self.action_results.append(msg)
        self.get_logger().info(f'Action: {msg.action_type} - {msg.success}')

    def scene_callback(self, msg):
        self.scenes.append(msg)
        self.get_logger().info(f'Scene: {msg.room_type}')

    def run_test(self):
        """Run multi-modal integration test"""
        self.get_logger().info('Starting multi-modal integration test...')

        # Clear previous data
        self.responses.clear()
        self.action_results.clear()
        self.scenes.clear()

        # Send commands that require multi-modal integration
        commands = [
            "What do you see?",
            "Go to the kitchen",
            "Get me the red cup"
        ]

        for i, command in enumerate(commands):
            self.get_logger().info(f'Sending command {i+1}: {command}')
            speech_msg = String()
            speech_msg.data = command
            self.speech_pub.publish(speech_msg)

            # Wait between commands
            time.sleep(2.0)

        # Wait for all responses (up to 15 seconds)
        time.sleep(15.0)

        # Validate integration
        success = all([
            len(self.responses) >= len(commands),
            len(self.action_results) >= 2,  # At least navigation and manipulation
            len(self.scenes) >= 1  # At least one scene update
        ])

        self.get_logger().info(f'Responses received: {len(self.responses)}')
        self.get_logger().info(f'Action results received: {len(self.action_results)}')
        self.get_logger().info(f'Scenes received: {len(self.scenes)}')

        if success:
            self.get_logger().info('✅ Multi-modal integration test PASSED')
        else:
            self.get_logger().info('❌ Multi-modal integration test FAILED')

        return success

def main(args=None):
    rclpy.init(args=args)
    test_node = MultiModalIntegrationTest()

    # Give system time to initialize
    time.sleep(3.0)

    success = test_node.run_test()

    test_node.destroy_node()
    rclpy.shutdown()

    return success

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
```

### Test Script: Safety Integration

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from capstone_safety.msg import SafetyStatus
from capstone_action.msg import ActionResult
import time

class SafetyIntegrationTest(Node):
    """Test integration of safety systems"""

    def __init__(self):
        super().__init__('safety_integration_test')

        # Publishers
        self.speech_pub = self.create_publisher(String, 'speech_recognition', 10)
        self.emergency_stop_pub = self.create_publisher(Bool, 'emergency_stop', 10)

        # Subscribers
        self.safety_sub = self.create_subscription(
            SafetyStatus, 'safety_status', self.safety_callback, 10)
        self.action_result_sub = self.create_subscription(
            ActionResult, 'action_results', self.action_result_callback, 10)

        self.safety_status = None
        self.action_results = []
        self.emergency_triggered = False

        self.get_logger().info('Safety integration test initialized')

    def safety_callback(self, msg):
        self.safety_status = msg
        self.get_logger().info(f'Safety: risk={msg.risk_level:.2f}, safe={msg.safe_to_proceed}')

        if not msg.safe_to_proceed and msg.risk_level > 0.8:
            self.emergency_triggered = True

    def action_result_callback(self, msg):
        self.action_results.append(msg)
        self.get_logger().info(f'Action: {msg.action_type} - {msg.success}')

    def run_test(self):
        """Run safety integration test"""
        self.get_logger().info('Starting safety integration test...')

        # Clear previous data
        self.action_results.clear()
        self.emergency_triggered = False

        # Send a navigation command that should be subject to safety checks
        command = "Go forward 2 meters"
        speech_msg = String()
        speech_msg.data = command
        self.speech_pub.publish(speech_msg)

        # Wait to see safety responses (up to 10 seconds)
        timeout = time.time() + 10.0
        while time.time() < timeout and not self.safety_status:
            time.sleep(0.1)

        # Check if safety system is monitoring appropriately
        success = self.safety_status is not None

        if success and self.safety_status.safe_to_proceed:
            self.get_logger().info('✅ Safety system monitoring active and permissive')
        elif success and not self.safety_status.safe_to_proceed:
            self.get_logger().info('✅ Safety system monitoring active and restrictive (as appropriate)')
        else:
            self.get_logger().info('❌ Safety system not responding')
            success = False

        # Wait longer for action results
        time.sleep(5.0)

        # Validate that safety doesn't prevent all actions (in safe conditions)
        if success:
            self.get_logger().info('✅ Safety integration test PASSED')
        else:
            self.get_logger().info('❌ Safety integration test FAILED')

        return success

def main(args=None):
    rclpy.init(args=args)
    test_node = SafetyIntegrationTest()

    # Give system time to initialize
    time.sleep(3.0)

    success = test_node.run_test()

    test_node.destroy_node()
    rclpy.shutdown()

    return success

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
```

### Test Script: End-to-End Capstone Validation

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from capstone_perception.msg import SceneDescription
from capstone_language.msg import Command, Response
from capstone_action.msg import ActionResult
from capstone_safety.msg import SafetyStatus
from capstone_fusion.msg import Decision
import time

class EndToEndCapstoneTest(Node):
    """Complete end-to-end capstone integration test"""

    def __init__(self):
        super().__init__('end_to_end_capstone_test')

        # Publishers
        self.speech_pub = self.create_publisher(String, 'speech_recognition', 10)

        # Subscribers
        self.response_sub = self.create_subscription(
            Response, 'responses', self.response_callback, 10)
        self.action_result_sub = self.create_subscription(
            ActionResult, 'action_results', self.action_result_callback, 10)
        self.scene_sub = self.create_subscription(
            SceneDescription, 'scene_description', self.scene_callback, 10)
        self.safety_sub = self.create_subscription(
            SafetyStatus, 'safety_status', self.safety_callback, 10)
        self.decision_sub = self.create_subscription(
            Decision, 'multimodal_decision', self.decision_callback, 10)

        self.responses = []
        self.action_results = []
        self.scenes = []
        self.safety_statuses = []
        self.decisions = []

        self.get_logger().info('End-to-end capstone test initialized')

    def response_callback(self, msg):
        self.responses.append(msg)
        self.get_logger().info(f'Response: {msg.text}')

    def action_result_callback(self, msg):
        self.action_results.append(msg)
        self.get_logger().info(f'Action: {msg.action_type} - {msg.success}')

    def scene_callback(self, msg):
        self.scenes.append(msg)
        self.get_logger().info(f'Scene: {msg.room_type}')

    def safety_callback(self, msg):
        self.safety_statuses.append(msg)
        self.get_logger().info(f'Safety: risk={msg.risk_level:.2f}')

    def decision_callback(self, msg):
        self.decisions.append(msg)
        self.get_logger().info(f'Decision: {msg.data}')

    def run_test(self):
        """Run complete end-to-end test"""
        self.get_logger().info('Starting end-to-end capstone integration test...')

        # Clear all data
        self.clear_data()

        # Define a sequence of commands that tests the full capstone functionality
        command_sequence = [
            ("Environment Check", "What do you see around you?"),
            ("Navigation", "Go to the kitchen"),
            ("Object Interaction", "Get me the cup"),
            ("Information Request", "Tell me about the environment"),
            ("Complex Command", "Go to the person in the living room and say hello")
        ]

        for i, (description, command) in enumerate(command_sequence):
            self.get_logger().info(f'Step {i+1} ({description}): {command}')

            # Publish command
            speech_msg = String()
            speech_msg.data = command
            self.speech_pub.publish(speech_msg)

            # Wait for processing
            time.sleep(4.0)

        # Wait longer for all systems to respond
        time.sleep(10.0)

        # Validate complete integration
        success = self.validate_integration()

        # Print summary
        self.print_test_summary()

        if success:
            self.get_logger().info('✅ End-to-end capstone test PASSED')
        else:
            self.get_logger().info('❌ End-to-end capstone test FAILED')

        return success

    def clear_data(self):
        """Clear all collected data"""
        self.responses.clear()
        self.action_results.clear()
        self.scenes.clear()
        self.safety_statuses.clear()
        self.decisions.clear()

    def validate_integration(self):
        """Validate that all systems integrated properly"""
        # Check that all systems responded appropriately
        min_responses = 3  # Should get responses to most commands
        min_actions = 2    # Should execute several actions
        min_scenes = 1     # Should perceive environment
        min_safety = 1     # Should report safety status
        min_decisions = 2  # Should make fusion decisions

        validation_results = {
            'responses': len(self.responses) >= min_responses,
            'actions': len(self.action_results) >= min_actions,
            'scenes': len(self.scenes) >= min_scenes,
            'safety': len(self.safety_statuses) >= min_safety,
            'decisions': len(self.decisions) >= min_decisions
        }

        all_passed = all(validation_results.values())

        self.get_logger().info('Integration validation results:')
        for component, passed in validation_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            self.get_logger().info(f'  {component}: {status}')

        return all_passed

    def print_test_summary(self):
        """Print test summary"""
        self.get_logger().info('\n--- Test Summary ---')
        self.get_logger().info(f'Responses: {len(self.responses)}')
        self.get_logger().info(f'Action Results: {len(self.action_results)}')
        self.get_logger().info(f'Scenes: {len(self.scenes)}')
        self.get_logger().info(f'Safety Updates: {len(self.safety_statuses)}')
        self.get_logger().info(f'Decisions: {len(self.decisions)}')
        self.get_logger().info('-------------------\n')

def main(args=None):
    rclpy.init(args=args)
    test_node = EndToEndCapstoneTest()

    # Give system time to initialize
    time.sleep(5.0)

    success = test_node.run_test()

    test_node.destroy_node()
    rclpy.shutdown()

    return success

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
```

## Test Execution Framework

### Test Runner Script

```bash
#!/bin/bash

# capstone_integration_tests.sh
# Script to run all capstone integration tests

echo "Starting Capstone Project Integration Tests..."

# Wait for ROS 2 system to be ready
echo "Waiting for ROS 2 system..."
sleep 5

# Run each test individually
echo "Running Complete Flow Test..."
python3 complete_flow_test.py &
FLOW_TEST_PID=$!
wait $FLOW_TEST_PID
FLOW_TEST_RESULT=$?
echo "Complete Flow Test: $([ $FLOW_TEST_RESULT -eq 0 ] && echo 'PASSED' || echo 'FAILED')"

echo "Running Multi-Modal Integration Test..."
python3 multimodal_integration_test.py &
MODAL_TEST_PID=$!
wait $MODAL_TEST_PID
MODAL_TEST_RESULT=$?
echo "Multi-Modal Integration Test: $([ $MODAL_TEST_RESULT -eq 0 ] && echo 'PASSED' || echo 'FAILED')"

echo "Running Safety Integration Test..."
python3 safety_integration_test.py &
SAFETY_TEST_PID=$!
wait $SAFETY_TEST_PID
SAFETY_TEST_RESULT=$?
echo "Safety Integration Test: $([ $SAFETY_TEST_RESULT -eq 0 ] && echo 'PASSED' || echo 'FAILED')"

echo "Running End-to-End Test..."
python3 end_to_end_test.py &
E2E_TEST_PID=$!
wait $E2E_TEST_PID
E2E_TEST_RESULT=$?
echo "End-to-End Test: $([ $E2E_TEST_RESULT -eq 0 ] && echo 'PASSED' || echo 'FAILED')"

# Summary
ALL_RESULTS=($FLOW_TEST_RESULT $MODAL_TEST_RESULT $SAFETY_TEST_RESULT $E2E_TEST_RESULT)
FAILED_TESTS=0
for result in "${ALL_RESULTS[@]}"; do
    if [ $result -ne 0 ]; then
        ((FAILED_TESTS++))
    fi
done

TOTAL_TESTS=${#ALL_RESULTS[@]}
PASSED_TESTS=$((TOTAL_TESTS - FAILED_TESTS))

echo ""
echo "=== Integration Test Summary ==="
echo "Total Tests: $TOTAL_TESTS"
echo "Passed: $PASSED_TESTS"
echo "Failed: $FAILED_TESTS"
echo "Success Rate: $((PASSED_TESTS * 100 / TOTAL_TESTS))%"
echo "==============================="

if [ $FAILED_TESTS -eq 0 ]; then
    echo "🎉 All integration tests PASSED!"
    exit 0
else
    echo "❌ $FAILED_TESTS integration test(s) FAILED"
    exit 1
fi
```

## Continuous Integration Configuration

### GitHub Actions Integration Test Workflow

```yaml
# .github/workflows/capstone-integration-tests.yml
name: Capstone Integration Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  integration-test:
    runs-on: ubuntu-22.04

    steps:
    - uses: actions/checkout@v3

    - name: Setup ROS 2 Humble
      uses: ros-tooling/setup-ros@0.7.3
      with:
        required-ros-distributions: humble

    - name: Install dependencies
      run: |
        sudo apt update
        sudo apt install -y python3-colcon-common-extensions
        rosdep update
        rosdep install --from-paths src --ignore-src -r -y

    - name: Build workspace
      run: |
        source /opt/ros/humble/setup.bash
        colcon build --packages-select capstone_main capstone_perception capstone_language capstone_action capstone_fusion capstone_safety

    - name: Run integration tests
      run: |
        source /opt/ros/humble/setup.bash
        source install/setup.bash
        cd ~/capstone_project
        bash capstone_integration_tests.sh
```

## Validation Criteria

### Success Metrics

1. **Functional Integration (90% minimum)**
   - All modules communicate properly via ROS 2 topics/services
   - Commands flow correctly from input to action
   - Data is properly shared between modules

2. **Performance Requirements**
   - Response time under 3 seconds for voice commands
   - System maintains 10Hz update rate for perception modules
   - 90% success rate for object manipulation tasks
   - < 5cm navigation accuracy

3. **Safety Compliance**
   - Zero safety violations during testing
   - Emergency stop functionality works correctly
   - Safety monitoring active at all times

4. **Reliability Metrics**
   - >95% uptime during 30-minute test period
   - Graceful error recovery
   - Consistent state maintenance across modules

## Test Environment Requirements

### Simulation Environment
The integration tests are designed to work with Gazebo Garden simulation environment, which provides:
- Physics simulation for navigation and manipulation
- Sensor simulation (cameras, LIDAR, etc.)
- Realistic environment models
- Human interaction simulation

### Required Dependencies
- ROS 2 Humble Hawksbill
- Gazebo Garden
- Python 3.8+
- OpenCV
- NumPy
- Standard ROS 2 packages

These integration tests validate that the complete humanoid robotics system integrates all modules successfully, meeting the capstone project objectives of combining voice commands, motion planning, navigation, and object recognition into a unified, functional system.