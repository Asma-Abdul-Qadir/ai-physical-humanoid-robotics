---
sidebar_position: 7
---

# AI Robot Brain Integration Tests: Perception Pipeline Validation

Welcome to the AI Robot Brain Integration Tests module, which focuses on comprehensive testing of the perception pipeline and multimodal integration for humanoid robots. This chapter covers testing strategies, implementation, and validation of integrated systems that connect vision, language, and action capabilities.

## Learning Objectives

By the end of this section, you will be able to:
- Design comprehensive integration tests for multimodal perception systems
- Implement test frameworks for vision-language-action pipelines
- Validate perception pipeline functionality and performance
- Test integration between different AI components
- Create automated test suites for robot brain modules
- Evaluate system robustness and reliability
- Implement continuous testing strategies

## Introduction to Integration Testing for AI Robot Brains

Integration testing for AI robot brains is critical because these systems involve multiple complex components that must work together seamlessly. Unlike unit testing individual components, integration testing validates that the combined system functions correctly as a whole.

### Integration Testing Challenges

AI robot brain systems present unique testing challenges:
- **Multimodal Data Flow**: Testing data flow between vision, audio, and language components
- **Real-time Performance**: Ensuring integrated systems meet real-time requirements
- **Complex State Management**: Validating state transitions across multiple components
- **Uncertainty Handling**: Testing systems that deal with uncertain, noisy sensor data
- **Safety Critical Operations**: Ensuring integrated safety mechanisms work correctly

### Integration Testing Strategy

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Vision        │    │   Language      │    │   Action        │
│   Component     │───▶│   Component     │───▶│   Component     │
│   (Unit Tests)  │    │   (Unit Tests)  │    │   (Unit Tests)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                Component Integration Layer                      │
│              (Interface Compatibility Tests)                   │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                 System Integration Layer                        │
│              (End-to-End Functionality Tests)                   │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Performance & Stress Layer                    │
│              (Load, Stress, and Robustness Tests)               │
└─────────────────────────────────────────────────────────────────┘
```

## Test Framework Architecture

### Modular Test Framework Design

```python
#!/usr/bin/env python3

import unittest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
import threading
import time
from typing import Dict, List, Any, Optional
import asyncio
from dataclasses import dataclass


@dataclass
class TestResult:
    """Data class for test results"""
    test_name: str
    success: bool
    duration: float
    error_message: Optional[str] = None
    details: Dict[str, Any] = None


class IntegrationTestFramework:
    """Modular framework for AI robot brain integration tests"""

    def __init__(self):
        self.test_results = []
        self.test_components = {}
        self.mock_components = {}
        self.test_scenarios = []
        self.performance_thresholds = {
            'perception_latency': 0.5,  # seconds
            'language_processing_time': 0.2,
            'action_execution_time': 1.0,
            'system_throughput': 10.0   # Hz
        }

    def register_component(self, name: str, component):
        """Register a component for testing"""
        self.test_components[name] = component

    def create_mock_component(self, name: str, interface_spec: Dict):
        """Create a mock component that implements the interface"""
        mock = Mock()

        # Set up mock methods based on interface specification
        for method_name, return_value in interface_spec.items():
            if callable(return_value):
                mock.configure_mock(**{method_name: return_value})
            else:
                mock.configure_mock(**{method_name: Mock(return_value=return_value)})

        self.mock_components[name] = mock
        return mock

    def add_test_scenario(self, name: str, test_func, description: str = ""):
        """Add a test scenario to the framework"""
        self.test_scenarios.append({
            'name': name,
            'function': test_func,
            'description': description,
            'setup_func': None,
            'teardown_func': None
        })

    def run_tests(self, test_names: List[str] = None) -> List[TestResult]:
        """Run all registered tests or specific tests"""
        results = []

        for scenario in self.test_scenarios:
            if test_names and scenario['name'] not in test_names:
                continue

            start_time = time.time()

            try:
                # Setup
                if scenario.get('setup_func'):
                    scenario['setup_func']()

                # Run test
                success = scenario['function']()

                # Teardown
                if scenario.get('teardown_func'):
                    scenario['teardown_func']()

                duration = time.time() - start_time

                results.append(TestResult(
                    test_name=scenario['name'],
                    success=success,
                    duration=duration
                ))

            except Exception as e:
                duration = time.time() - start_time
                results.append(TestResult(
                    test_name=scenario['name'],
                    success=False,
                    duration=duration,
                    error_message=str(e)
                ))

        self.test_results.extend(results)
        return results

    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of test results"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.success)
        failed_tests = total_tests - passed_tests

        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'total_duration': sum(r.duration for r in self.test_results)
        }


class PerceptionPipelineTester:
    """Specific tester for perception pipeline integration"""

    def __init__(self, framework: IntegrationTestFramework):
        self.framework = framework
        self.perception_results = []

    def test_vision_language_integration(self):
        """Test integration between vision and language components"""
        # Create mock vision component
        mock_vision = self.framework.create_mock_component('vision', {
            'detect_objects': lambda: [
                {'class': 'person', 'bbox': [100, 100, 200, 300], 'confidence': 0.85},
                {'class': 'chair', 'bbox': [300, 200, 150, 150], 'confidence': 0.72}
            ],
            'get_scene_description': lambda: "A person sitting on a chair in a room"
        })

        # Create mock language component
        mock_language = self.framework.create_mock_component('language', {
            'understand_command': lambda cmd: {'intent': 'find_person', 'entities': {'target': 'person'}},
            'generate_response': lambda context: "I see a person sitting on a chair"
        })

        # Test the integration
        vision_data = mock_vision.detect_objects()
        language_input = mock_vision.get_scene_description()
        language_output = mock_language.generate_response(language_input)

        # Validate results
        success = (
            len(vision_data) > 0 and
            'person' in language_input and
            'person' in language_output
        )

        return success

    def test_audio_vision_integration(self):
        """Test integration between audio and vision components"""
        # Create mock audio component
        mock_audio = self.framework.create_mock_component('audio', {
            'recognize_speech': lambda: {'text': 'Where is the person?', 'confidence': 0.85},
            'detect_sound': lambda: {'type': 'voice', 'direction': 45.0, 'confidence': 0.9}
        })

        # Create mock vision component
        mock_vision = self.framework.create_mock_component('vision', {
            'detect_objects': lambda: [
                {'class': 'person', 'bbox': [100, 100, 200, 300], 'confidence': 0.85, 'direction': 45.0}
            ]
        })

        # Test integration
        audio_result = mock_audio.recognize_speech()
        audio_direction = mock_audio.detect_sound()['direction']
        vision_result = mock_vision.detect_objects()

        # Check if vision detected object in audio direction
        person_detected_in_direction = any(
            abs(obj.get('direction', 0) - audio_direction) < 30 for obj in vision_result
        )

        success = (
            'person' in audio_result['text'].lower() and
            person_detected_in_direction
        )

        return success

    def test_multimodal_fusion(self):
        """Test multimodal fusion of vision, audio, and language"""
        # Create mock components for all modalities
        mock_vision = self.framework.create_mock_component('vision', {
            'detect_objects': lambda: [
                {'class': 'person', 'bbox': [100, 100, 200, 300], 'confidence': 0.85}
            ],
            'track_objects': lambda: [{'id': 1, 'class': 'person', 'position': (150, 200)}]
        })

        mock_audio = self.framework.create_mock_component('audio', {
            'recognize_speech': lambda: {'text': 'Hello person', 'confidence': 0.9}
        })

        mock_language = self.framework.create_mock_component('language', {
            'understand_command': lambda cmd: {'intent': 'greet', 'entities': {'target': 'person'}},
            'generate_response': lambda context: "Hello! How can I help you?"
        })

        # Simulate multimodal input
        vision_data = mock_vision.track_objects()
        audio_data = mock_audio.recognize_speech()

        # Create fusion context
        fusion_context = {
            'objects': vision_data,
            'speech': audio_data['text'],
            'intent': mock_language.understand_command(audio_data['text'])
        }

        response = mock_language.generate_response(fusion_context)

        success = (
            len(vision_data) > 0 and
            'person' in audio_data['text'] and
            'Hello' in response
        )

        return success


class ActionIntegrationTester:
    """Tester for action execution integration"""

    def __init__(self, framework: IntegrationTestFramework):
        self.framework = framework

    def test_navigation_integration(self):
        """Test navigation system integration with perception"""
        # Mock perception components
        mock_vision = self.framework.create_mock_component('vision', {
            'detect_obstacles': lambda: [
                {'type': 'chair', 'position': (2.0, 1.0, 0.0), 'size': (0.5, 0.5, 0.8)}
            ],
            'get_free_space': lambda: [(0.5, 0.5), (1.5, 0.5), (0.5, 1.5)]
        })

        # Mock navigation component
        mock_navigation = self.framework.create_mock_component('navigation', {
            'plan_path': lambda start, goal, obstacles: [(0, 0), (1, 0), (1, 1), (2, 1)],
            'execute_path': lambda path: {'status': 'success', 'duration': 5.0}
        })

        # Test integration
        obstacles = mock_vision.detect_obstacles()
        free_spaces = mock_vision.get_free_space()

        if free_spaces:
            goal = free_spaces[0]
            path = mock_navigation.plan_path((0, 0), goal, obstacles)
            result = mock_navigation.execute_path(path)

            success = result['status'] == 'success' and len(path) > 0
        else:
            success = False

        return success

    def test_manipulation_integration(self):
        """Test manipulation system integration with perception"""
        # Mock perception
        mock_vision = self.framework.create_mock_component('vision', {
            'detect_graspable_objects': lambda: [
                {'class': 'cup', 'position': (1.0, 0.5, 0.2), 'orientation': (0, 0, 0, 1)}
            ]
        })

        # Mock manipulation
        mock_manipulation = self.framework.create_mock_component('manipulation', {
            'plan_grasp': lambda obj: {'pose': (1.0, 0.5, 0.3), 'gripper_width': 0.05},
            'execute_grasp': lambda plan: {'status': 'success', 'grasped': True}
        })

        # Test integration
        objects = mock_vision.detect_graspable_objects()

        if objects:
            obj = objects[0]
            grasp_plan = mock_manipulation.plan_grasp(obj)
            result = mock_manipulation.execute_grasp(grasp_plan)

            success = result['status'] == 'success' and result.get('grasped', False)
        else:
            success = False

        return success


def create_integration_test_suite():
    """Create a comprehensive integration test suite"""
    framework = IntegrationTestFramework()
    perception_tester = PerceptionPipelineTester(framework)
    action_tester = ActionIntegrationTester(framework)

    # Register testers
    framework.register_component('perception_tester', perception_tester)
    framework.register_component('action_tester', action_tester)

    # Add perception integration tests
    framework.add_test_scenario(
        'vision_language_integration',
        perception_tester.test_vision_language_integration,
        'Test integration between vision and language components'
    )

    framework.add_test_scenario(
        'audio_vision_integration',
        perception_tester.test_audio_vision_integration,
        'Test integration between audio and vision components'
    )

    framework.add_test_scenario(
        'multimodal_fusion',
        perception_tester.test_multimodal_fusion,
        'Test multimodal fusion of vision, audio, and language'
    )

    # Add action integration tests
    framework.add_test_scenario(
        'navigation_integration',
        action_tester.test_navigation_integration,
        'Test navigation system integration with perception'
    )

    framework.add_test_scenario(
        'manipulation_integration',
        action_tester.test_manipulation_integration,
        'Test manipulation system integration with perception'
    )

    return framework


def run_integration_tests():
    """Run the complete integration test suite"""
    test_framework = create_integration_test_suite()
    results = test_framework.run_tests()

    # Print results
    summary = test_framework.get_test_summary()

    print(f"\n=== Integration Test Results ===")
    print(f"Total tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Success rate: {summary['success_rate']:.2%}")
    print(f"Total duration: {summary['total_duration']:.2f}s")

    print(f"\nDetailed Results:")
    for result in results:
        status = "PASS" if result.success else "FAIL"
        print(f"  {result.test_name}: {status} ({result.duration:.3f}s)")
        if not result.success and result.error_message:
            print(f"    Error: {result.error_message}")

    return results, summary


if __name__ == '__main__':
    run_integration_tests()
```

## Vision System Integration Tests

### Testing Vision Pipeline Components

```python
#!/usr/bin/env python3

import unittest
import numpy as np
import cv2
from unittest.mock import Mock, patch
import threading
import time
from typing import List, Dict, Any


class VisionIntegrationTests(unittest.TestCase):
    """Integration tests for vision system components"""

    def setUp(self):
        """Set up test fixtures"""
        # Create mock camera input
        self.mock_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.mock_depth = np.random.rand(480, 640).astype(np.float32)

        # Create mock vision components
        self.object_detector = Mock()
        self.feature_extractor = Mock()
        self.scene_analyzer = Mock()
        self.object_tracker = Mock()

    def test_object_detection_integration(self):
        """Test integration of object detection with other vision components"""
        from vision_pipeline import ObjectDetector, FeatureExtractor, SceneAnalyzer

        # Mock the actual implementations
        with patch('vision_pipeline.ObjectDetector') as mock_detector, \
             patch('vision_pipeline.FeatureExtractor') as mock_extractor, \
             patch('vision_pipeline.SceneAnalyzer') as mock_analyzer:

            # Configure mocks
            mock_detector.return_value.detect.return_value = [
                {'class': 'person', 'bbox': [100, 100, 200, 300], 'confidence': 0.85}
            ]
            mock_extractor.return_value.extract.return_value = np.random.rand(128).astype(np.float32)
            mock_analyzer.return_value.analyze.return_value = {'room_type': 'office', 'activity': 'working'}

            # Test integration
            detector = mock_detector.return_value
            extractor = mock_extractor.return_value
            analyzer = mock_analyzer.return_value

            # Run detection
            detections = detector.detect(self.mock_image, threshold=0.5)

            # Extract features from detected objects
            features = []
            for detection in detections:
                # Simulate cropping object region
                x, y, w, h = detection['bbox']
                obj_region = self.mock_image[y:y+h, x:x+w]
                feature = extractor.extract(obj_region)
                features.append(feature)

            # Analyze scene context
            scene_context = analyzer.analyze(self.mock_image, detections)

            # Validate integration results
            self.assertTrue(len(detections) > 0, "Should detect objects")
            self.assertTrue(len(features) == len(detections), "Should extract features for each detection")
            self.assertIn('room_type', scene_context, "Should analyze scene context")

            # Verify method calls
            mock_detector.assert_called_once()
            mock_extractor.assert_called()
            mock_analyzer.assert_called_once()

    def test_depth_perception_integration(self):
        """Test integration of depth perception with object detection"""
        from vision_pipeline import DepthPerception, ObjectDetector

        with patch('vision_pipeline.DepthPerception') as mock_depth_perception, \
             patch('vision_pipeline.ObjectDetector') as mock_detector:

            # Configure mocks
            mock_depth_perception.return_value.process_depth.return_value = [
                {'type': 'obstacle', 'distance': 1.5, 'position': (2.0, 1.0, 0.5)}
            ]
            mock_detector.return_value.detect.return_value = [
                {'class': 'chair', 'bbox': [150, 200, 100, 100], 'confidence': 0.75}
            ]

            depth_processor = mock_depth_perception.return_value
            detector = mock_detector.return_value

            # Process depth and visual data
            depth_results = depth_processor.process_depth(self.mock_depth)
            visual_results = detector.detect(self.mock_image, threshold=0.5)

            # Integrate depth and visual information
            integrated_results = self.integrate_depth_visual(depth_results, visual_results)

            # Validate integration
            self.assertTrue(len(integrated_results) > 0, "Should integrate depth and visual data")
            for result in integrated_results:
                self.assertIn('distance', result, "Should include distance information")
                self.assertIn('class', result, "Should include object class")

            # Verify method calls
            mock_depth_perception.assert_called_once()
            mock_detector.assert_called_once()

    def integrate_depth_visual(self, depth_results: List[Dict], visual_results: List[Dict]) -> List[Dict]:
        """Helper method to integrate depth and visual results"""
        integrated = []

        for depth_obj in depth_results:
            for visual_obj in visual_results:
                # Simple integration based on position correlation
                if self.is_correlated(depth_obj, visual_obj):
                    integrated.append({
                        **depth_obj,
                        **visual_obj,
                        'integrated': True
                    })

        return integrated

    def is_correlated(self, depth_obj: Dict, visual_obj: Dict) -> bool:
        """Check if depth and visual objects are correlated"""
        # Simplified correlation check
        depth_x, depth_y = depth_obj.get('position', (0, 0))[:2]
        visual_bbox = visual_obj.get('bbox', [0, 0, 0, 0])
        visual_x, visual_y = visual_bbox[0] + visual_bbox[2]//2, visual_bbox[1] + visual_bbox[3]//2

        distance = np.sqrt((depth_x - visual_x)**2 + (depth_y - visual_y)**2)
        return distance < 100  # Threshold for correlation

    def test_object_tracking_integration(self):
        """Test integration of object tracking with detection"""
        from vision_pipeline import ObjectTracker

        with patch('vision_pipeline.ObjectTracker') as mock_tracker:
            # Configure mock
            mock_tracker.return_value.update.return_value = {
                'obj_1': {'class': 'person', 'position': (100, 100), 'velocity': (5, 0)}
            }

            tracker = mock_tracker.return_value

            # Simulate tracking over multiple frames
            detections_sequence = [
                [{'class': 'person', 'bbox': [95, 95, 50, 100], 'confidence': 0.8}],
                [{'class': 'person', 'bbox': [100, 100, 50, 100], 'confidence': 0.85}],
                [{'class': 'person', 'bbox': [105, 100, 50, 100], 'confidence': 0.82}]
            ]

            tracked_objects = {}
            for i, detections in enumerate(detections_sequence):
                timestamp = i * 0.1  # 10Hz
                tracked = tracker.update(detections, timestamp)
                tracked_objects.update(tracked)

            # Validate tracking results
            self.assertTrue(len(tracked_objects) > 0, "Should track objects")
            for obj_id, obj_data in tracked_objects.items():
                self.assertIn('velocity', obj_data, "Should calculate velocity")
                self.assertGreaterEqual(abs(obj_data['velocity'][0]), 0, "Should have calculated movement")

            # Verify method calls
            self.assertEqual(mock_tracker.return_value.update.call_count, len(detections_sequence))

    def test_feature_matching_integration(self):
        """Test integration of feature extraction and matching"""
        from vision_pipeline import FeatureExtractor

        with patch('vision_pipeline.FeatureExtractor') as mock_extractor:
            # Configure mock
            mock_extractor.return_value.extract.return_value = np.random.rand(128).astype(np.float32)

            extractor = mock_extractor.return_value

            # Extract features from multiple images
            images = [self.mock_image, self.mock_image.copy(), self.mock_image.copy()]
            features_list = []

            for img in images:
                # Add some variation to simulate different views
                img_varied = img + np.random.randint(-10, 10, img.shape, dtype=np.int16)
                img_varied = np.clip(img_varied, 0, 255).astype(np.uint8)

                features = extractor.extract(img_varied)
                features_list.append(features)

            # Test feature matching
            matches = self.match_features(features_list)

            # Validate matching results
            self.assertTrue(len(matches) > 0, "Should find feature matches")
            for match in matches:
                self.assertIn('similarity', match, "Should calculate similarity")
                self.assertGreaterEqual(match['similarity'], 0, "Similarity should be non-negative")
                self.assertLessEqual(match['similarity'], 1, "Similarity should be <= 1")

            # Verify method calls
            self.assertEqual(mock_extractor.return_value.extract.call_count, len(images))

    def match_features(self, features_list: List[np.ndarray]) -> List[Dict]:
        """Helper method to match features between feature vectors"""
        matches = []

        for i in range(len(features_list)):
            for j in range(i + 1, len(features_list)):
                # Calculate cosine similarity
                features1, features2 = features_list[i], features_list[j]
                similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))

                matches.append({
                    'image1_idx': i,
                    'image2_idx': j,
                    'similarity': float(similarity)
                })

        return matches

    def test_real_time_vision_pipeline(self):
        """Test real-time vision pipeline with continuous processing"""
        # Simulate real-time processing
        processing_times = []
        frame_count = 0

        start_time = time.time()

        # Simulate processing multiple frames
        for _ in range(100):  # Process 100 frames
            frame_start = time.time()

            # Simulate vision processing
            detections = self.simulate_detection(self.mock_image)
            features = self.simulate_feature_extraction(self.mock_image)
            analysis = self.simulate_scene_analysis(self.mock_image, detections)

            frame_time = time.time() - frame_start
            processing_times.append(frame_time)
            frame_count += 1

            # Simulate real-time constraint (30 FPS = 33.3ms per frame)
            if frame_time < 0.033:
                time.sleep(0.033 - frame_time)  # Maintain 30 FPS

        total_time = time.time() - start_time
        avg_processing_time = sum(processing_times) / len(processing_times)
        achieved_fps = frame_count / total_time

        # Validate real-time performance
        self.assertLess(avg_processing_time, 0.1, "Should process frames in less than 100ms")
        self.assertGreaterEqual(achieved_fps, 10, f"Should achieve at least 10 FPS, got {achieved_fps:.2f}")

        print(f"Real-time test results: {achieved_fps:.2f} FPS, avg processing time: {avg_processing_time*1000:.1f}ms")

    def simulate_detection(self, image):
        """Simulate object detection"""
        # Simulate detection with some randomness
        if np.random.random() > 0.3:  # 70% chance of detecting something
            return [{'class': 'object', 'bbox': [100, 100, 50, 50], 'confidence': 0.8}]
        return []

    def simulate_feature_extraction(self, image):
        """Simulate feature extraction"""
        return np.random.rand(64).astype(np.float32)

    def simulate_scene_analysis(self, image, detections):
        """Simulate scene analysis"""
        return {'object_count': len(detections), 'scene_type': 'indoor'}


def create_vision_integration_suite():
    """Create a test suite for vision integration tests"""
    suite = unittest.TestSuite()

    # Add all test methods
    loader = unittest.TestLoader()
    suite.addTests(loader.loadTestsFromTestCase(VisionIntegrationTests))

    return suite


def run_vision_integration_tests():
    """Run vision integration tests"""
    suite = create_vision_integration_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    run_vision_integration_tests()
```

## Language Understanding Integration Tests

### Testing Language System Integration

```python
#!/usr/bin/env python3

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import asyncio
import time
from typing import Dict, List, Any


class LanguageIntegrationTests(unittest.TestCase):
    """Integration tests for language understanding components"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_commands = [
            "Go to the kitchen",
            "Get me the red cup",
            "What time is it?",
            "Hello how are you?",
            "Navigate to the person"
        ]

        self.mock_vision_data = {
            'objects': [
                {'class': 'cup', 'color': 'red', 'position': (1.0, 2.0, 0.0)},
                {'class': 'person', 'position': (0.0, 0.0, 0.0)}
            ],
            'room_type': 'kitchen'
        }

    def test_command_parsing_integration(self):
        """Test integration of command parsing with context awareness"""
        from language_pipeline import CommandParser, ContextManager

        with patch('language_pipeline.CommandParser') as mock_parser, \
             patch('language_pipeline.ContextManager') as mock_context:

            # Configure mocks
            mock_parser.return_value.parse.return_value = {
                'intent': 'navigation',
                'entities': {'destination': 'kitchen'},
                'confidence': 0.85
            }
            mock_context.return_value.get_context.return_value = {
                'current_location': 'living_room',
                'available_destinations': ['kitchen', 'bedroom', 'office']
            }

            parser = mock_parser.return_value
            context_manager = mock_context.return_value

            # Test with multiple commands
            for command in self.test_commands:
                # Parse command
                parsed = parser.parse(command)

                # Get context
                context = context_manager.get_context()

                # Integrate context with parsed command
                integrated_result = self.integrate_command_context(parsed, context)

                # Validate integration
                self.assertIsNotNone(integrated_result['intent'], "Should have intent")
                self.assertGreaterEqual(integrated_result['confidence'], 0, "Should have confidence")
                self.assertIn('context_enriched', integrated_result, "Should be context-enriched")

            # Verify method calls
            self.assertEqual(mock_parser.return_value.parse.call_count, len(self.test_commands))
            self.assertEqual(mock_context.return_value.get_context.call_count, len(self.test_commands))

    def integrate_command_context(self, parsed_command: Dict, context: Dict) -> Dict:
        """Helper to integrate command parsing with context"""
        integrated = {**parsed_command}
        integrated['context_enriched'] = True
        integrated['available_options'] = context.get('available_destinations', [])

        # Validate that parsed command is compatible with context
        if parsed_command.get('intent') == 'navigation':
            destination = parsed_command.get('entities', {}).get('destination')
            if destination and destination not in integrated['available_options']:
                integrated['validation_error'] = f"Destination {destination} not available"

        return integrated

    def test_vision_language_fusion(self):
        """Test fusion of vision and language information"""
        from language_pipeline import LanguageUnderstanding
        from vision_pipeline import VisionSystem

        with patch('language_pipeline.LanguageUnderstanding') as mock_language, \
             patch('vision_pipeline.VisionSystem') as mock_vision:

            # Configure mocks
            mock_language.return_value.understand.return_value = {
                'intent': 'find_object',
                'entities': {'target': 'red cup'},
                'confidence': 0.9
            }
            mock_vision.return_value.get_scene_description.return_value = self.mock_vision_data

            language_processor = mock_language.return_value
            vision_processor = mock_vision.return_value

            # Get vision data
            vision_context = vision_processor.get_scene_description()

            # Process language command
            for command in ["Get me the red cup", "Find the red cup"]:
                language_result = language_processor.understand(command)

                # Fuse vision and language data
                fused_result = self.fuse_vision_language(vision_context, language_result)

                # Validate fusion
                self.assertTrue(fused_result['object_found'], "Should find the object")
                self.assertIn('object_location', fused_result, "Should provide object location")
                self.assertGreaterEqual(fused_result['confidence'], 0.7, "Should have good confidence")

        # Verify method calls
        mock_language.return_value.understand.assert_called()
        mock_vision.return_value.get_scene_description.assert_called_once()

    def fuse_vision_language(self, vision_data: Dict, language_result: Dict) -> Dict:
        """Fuse vision and language information"""
        fused = {
            'intent': language_result['intent'],
            'target_object': language_result.get('entities', {}).get('target'),
            'object_found': False,
            'object_location': None,
            'confidence': language_result['confidence']
        }

        # Search for target object in vision data
        target = fused['target_object']
        if target:
            for obj in vision_data.get('objects', []):
                if target.lower() in f"{obj.get('color', '')} {obj['class']}".lower():
                    fused['object_found'] = True
                    fused['object_location'] = obj.get('position')
                    break

        return fused

    def test_conversation_context_integration(self):
        """Test integration of conversation context management"""
        from language_pipeline import ConversationManager

        with patch('language_pipeline.ConversationManager') as mock_conversation:
            # Configure mock
            mock_conversation.return_value.process.return_value = {
                'response': 'Hello! How can I help you?',
                'intent': 'greeting',
                'entities': {}
            }
            mock_conversation.return_value.update_context.return_value = {
                'current_topic': 'greeting',
                'user_satisfaction': 0.8
            }

            conv_manager = mock_conversation.return_value

            # Simulate a conversation
            conversation_history = [
                ("Hello", "greeting"),
                ("How are you?", "wellbeing_check"),
                ("What can you do?", "capability_inquiry")
            ]

            context_history = []

            for user_input, expected_intent in conversation_history:
                # Process user input
                response = conv_manager.process(user_input)

                # Update conversation context
                context = conv_manager.update_context(user_input, response['response'])
                context_history.append(context)

                # Validate response
                self.assertIsNotNone(response['response'], "Should generate response")
                self.assertIsNotNone(response['intent'], "Should identify intent")

            # Validate context evolution
            self.assertGreaterEqual(len(context_history), len(conversation_history), "Should maintain context history")
            for context in context_history:
                self.assertIn('current_topic', context, "Should track conversation topic")

            # Verify method calls
            self.assertEqual(mock_conversation.return_value.process.call_count, len(conversation_history))
            self.assertEqual(mock_conversation.return_value.update_context.call_count, len(conversation_history))

    def test_multilingual_integration(self):
        """Test integration with multilingual support"""
        from language_pipeline import MultilingualProcessor

        with patch('language_pipeline.MultilingualProcessor') as mock_multilingual:
            # Configure mock for different languages
            language_responses = {
                'en': {'text': 'Hello', 'language': 'en', 'confidence': 0.95},
                'es': {'text': 'Hola', 'language': 'es', 'confidence': 0.92},
                'fr': {'text': 'Bonjour', 'language': 'fr', 'confidence': 0.90}
            }

            def mock_process(text, target_lang):
                return language_responses.get(target_lang, language_responses['en'])

            mock_multilingual.return_value.translate.side_effect = mock_process
            mock_multilingual.return_value.detect_language.return_value = 'en'

            multilingual_processor = mock_multilingual.return_value

            # Test translation for different languages
            test_phrases = ['hello', 'thank you', 'help']
            target_languages = ['es', 'fr', 'de']

            for phrase in test_phrases:
                for lang in target_languages:
                    # Detect language
                    detected_lang = multilingual_processor.detect_language(phrase)

                    # Translate to target language
                    translation = multilingual_processor.translate(phrase, lang)

                    # Validate results
                    self.assertIsNotNone(translation['text'], "Should translate text")
                    self.assertEqual(translation['language'], lang, "Should use correct target language")
                    self.assertGreaterEqual(translation['confidence'], 0.8, "Should have good confidence")

        # Verify method calls
        expected_calls = len(test_phrases) * len(target_languages)
        self.assertEqual(mock_multilingual.return_value.translate.call_count, expected_calls)
        self.assertEqual(mock_multilingual.return_value.detect_language.call_count, len(test_phrases))

    def test_language_action_mapping(self):
        """Test mapping of language understanding to action execution"""
        from language_pipeline import LanguageUnderstanding
        from action_pipeline import ActionExecutor

        with patch('language_pipeline.LanguageUnderstanding') as mock_language, \
             patch('action_pipeline.ActionExecutor') as mock_action:

            # Configure mocks
            mock_language.return_value.understand.return_value = {
                'intent': 'navigation',
                'entities': {'destination': 'kitchen'},
                'confidence': 0.88
            }
            mock_action.return_value.execute.return_value = {
                'status': 'success',
                'execution_time': 2.5,
                'details': 'Navigated to kitchen successfully'
            }

            language_processor = mock_language.return_value
            action_executor = mock_action.return_value

            # Test language-to-action pipeline
            test_commands = [
                "Go to the kitchen",
                "Navigate to the living room",
                "Move to the bedroom"
            ]

            for command in test_commands:
                # Process language
                language_result = language_processor.understand(command)

                # Execute corresponding action
                action_result = action_executor.execute(language_result)

                # Validate the pipeline
                self.assertIsNotNone(language_result['intent'], "Should understand intent")
                self.assertEqual(action_result['status'], 'success', "Should execute successfully")
                self.assertGreaterEqual(language_result['confidence'], 0.7, "Should have good confidence")

        # Verify method calls
        self.assertEqual(mock_language.return_value.understand.call_count, len(test_commands))
        self.assertEqual(mock_action.return_value.execute.call_count, len(test_commands))

    def test_performance_under_load(self):
        """Test language system performance under load"""
        from language_pipeline import LanguageUnderstanding

        with patch('language_pipeline.LanguageUnderstanding') as mock_language:
            # Configure mock with realistic response times
            def slow_understand(text):
                time.sleep(0.05)  # Simulate processing time
                return {
                    'intent': 'test',
                    'entities': {},
                    'confidence': 0.8
                }

            mock_language.return_value.understand.side_effect = slow_understand

            language_processor = mock_language.return_value

            # Test with concurrent requests (simulated)
            start_time = time.time()
            request_count = 50

            for i in range(request_count):
                result = language_processor.understand(f"test command {i}")
                self.assertIsNotNone(result['intent'], f"Request {i} should succeed")

            total_time = time.time() - start_time
            avg_time = total_time / request_count
            throughput = request_count / total_time

            # Validate performance
            self.assertLess(avg_time, 0.1, f"Should process requests in less than 100ms, got {avg_time:.3f}s")
            self.assertGreaterEqual(throughput, 5, f"Should handle at least 5 requests/sec, got {throughput:.2f}")

            print(f"Performance test: {throughput:.2f} req/s, avg {avg_time*1000:.1f}ms per request")


def create_language_integration_suite():
    """Create a test suite for language integration tests"""
    suite = unittest.TestSuite()

    # Add all test methods
    loader = unittest.TestLoader()
    suite.addTests(loader.loadTestsFromTestCase(LanguageIntegrationTests))

    return suite


def run_language_integration_tests():
    """Run language integration tests"""
    suite = create_language_integration_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    run_language_integration_tests()
```

## Action Execution Integration Tests

### Testing Action System Integration

```python
#!/usr/bin/env python3

import unittest
import numpy as np
from unittest.mock import Mock, patch
import time
from typing import Dict, List, Any


class ActionIntegrationTests(unittest.TestCase):
    """Integration tests for action execution components"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_navigation_goals = [
            (1.0, 1.0, 0.0),
            (2.0, 0.0, 1.57),
            (-1.0, -1.0, 3.14)
        ]

        self.mock_object_poses = [
            {'position': (0.5, 0.5, 0.0), 'orientation': (0, 0, 0, 1), 'class': 'cup'},
            {'position': (1.5, 1.0, 0.0), 'orientation': (0, 0, 0, 1), 'class': 'book'}
        ]

    def test_navigation_integration(self):
        """Test integration of navigation with perception and planning"""
        from action_pipeline import NavigationSystem, PathPlanner
        from vision_pipeline import ObstacleDetector

        with patch('action_pipeline.NavigationSystem') as mock_nav, \
             patch('action_pipeline.PathPlanner') as mock_planner, \
             patch('vision_pipeline.ObstacleDetector') as mock_obstacle:

            # Configure mocks
            mock_obstacle.return_value.detect.return_value = [
                {'type': 'chair', 'position': (1.2, 0.8, 0.0), 'size': (0.5, 0.5, 0.8)}
            ]
            mock_planner.return_value.plan_path.return_value = [(0, 0), (0.5, 0.5), (1.0, 1.0)]
            mock_nav.return_value.execute_path.return_value = {'status': 'success', 'duration': 3.2}

            obstacle_detector = mock_obstacle.return_value
            path_planner = mock_planner.return_value
            navigation_system = mock_nav.return_value

            # Test navigation integration
            for goal in self.mock_navigation_goals:
                # Detect obstacles
                obstacles = obstacle_detector.detect()

                # Plan path considering obstacles
                path = path_planner.plan_path((0, 0), goal[:2], obstacles)

                # Execute navigation
                result = navigation_system.execute_path(path)

                # Validate integration
                self.assertEqual(result['status'], 'success', f"Navigation to {goal} should succeed")
                self.assertGreaterEqual(len(path), 1, "Should generate valid path")
                self.assertGreater(result['duration'], 0, "Should record execution time")

        # Verify method calls
        self.assertEqual(mock_obstacle.return_value.detect.call_count, len(self.mock_navigation_goals))
        self.assertEqual(mock_planner.return_value.plan_path.call_count, len(self.mock_navigation_goals))
        self.assertEqual(mock_nav.return_value.execute_path.call_count, len(self.mock_navigation_goals))

    def test_manipulation_integration(self):
        """Test integration of manipulation with perception and planning"""
        from action_pipeline import ManipulationSystem, GraspPlanner
        from vision_pipeline import ObjectDetector

        with patch('action_pipeline.ManipulationSystem') as mock_manipulation, \
             patch('action_pipeline.GraspPlanner') as mock_grasp_planner, \
             patch('vision_pipeline.ObjectDetector') as mock_vision:

            # Configure mocks
            mock_vision.return_value.detect_graspable.return_value = self.mock_object_poses
            mock_grasp_planner.return_value.plan_grasp.return_value = {
                'position': (0.5, 0.5, 0.3),
                'orientation': (0, 0, 0, 1),
                'gripper_width': 0.05
            }
            mock_manipulation.return_value.execute_grasp.return_value = {
                'status': 'success',
                'grasped_object': 'cup',
                'success': True
            }

            vision_system = mock_vision.return_value
            grasp_planner = mock_grasp_planner.return_value
            manipulation_system = mock_manipulation.return_value

            # Test manipulation integration
            objects = vision_system.detect_graspable()

            for obj in objects:
                # Plan grasp for object
                grasp_plan = grasp_planner.plan_grasp(obj)

                # Execute grasp
                result = manipulation_system.execute_grasp(grasp_plan)

                # Validate integration
                self.assertTrue(result['success'], f"Grasping {obj['class']} should succeed")
                self.assertEqual(result['grasped_object'], obj['class'], "Should grasp correct object")
                self.assertIsNotNone(grasp_plan['position'], "Should generate grasp position")

        # Verify method calls
        mock_vision.return_value.detect_graspable.assert_called_once()
        self.assertEqual(mock_grasp_planner.return_value.plan_grasp.call_count, len(self.mock_object_poses))
        self.assertEqual(mock_manipulation.return_value.execute_grasp.call_count, len(self.mock_object_poses))

    def test_social_action_integration(self):
        """Test integration of social actions with perception and context"""
        from action_pipeline import SocialActionSystem
        from vision_pipeline import PersonDetector

        with patch('action_pipeline.SocialActionSystem') as mock_social, \
             patch('vision_pipeline.PersonDetector') as mock_person_detector:

            # Configure mocks
            mock_person_detector.return_value.detect_people.return_value = [
                {'id': 1, 'position': (1.0, 0.0, 0.0), 'orientation': 0.0},
                {'id': 2, 'position': (2.0, 1.0, 0.0), 'orientation': 1.57}
            ]
            mock_social.return_value.execute_social_action.return_value = {
                'status': 'success',
                'action_performed': 'wave',
                'target_person': 1
            }

            person_detector = mock_person_detector.return_value
            social_system = mock_social.return_value

            # Test social action integration
            people = person_detector.detect_people()

            # Test greeting action
            greeting_result = social_system.execute_social_action('greet', people[0]['position'])
            self.assertEqual(greeting_result['status'], 'success', "Greeting should succeed")
            self.assertEqual(greeting_result['action_performed'], 'wave', "Should perform wave action")

            # Test attention action
            attention_result = social_system.execute_social_action('attend', people[1]['position'])
            self.assertEqual(attention_result['status'], 'success', "Attention should succeed")

            # Test follow action
            follow_result = social_system.execute_social_action('follow', people[0]['position'])
            self.assertEqual(follow_result['status'], 'success', "Follow should succeed")

        # Verify method calls
        mock_person_detector.return_value.detect_people.assert_called_once()
        self.assertEqual(mock_social.return_value.execute_social_action.call_count, 3)

    def test_action_sequence_integration(self):
        """Test integration of sequential actions"""
        from action_pipeline import ActionSequencer, NavigationSystem, ManipulationSystem

        with patch('action_pipeline.ActionSequencer') as mock_sequencer, \
             patch('action_pipeline.NavigationSystem') as mock_nav, \
             patch('action_pipeline.ManipulationSystem') as mock_manip:

            # Configure mocks
            mock_nav.return_value.execute_path.return_value = {'status': 'success', 'duration': 2.0}
            mock_manip.return_value.execute_grasp.return_value = {'status': 'success', 'success': True}
            mock_sequencer.return_value.execute_sequence.return_value = [
                {'action': 'navigate', 'status': 'success'},
                {'action': 'grasp', 'status': 'success'},
                {'action': 'return', 'status': 'success'}
            ]

            sequencer = mock_sequencer.return_value

            # Define action sequence
            action_sequence = [
                {'action': 'navigate', 'params': {'destination': (1.0, 1.0, 0.0)}},
                {'action': 'grasp', 'params': {'object': 'cup'}},
                {'action': 'navigate', 'params': {'destination': (0.0, 0.0, 0.0)}}
            ]

            # Execute sequence
            sequence_result = sequencer.execute_sequence(action_sequence)

            # Validate sequence execution
            self.assertEqual(len(sequence_result), len(action_sequence), "Should execute all actions")
            for result in sequence_result:
                self.assertEqual(result['status'], 'success', f"Action {result['action']} should succeed")

            # Verify method calls
            mock_sequencer.return_value.execute_sequence.assert_called_once_with(action_sequence)

    def test_safety_integration(self):
        """Test integration of safety checks with action execution"""
        from action_pipeline import SafetyMonitor, ActionExecutor
        from vision_pipeline import SafetyPerception

        with patch('action_pipeline.SafetyMonitor') as mock_safety, \
             patch('action_pipeline.ActionExecutor') as mock_executor, \
             patch('vision_pipeline.SafetyPerception') as mock_safety_perception:

            # Configure mocks
            mock_safety_perception.return_value.check_safety.return_value = {
                'safe_to_proceed': True,
                'obstacles': [],
                'people_nearby': 0,
                'risk_level': 'low'
            }
            mock_safety.return_value.validate_action.return_value = {'safe': True, 'reason': 'All clear'}
            mock_executor.return_value.execute.return_value = {'status': 'success', 'details': 'Action completed safely'}

            safety_perception = mock_safety_perception.return_value
            safety_monitor = mock_safety.return_value
            action_executor = mock_executor.return_value

            # Test safety-integrated action execution
            test_actions = [
                {'type': 'navigation', 'destination': (1.0, 1.0, 0.0)},
                {'type': 'manipulation', 'object': 'cup'},
                {'type': 'greeting', 'person': 1}
            ]

            for action in test_actions:
                # Check safety
                safety_status = safety_perception.check_safety()
                self.assertTrue(safety_status['safe_to_proceed'], f"Environment should be safe for {action['type']}")

                # Validate action safety
                validation = safety_monitor.validate_action(action)
                self.assertTrue(validation['safe'], f"Action {action['type']} should be validated as safe")

                # Execute action
                result = action_executor.execute(action)
                self.assertEqual(result['status'], 'success', f"Action {action['type']} should execute successfully")

        # Verify method calls
        self.assertEqual(mock_safety_perception.return_value.check_safety.call_count, len(test_actions))
        self.assertEqual(mock_safety.return_value.validate_action.call_count, len(test_actions))
        self.assertEqual(mock_executor.return_value.execute.call_count, len(test_actions))

    def test_action_context_integration(self):
        """Test integration of action execution with context awareness"""
        from action_pipeline import ContextAwareActionExecutor
        from memory_system import ContextManager

        with patch('action_pipeline.ContextAwareActionExecutor') as mock_context_action, \
             patch('memory_system.ContextManager') as mock_context:

            # Configure mocks
            mock_context.return_value.get_context.return_value = {
                'current_task': 'fetch_item',
                'user_preferences': {'handedness': 'right', 'speed_preference': 'normal'},
                'environment_state': {'room': 'kitchen', 'lighting': 'bright'}
            }
            mock_context_action.return_value.execute_contextual_action.return_value = {
                'status': 'success',
                'adaptation_applied': True,
                'execution_time': 2.5
            }

            context_manager = mock_context.return_value
            context_action_executor = mock_context_action.return_value

            # Test context-aware action execution
            actions_with_context = [
                {'action': 'navigate', 'target': 'fridge', 'context_required': True},
                {'action': 'grasp', 'object': 'mug', 'context_required': True},
                {'action': 'deliver', 'recipient': 'user', 'context_required': True}
            ]

            for action in actions_with_context:
                # Get current context
                context = context_manager.get_context()

                # Execute context-aware action
                result = context_action_executor.execute_contextual_action(action, context)

                # Validate context-aware execution
                self.assertEqual(result['status'], 'success', f"Contextual action should succeed")
                self.assertTrue(result['adaptation_applied'], "Should apply context-based adaptations")
                self.assertIn('execution_time', result, "Should record execution metrics")

        # Verify method calls
        self.assertEqual(mock_context.return_value.get_context.call_count, len(actions_with_context))
        self.assertEqual(mock_context_action.return_value.execute_contextual_action.call_count, len(actions_with_context))

    def test_performance_stress_test(self):
        """Test action system performance under stress"""
        from action_pipeline import ActionExecutor

        with patch('action_pipeline.ActionExecutor') as mock_executor:
            # Configure mock with realistic execution times
            def slow_execute(action):
                time.sleep(0.1)  # Simulate execution time
                return {'status': 'success', 'execution_time': 0.1}

            mock_executor.return_value.execute.side_effect = slow_execute

            action_executor = mock_executor.return_value

            # Test with high load
            start_time = time.time()
            action_count = 100
            actions = [{'type': f'action_{i}', 'params': {}} for i in range(action_count)]

            execution_times = []
            for action in actions:
                exec_start = time.time()
                result = action_executor.execute(action)
                exec_time = time.time() - exec_start
                execution_times.append(exec_time)

                self.assertEqual(result['status'], 'success', f"Action {action['type']} should succeed")

            total_time = time.time() - start_time
            avg_time = sum(execution_times) / len(execution_times)
            throughput = action_count / total_time

            # Validate performance under stress
            self.assertLess(avg_time, 0.2, f"Should execute actions in less than 200ms under load, got {avg_time:.3f}s")
            self.assertGreaterEqual(throughput, 2, f"Should handle at least 2 actions/sec under load, got {throughput:.2f}")

            print(f"Stress test: {throughput:.2f} actions/s, avg {avg_time*1000:.1f}ms per action")


def create_action_integration_suite():
    """Create a test suite for action integration tests"""
    suite = unittest.TestSuite()

    # Add all test methods
    loader = unittest.TestLoader()
    suite.addTests(loader.loadTestsFromTestCase(ActionIntegrationTests))

    return suite


def run_action_integration_tests():
    """Run action integration tests"""
    suite = create_action_integration_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    run_action_integration_tests()
```

## End-to-End Integration Tests

### Complete System Integration Testing

```python
#!/usr/bin/env python3

import unittest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import asyncio
import time
from typing import Dict, List, Any
import threading
import queue


class EndToEndIntegrationTests(unittest.TestCase):
    """End-to-end integration tests for complete AI robot brain system"""

    def setUp(self):
        """Set up test fixtures for end-to-end tests"""
        self.test_scenarios = [
            {
                'name': 'fetch_object_basic',
                'input': 'Get me the red cup from the table',
                'expected_actions': ['navigate', 'grasp', 'return'],
                'environment': {'objects': ['red cup', 'table'], 'layout': 'kitchen'}
            },
            {
                'name': 'navigation_simple',
                'input': 'Go to the kitchen',
                'expected_actions': ['navigate'],
                'environment': {'rooms': ['kitchen', 'living room'], 'layout': 'apartment'}
            },
            {
                'name': 'social_interaction',
                'input': 'Hello how are you?',
                'expected_actions': ['greet'],
                'environment': {'people': 1, 'layout': 'office'}
            }
        ]

    def test_perception_action_pipeline(self):
        """Test complete perception-to-action pipeline"""
        from ai_robot_system import (
            VisionSystem, AudioSystem, LanguageSystem,
            ActionSystem, IntegrationCoordinator
        )

        with patch.multiple('ai_robot_system',
                           VisionSystem=Mock(),
                           AudioSystem=Mock(),
                           LanguageSystem=Mock(),
                           ActionSystem=Mock(),
                           IntegrationCoordinator=Mock()):

            # Configure system mocks
            VisionSystem.detect_objects.return_value = [
                {'class': 'cup', 'color': 'red', 'position': (1.0, 2.0, 0.0), 'confidence': 0.85}
            ]
            AudioSystem.recognize_speech.return_value = {
                'text': 'Get me the red cup',
                'confidence': 0.9
            }
            LanguageSystem.understand.return_value = {
                'intent': 'fetch_object',
                'entities': {'object': 'red cup'},
                'confidence': 0.85
            }
            ActionSystem.execute.return_value = {
                'status': 'success',
                'action_sequence': ['navigate_to_object', 'grasp_object', 'return_to_user']
            }
            IntegrationCoordinator.coordinate.return_value = {
                'final_status': 'completed',
                'execution_time': 15.5,
                'success_rate': 1.0
            }

            # Test the complete pipeline
            for scenario in self.test_scenarios:
                if 'cup' in scenario['input'].lower():
                    # Simulate the complete pipeline
                    vision_data = VisionSystem.detect_objects()
                    audio_data = AudioSystem.recognize_speech()
                    language_result = LanguageSystem.understand(audio_data['text'])
                    action_result = ActionSystem.execute(language_result)
                    coordination_result = IntegrationCoordinator.coordinate(
                        vision_data, audio_data, language_result, action_result
                    )

                    # Validate complete pipeline
                    self.assertEqual(coordination_result['final_status'], 'completed')
                    self.assertGreaterEqual(coordination_result['success_rate'], 0.8)
                    self.assertGreater(coordination_result['execution_time'], 0)

                    # Verify all system components were called
                    VisionSystem.detect_objects.assert_called()
                    AudioSystem.recognize_speech.assert_called()
                    LanguageSystem.understand.assert_called()
                    ActionSystem.execute.assert_called()
                    IntegrationCoordinator.coordinate.assert_called()

    def test_conversation_flow_integration(self):
        """Test complete conversation flow integration"""
        from ai_robot_system import ConversationManager, MultimodalFusion

        with patch('ai_robot_system.ConversationManager') as mock_conv, \
             patch('ai_robot_system.MultimodalFusion') as mock_fusion:

            # Configure conversation mock
            conversation_flows = [
                (['Hello'], ['Hi there! How can I help you?']),
                (['Get me water'], ['I will get you water. Where is the water?']),
                (['In the kitchen'], ['Going to kitchen to get water.'])
            ]

            def mock_process_turn(user_input, context):
                if 'Hello' in user_input:
                    return {'response': 'Hi there! How can I help you?', 'intent': 'greeting'}
                elif 'water' in user_input and 'get' in user_input:
                    return {'response': 'I will get you water. Where is the water?', 'intent': 'fetch_request'}
                elif 'kitchen' in user_input:
                    return {'response': 'Going to kitchen to get water.', 'intent': 'navigation_command'}
                else:
                    return {'response': 'I understand.', 'intent': 'acknowledgment'}

            mock_conv.return_value.process_turn.side_effect = mock_process_turn
            mock_fusion.return_value.fuse.return_value = {
                'context_enriched_response': 'Going to kitchen to get water.',
                'confidence': 0.92,
                'next_action': 'navigate_to_kitchen'
            }

            conversation_manager = mock_conv.return_value
            fusion_system = mock_fusion.return_value

            # Test multi-turn conversation
            conversation_history = [
                "Hello",
                "Get me water",
                "In the kitchen"
            ]

            context = {}
            for user_input in conversation_history:
                # Process turn
                turn_result = conversation_manager.process_turn(user_input, context)

                # Fuse with multimodal context
                fusion_result = fusion_system.fuse(turn_result, context)

                # Update context
                context['last_intent'] = turn_result['intent']
                context['conversation_turn'] = len(context.get('history', [])) + 1

                # Validate turn
                self.assertIsNotNone(turn_result['response'], "Should generate response")
                self.assertGreaterEqual(fusion_result['confidence'], 0.7, "Should have good confidence")
                self.assertIn('next_action', fusion_result, "Should determine next action")

            # Verify method calls
            self.assertEqual(mock_conv.return_value.process_turn.call_count, len(conversation_history))
            self.assertEqual(mock_fusion.return_value.fuse.call_count, len(conversation_history))

    def test_error_recovery_integration(self):
        """Test error recovery in integrated system"""
        from ai_robot_system import ErrorRecoverySystem, ActionExecutor

        with patch('ai_robot_system.ErrorRecoverySystem') as mock_recovery, \
             patch('ai_robot_system.ActionExecutor') as mock_action:

            # Configure mocks with some failures
            action_results = [
                {'status': 'success'},  # First action succeeds
                {'status': 'failed', 'error': 'obstacle_detected'},  # Second fails
                {'status': 'success'},  # Recovery action succeeds
                {'status': 'success'}   # Final action succeeds
            ]

            call_count = 0
            def mock_execute_action(action):
                nonlocal call_count
                result = action_results[call_count % len(action_results)]
                call_count += 1
                return result

            mock_action.return_value.execute.side_effect = mock_execute_action
            mock_recovery.return_value.handle_error.return_value = [
                {'action': 'avoid_obstacle', 'params': {}},
                {'action': 'retry_original', 'params': {}}
            ]

            action_executor = mock_action.return_value
            recovery_system = mock_recovery.return_value

            # Test error recovery
            actions_to_execute = [
                {'type': 'navigate', 'destination': (1.0, 1.0, 0.0)},
                {'type': 'grasp', 'object': 'cup'},
                {'type': 'navigate', 'destination': (0.0, 0.0, 0.0)}
            ]

            successful_actions = 0
            for action in actions_to_execute:
                result = action_executor.execute(action)

                if result['status'] == 'failed':
                    # Trigger recovery
                    recovery_actions = recovery_system.handle_error(result)

                    # Execute recovery actions
                    for recovery_action in recovery_actions:
                        recovery_result = action_executor.execute(recovery_action)
                        if recovery_result['status'] == 'success':
                            successful_actions += 1
                else:
                    successful_actions += 1

            # Validate error recovery
            self.assertGreaterEqual(successful_actions, len(actions_to_execute) - 1,
                                  "Should recover from errors successfully")

            # Verify method calls
            self.assertGreaterEqual(mock_action.return_value.execute.call_count, len(actions_to_execute))
            self.assertGreaterEqual(mock_recovery.return_value.handle_error.call_count, 1)

    def test_real_time_integration(self):
        """Test real-time performance of integrated system"""
        from ai_robot_system import RealTimeCoordinator

        with patch('ai_robot_system.RealTimeCoordinator') as mock_coord:
            # Configure mock for real-time behavior
            def mock_process_cycle():
                time.sleep(0.05)  # Simulate 20Hz processing
                return {
                    'cycle_time': 0.05,
                    'components_processed': 4,  # vision, audio, language, action
                    'status': 'ok'
                }

            mock_coord.return_value.process_cycle.side_effect = mock_process_cycle

            coordinator = mock_coord.return_value

            # Test real-time performance over multiple cycles
            start_time = time.time()
            cycle_count = 100  # Test for 5 seconds at 20Hz

            processing_times = []
            for i in range(cycle_count):
                cycle_start = time.time()
                result = coordinator.process_cycle()
                cycle_time = time.time() - cycle_start
                processing_times.append(cycle_time)

                self.assertEqual(result['status'], 'ok', f"Cycle {i} should succeed")
                self.assertLessEqual(cycle_time, 0.1, f"Cycle {i} should meet real-time constraints")

            total_time = time.time() - start_time
            avg_cycle_time = sum(processing_times) / len(processing_times)
            achieved_rate = cycle_count / total_time

            # Validate real-time performance
            self.assertLess(avg_cycle_time, 0.06, f"Should maintain <60ms average cycle time, got {avg_cycle_time*1000:.1f}ms")
            self.assertGreaterEqual(achieved_rate, 15, f"Should maintain >15Hz rate, got {achieved_rate:.2f}Hz")

            print(f"Real-time test: {achieved_rate:.2f}Hz, avg {avg_cycle_time*1000:.1f}ms/cycle")

    def test_multi_robot_coordination(self):
        """Test integration with multiple robots (simulated)"""
        from ai_robot_system import MultiRobotCoordinator

        with patch('ai_robot_system.MultiRobotCoordinator') as mock_multi:
            # Configure mock for multi-robot coordination
            mock_multi.return_value.coordinate_robots.return_value = {
                'task_distribution': {'robot1': ['navigate'], 'robot2': ['manipulate']},
                'synchronization_status': 'synchronized',
                'collision_avoidance': 'active'
            }

            multi_coordinator = mock_multi.return_value

            # Simulate multi-robot scenario
            robots = ['robot1', 'robot2', 'robot3']
            task = {'type': 'collaborative_fetch', 'object': 'large_table', 'destination': 'conference_room'}

            coordination_result = multi_coordinator.coordinate_robots(robots, task)

            # Validate multi-robot coordination
            self.assertEqual(coordination_result['synchronization_status'], 'synchronized')
            self.assertEqual(coordination_result['collision_avoidance'], 'active')
            self.assertIn('task_distribution', coordination_result)

            # Verify task was distributed
            distributed_robots = list(coordination_result['task_distribution'].keys())
            self.assertEqual(set(distributed_robots), set(robots[:2]), "Should distribute to subset of robots")

    def test_long_term_memory_integration(self):
        """Test integration with long-term memory system"""
        from ai_robot_system import LongTermMemorySystem, ContextManager

        with patch('ai_robot_system.LongTermMemorySystem') as mock_memory, \
             patch('ai_robot_system.ContextManager') as mock_context:

            # Configure memory system mock
            mock_memory.return_value.store_interaction.return_value = {'success': True, 'id': 'int_001'}
            mock_memory.return_value.retrieve_context.return_value = {
                'previous_interactions': 5,
                'user_preferences': {'name': 'John', 'preferences': ['morning_greets', 'coffee_assistance']},
                'learned_patterns': ['greeting_time', 'object_location_favorites']
            }
            mock_context.return_value.enrich_with_memory.return_value = {
                'personalized_response': 'Good morning John! Would you like your usual coffee?',
                'confidence_boost': 0.15
            }

            memory_system = mock_memory.return_value
            context_manager = mock_context.return_value

            # Test memory-augmented interaction
            user_input = "Good morning"
            interaction_id = 'int_001'

            # Store interaction
            store_result = memory_system.store_interaction(interaction_id, user_input, {})
            self.assertTrue(store_result['success'])

            # Retrieve context from memory
            memory_context = memory_system.retrieve_context('John')

            # Enrich current context with memory
            enriched_context = context_manager.enrich_with_memory(user_input, memory_context)

            # Validate memory integration
            self.assertIn('personalized_response', enriched_context)
            self.assertGreaterEqual(enriched_context.get('confidence_boost', 0), 0)
            self.assertIn('John', enriched_context['personalized_response'])

            # Verify method calls
            mock_memory.return_value.store_interaction.assert_called_once()
            mock_memory.return_value.retrieve_context.assert_called_once_with('John')
            mock_context.return_value.enrich_with_memory.assert_called_once()

    def test_system_reliability_under_failure(self):
        """Test system reliability when components fail"""
        from ai_robot_system import FallbackManager

        with patch('ai_robot_system.FallbackManager') as mock_fallback:
            # Simulate component failures and fallbacks
            primary_methods = ['vision_detect', 'audio_recognize', 'language_understand']
            fallback_methods = ['vision_alternate', 'audio_alternate', 'language_alternate']

            def mock_execute_with_fallback(primary, fallback):
                # Simulate 30% failure rate for primary methods
                if np.random.random() < 0.3:
                    # Primary failed, use fallback
                    return {'method_used': fallback, 'status': 'success', 'quality': 0.7}
                else:
                    # Primary succeeded
                    return {'method_used': primary, 'status': 'success', 'quality': 0.9}

            mock_fallback.return_value.execute_with_fallback.side_effect = mock_execute_with_fallback

            fallback_manager = mock_fallback.return_value

            # Test reliability over many operations
            operations = 100
            successful_ops = 0
            high_quality_ops = 0

            for i in range(operations):
                primary = primary_methods[i % len(primary_methods)]
                fallback = fallback_methods[i % len(fallback_methods)]

                result = fallback_manager.execute_with_fallback(primary, fallback)

                if result['status'] == 'success':
                    successful_ops += 1
                    if result['quality'] >= 0.7:
                        high_quality_ops += 1

            success_rate = successful_ops / operations
            high_quality_rate = high_quality_ops / operations

            # Validate reliability
            self.assertGreaterEqual(success_rate, 0.95, f"Should maintain >95% success rate with fallbacks, got {success_rate:.2%}")
            self.assertGreaterEqual(high_quality_rate, 0.70, f"Should maintain >70% high-quality rate, got {high_quality_rate:.2%}")

            print(f"Reliability test: {success_rate:.2%} success, {high_quality_rate:.2%} high quality")


def create_end_to_end_suite():
    """Create a test suite for end-to-end integration tests"""
    suite = unittest.TestSuite()

    # Add all test methods
    loader = unittest.TestLoader()
    suite.addTests(loader.loadTestsFromTestCase(EndToEndIntegrationTests))

    return suite


def run_end_to_end_tests():
    """Run end-to-end integration tests"""
    suite = create_end_to_end_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    run_end_to_end_tests()
```

## Performance and Stress Testing

### Testing System Performance Under Load

```python
#!/usr/bin/env python3

import unittest
import time
import threading
import queue
import psutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any


class PerformanceAndStressTests(unittest.TestCase):
    """Performance and stress tests for AI robot brain system"""

    def setUp(self):
        """Set up performance testing environment"""
        self.baseline_performance = {
            'vision_fps': 15.0,
            'language_latency': 0.1,  # seconds
            'action_execution_time': 1.0,  # seconds
            'memory_usage_mb': 500.0,
            'cpu_usage_percent': 30.0
        }

        self.stress_test_params = {
            'max_concurrent_requests': 50,
            'test_duration_seconds': 30,
            'ramp_up_time': 5,  # seconds to reach max load
            'acceptable_degradation': 0.2  # 20% performance degradation allowed
        }

    def test_vision_pipeline_performance(self):
        """Test performance of vision pipeline under various loads"""
        import cv2
        np.random.seed(42)  # For reproducible results

        # Generate test images
        test_images = []
        for _ in range(100):
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            test_images.append(img)

        # Simulate vision processing
        processing_times = []
        for img in test_images:
            start_time = time.perf_counter()

            # Simulate vision processing (object detection, feature extraction, etc.)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Simulate object detection
            objects = []
            for contour in contours[:10]:  # Limit to first 10 contours
                area = cv2.contourArea(contour)
                if area > 100:  # Filter small contours
                    objects.append({'area': area, 'contour': contour})

            processing_time = time.perf_counter() - start_time
            processing_times.append(processing_time)

        avg_processing_time = np.mean(processing_times)
        std_processing_time = np.std(processing_times)
        fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0

        # Validate performance
        self.assertLess(avg_processing_time, 0.1, f"Vision processing should be <100ms, got {avg_processing_time:.3f}s")
        self.assertGreaterEqual(fps, 10, f"Vision pipeline should achieve >10 FPS, got {fps:.2f}")

        print(f"Vision performance: {fps:.2f} FPS, avg {avg_processing_time*1000:.1f}ms, std {std_processing_time*1000:.1f}ms")

    def test_language_processing_performance(self):
        """Test performance of language processing pipeline"""
        test_sentences = [
            "Go to the kitchen and get me a cup of water",
            "Please navigate to the person sitting on the chair",
            "What time is it and how many people are in the room",
            "Hello there, how are you doing today?",
            "Can you bring me the red book from the table?"
        ] * 20  # Repeat to get more samples

        processing_times = []

        for sentence in test_sentences:
            start_time = time.perf_counter()

            # Simulate language processing (tokenization, parsing, intent recognition)
            tokens = sentence.lower().split()
            intent_keywords = {
                'navigation': ['go', 'navigate', 'move', 'walk'],
                'manipulation': ['get', 'bring', 'fetch', 'pick'],
                'information': ['what', 'how', 'when', 'where'],
                'social': ['hello', 'hi', 'goodbye', 'thank']
            }

            # Simple intent recognition
            intent = 'unknown'
            for intent_type, keywords in intent_keywords.items():
                if any(keyword in tokens for keyword in keywords):
                    intent = intent_type
                    break

            # Entity extraction
            entities = []
            for token in tokens:
                if token in ['kitchen', 'living room', 'bedroom', 'cup', 'book', 'person']:
                    entities.append(token)

            processing_time = time.perf_counter() - start_time
            processing_times.append(processing_time)

        avg_processing_time = np.mean(processing_times)
        p95_processing_time = np.percentile(processing_times, 95)
        throughput = len(test_sentences) / sum(processing_times)

        # Validate performance
        self.assertLess(avg_processing_time, 0.05, f"Language processing should be <50ms, got {avg_processing_time:.3f}s")
        self.assertLess(p95_processing_time, 0.1, f"95th percentile should be <100ms, got {p95_processing_time:.3f}s")
        self.assertGreaterEqual(throughput, 15, f"Should handle >15 sentences/sec, got {throughput:.2f}")

        print(f"Language performance: {throughput:.2f} sent/sec, avg {avg_processing_time*1000:.1f}ms, p95 {p95_processing_time*1000:.1f}ms")

    def test_concurrent_vision_processing(self):
        """Test concurrent vision processing performance"""
        def process_image_task(img_data):
            """Task to process a single image"""
            start_time = time.perf_counter()

            # Simulate processing
            gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

            # Count contours as processing result
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            result = len(contours)

            processing_time = time.perf_counter() - start_time
            return processing_time, result

        # Generate test images
        test_images = []
        for _ in range(50):
            img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)  # Smaller images for concurrency test
            test_images.append(img)

        # Test with different thread counts
        thread_counts = [1, 2, 4, 8]

        for thread_count in thread_counts:
            start_time = time.time()

            with ThreadPoolExecutor(max_workers=thread_count) as executor:
                futures = [executor.submit(process_image_task, img) for img in test_images]
                results = [future.result() for future in as_completed(futures)]

            total_time = time.time() - start_time
            processing_times = [result[0] for result in results]
            avg_time = np.mean(processing_times)
            throughput = len(test_images) / total_time

            print(f"Vision concurrency (threads={thread_count}): {throughput:.2f} img/sec, avg {avg_time*1000:.1f}ms per image")

            # Validate that throughput scales reasonably
            if thread_count == 1:
                baseline_throughput = throughput
            else:
                efficiency = throughput / (baseline_throughput * thread_count)
                self.assertGreaterEqual(efficiency, 0.5, f"Should maintain >50% efficiency with {thread_count} threads, got {efficiency:.2%}")

    def test_memory_usage_under_load(self):
        """Test memory usage under sustained load"""
        import gc

        # Monitor initial memory
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Simulate sustained processing
        start_time = time.time()
        processed_items = 0

        # Create some data structures that would be used in real processing
        object_detections = []
        language_contexts = []
        action_plans = []

        try:
            while time.time() - start_time < 10:  # 10 seconds of sustained load
                # Simulate processing pipeline
                for i in range(10):  # Process 10 items per iteration
                    # Simulate object detection
                    detection = {
                        'id': f'obj_{processed_items}_{i}',
                        'class': np.random.choice(['person', 'chair', 'table', 'cup']),
                        'bbox': np.random.rand(4).tolist(),
                        'confidence': np.random.uniform(0.5, 1.0)
                    }
                    object_detections.append(detection)

                    # Simulate language context
                    context = {
                        'timestamp': time.time(),
                        'entities': [f'entity_{processed_items}_{i}'],
                        'intent': np.random.choice(['navigate', 'grasp', 'greet'])
                    }
                    language_contexts.append(context)

                    # Simulate action plan
                    plan = {
                        'action_id': f'action_{processed_items}_{i}',
                        'type': np.random.choice(['navigation', 'manipulation', 'social']),
                        'params': {'x': np.random.rand(), 'y': np.random.rand()}
                    }
                    action_plans.append(plan)

                processed_items += 10

                # Periodically clean up some data to simulate garbage collection
                if processed_items % 100 == 0:
                    object_detections = object_detections[-50:]  # Keep last 50
                    language_contexts = language_contexts[-50:]
                    action_plans = action_plans[-50:]

        finally:
            # Force garbage collection
            gc.collect()

        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(f"Memory usage: started at {initial_memory:.1f}MB, ended at {final_memory:.1f}MB, increase {memory_increase:.1f}MB")

        # Validate memory usage
        self.assertLess(memory_increase, 200, f"Memory increase should be <200MB, got {memory_increase:.1f}MB")
        self.assertGreater(processed_items, 0, "Should have processed some items")

    def test_system_stress_test(self):
        """Comprehensive stress test of the entire system"""
        # This simulates a realistic stress scenario
        # where multiple components are under load simultaneously

        stop_event = threading.Event()
        results_queue = queue.Queue()

        def vision_worker():
            """Worker for vision processing"""
            processed = 0
            errors = 0

            while not stop_event.is_set():
                try:
                    # Simulate vision processing
                    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 50, 150)

                    processed += 1
                    time.sleep(0.02)  # Simulate processing time

                except Exception as e:
                    errors += 1

            results_queue.put(('vision', processed, errors))

        def language_worker():
            """Worker for language processing"""
            processed = 0
            errors = 0

            while not stop_event.is_set():
                try:
                    # Simulate language processing
                    sentence = np.random.choice([
                        "go to kitchen", "get the cup", "hello there", "what time is it"
                    ])

                    # Simple processing
                    tokens = sentence.split()
                    intent = 'unknown'
                    if any(w in sentence for w in ['go', 'navigate']):
                        intent = 'navigation'
                    elif any(w in sentence for w in ['get', 'bring']):
                        intent = 'manipulation'

                    processed += 1
                    time.sleep(0.01)  # Simulate processing time

                except Exception as e:
                    errors += 1

            results_queue.put(('language', processed, errors))

        def action_worker():
            """Worker for action execution simulation"""
            processed = 0
            errors = 0

            while not stop_event.is_set():
                try:
                    # Simulate action execution
                    action_type = np.random.choice(['navigate', 'grasp', 'greet'])
                    params = {'x': np.random.rand(), 'y': np.random.rand()}

                    # Simulate execution time
                    time.sleep(0.05)

                    processed += 1

                except Exception as e:
                    errors += 1

            results_queue.put(('action', processed, errors))

        # Start workers
        vision_thread = threading.Thread(target=vision_worker)
        language_thread = threading.Thread(target=language_worker)
        action_thread = threading.Thread(target=action_worker)

        vision_thread.start()
        language_thread.start()
        action_thread.start()

        # Let them run for 15 seconds
        time.sleep(15)

        # Stop workers
        stop_event.set()

        # Wait for threads to finish
        vision_thread.join(timeout=2)
        language_thread.join(timeout=2)
        action_thread.join(timeout=2)

        # Collect results
        results = {}
        while not results_queue.empty():
            component, processed, errors = results_queue.get()
            results[component] = {'processed': processed, 'errors': errors}

        # Calculate rates
        test_duration = 15.0
        for component, data in results.items():
            rate = data['processed'] / test_duration
            error_rate = (data['errors'] / (data['processed'] + data['errors'])) if (data['processed'] + data['errors']) > 0 else 0

            print(f"Stress test - {component}: {rate:.2f} ops/sec, {error_rate:.2%} error rate")

            # Validate performance under stress
            self.assertGreaterEqual(rate, 10, f"{component} should maintain >10 ops/sec under stress, got {rate:.2f}")
            self.assertLessEqual(error_rate, 0.05, f"{component} error rate should be <5%, got {error_rate:.2%}")

    def test_resource_contention(self):
        """Test how the system handles resource contention"""
        # Simulate CPU-intensive background tasks
        background_tasks = []

        def cpu_intensive_task():
            """CPU-intensive task to create contention"""
            count = 0
            start_time = time.time()
            while time.time() - start_time < 5:  # Run for 5 seconds
                # Perform CPU-intensive operation
                arr = np.random.rand(100, 100)
                result = np.linalg.det(arr)  # CPU-intensive operation
                count += 1
            return count

        # Start several CPU-intensive background tasks
        with ThreadPoolExecutor(max_workers=4) as bg_executor:
            bg_futures = [bg_executor.submit(cpu_intensive_task) for _ in range(4)]

            # Meanwhile, run the main system tasks
            main_start = time.time()
            main_processed = 0

            while time.time() - main_start < 5:
                # Simulate main system processing
                img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _ = cv2.countNonZero(gray)  # Simple operation
                main_processed += 1
                time.sleep(0.01)  # Simulate real processing time

            # Wait for background tasks
            bg_results = [future.result() for future in bg_futures]

        main_rate = main_processed / 5.0  # ops per second
        bg_total_ops = sum(bg_results)

        print(f"Resource contention test: Main system {main_rate:.2f} ops/sec, Background {bg_total_ops} total ops")

        # Even under contention, main system should maintain reasonable performance
        self.assertGreaterEqual(main_rate, 5, f"Main system should maintain >5 ops/sec under CPU contention, got {main_rate:.2f}")

    def test_scalability_analysis(self):
        """Analyze how performance scales with increasing load"""
        load_levels = [1, 5, 10, 20, 30]  # Different load levels
        performance_results = {}

        for load_level in load_levels:
            start_time = time.time()

            # Simulate load level by running concurrent tasks
            with ThreadPoolExecutor(max_workers=load_level) as executor:
                futures = []

                for i in range(load_level * 10):  # Each thread processes 10 items
                    future = executor.submit(self._simulate_processing_task)
                    futures.append(future)

                # Wait for all tasks
                for future in as_completed(futures):
                    future.result()  # Wait for completion

            total_time = time.time() - start_time
            total_processed = load_level * 10
            throughput = total_processed / total_time
            per_task_time = total_time / total_processed * 1000  # ms per task

            performance_results[load_level] = {
                'throughput': throughput,
                'per_task_time': per_task_time,
                'total_time': total_time
            }

            print(f"Load {load_level}: {throughput:.2f} ops/sec, {per_task_time:.1f}ms per task")

        # Analyze scalability
        # Performance should not degrade linearly (Amdahl's law), but should scale reasonably
        if len(load_levels) > 1:
            initial_throughput = performance_results[load_levels[0]]['throughput']
            final_throughput = performance_results[load_levels[-1]]['throughput']
            efficiency = final_throughput / (initial_throughput * load_levels[-1])

            print(f"Scalability efficiency: {efficiency:.2%}")
            # Acceptable efficiency depends on the system, but >30% is reasonable for complex systems
            self.assertGreaterEqual(efficiency, 0.3, f"System should maintain >30% efficiency when scaling, got {efficiency:.2%}")

    def _simulate_processing_task(self):
        """Helper method to simulate a processing task"""
        # Simulate some processing work
        time.sleep(0.01)  # 10ms of simulated work
        arr = np.random.rand(50, 50)
        _ = np.sum(arr * arr)  # Simple computation
        return "completed"


def create_performance_suite():
    """Create a test suite for performance tests"""
    suite = unittest.TestSuite()

    # Add all test methods
    loader = unittest.TestLoader()
    suite.addTests(loader.loadTestsFromTestCase(PerformanceAndStressTests))

    return suite


def run_performance_tests():
    """Run performance and stress tests"""
    suite = create_performance_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    run_performance_tests()
```

## Continuous Integration and Testing

### Setting up CI/CD for Integration Tests

```python
#!/usr/bin/env python3

import os
import sys
import subprocess
import unittest
from datetime import datetime
import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Any


class ContinuousIntegrationTests(unittest.TestCase):
    """Tests for continuous integration setup and execution"""

    def setUp(self):
        """Set up CI environment"""
        self.test_results_dir = 'test-results'
        self.coverage_threshold = 80.0  # 80% coverage required
        self.performance_thresholds = {
            'vision_fps': 10.0,
            'language_latency_ms': 100.0,
            'action_success_rate': 0.95
        }

    def test_code_coverage(self):
        """Test that code coverage meets requirements"""
        try:
            # Run coverage analysis
            result = subprocess.run([
                sys.executable, '-m', 'coverage', 'run',
                '-m', 'unittest', 'discover', '-s', 'tests/', '-p', '*test*.py'
            ], capture_output=True, text=True)

            if result.returncode == 0:
                # Get coverage report
                coverage_result = subprocess.run([
                    sys.executable, '-m', 'coverage', 'report', '--format=json'
                ], capture_output=True, text=True)

                if coverage_result.returncode == 0:
                    coverage_data = json.loads(coverage_report.stdout)
                    overall_coverage = coverage_data.get('overall_coverage', 0)

                    self.assertGreaterEqual(
                        overall_coverage,
                        self.coverage_threshold,
                        f"Code coverage {overall_coverage}% is below required {self.coverage_threshold}%"
                    )

                    print(f"Code coverage: {overall_coverage}%")
                else:
                    self.fail(f"Coverage report failed: {coverage_result.stderr}")
            else:
                self.fail(f"Coverage run failed: {result.stderr}")

        except FileNotFoundError:
            print("Coverage tool not available, skipping coverage test")
            self.skipTest("Coverage tool not available")

    def test_static_analysis(self):
        """Test static analysis tools"""
        # Check code quality with pylint
        try:
            result = subprocess.run([
                'pylint', 'ai_robot_system/', '--output-format=json'
            ], capture_output=True, text=True)

            if result.returncode == 0:
                # Parse pylint output
                try:
                    pylint_data = json.loads(result.stdout)
                    # Check for critical errors
                    critical_errors = [msg for msg in pylint_data if msg.get('type') == 'error']

                    self.assertEqual(
                        len(critical_errors), 0,
                        f"Found {len(critical_errors)} critical errors in code quality"
                    )

                    print(f"Static analysis: {len(pylint_data)} issues found, {len(critical_errors)} critical")

                except json.JSONDecodeError:
                    # Pylint might output in different format
                    print("Pylint output format not JSON, checking for errors in text")
                    if "error" in result.stdout.lower():
                        self.fail("Static analysis found errors")

            elif result.returncode > 3:  # Only fail on critical errors, not style warnings
                self.fail(f"Static analysis failed: {result.stderr}")

        except FileNotFoundError:
            print("Pylint not available, skipping static analysis")
            self.skipTest("Pylint not available")

    def test_dependency_security(self):
        """Test for security vulnerabilities in dependencies"""
        try:
            result = subprocess.run([
                'pip', 'list', '--outdated', '--format=json'
            ], capture_output=True, text=True)

            if result.returncode == 0:
                outdated_data = json.loads(result.stdout)

                # Check for outdated packages (security risk)
                critical_outdated = []
                for pkg in outdated_data:
                    # In a real system, you'd check against known vulnerabilities
                    # For now, just log outdated packages
                    if pkg.get('latest_version') != pkg.get('version'):
                        critical_outdated.append(pkg)

                print(f"Outdated packages: {len(critical_outdated)}")
                for pkg in critical_outdated[:5]:  # Show first 5
                    print(f"  {pkg['name']}: {pkg['version']} -> {pkg['latest_version']}")

            else:
                print(f"Dependency check failed: {result.stderr}")

        except Exception as e:
            print(f"Dependency security check error: {e}")
            self.skipTest("Dependency security check failed")

    def test_performance_regression(self):
        """Test for performance regressions compared to baseline"""
        # This would typically compare against stored baseline results
        # For this example, we'll simulate the test

        # Simulate current performance results
        current_results = {
            'vision_fps': 12.5,
            'language_avg_time_ms': 85.0,
            'action_success_rate': 0.96
        }

        # Check against thresholds
        for metric, current_value in current_results.items():
            threshold = self.performance_thresholds.get(metric)
            if threshold:
                if 'rate' in metric or 'ratio' in metric:
                    # Higher is better for rates/ratios
                    self.assertGreaterEqual(
                        current_value, threshold,
                        f"{metric} {current_value} below threshold {threshold}"
                    )
                else:
                    # Lower is better for times/latencies
                    self.assertLessEqual(
                        current_value, threshold,
                        f"{metric} {current_value} above threshold {threshold}"
                    )

        print(f"Performance regression check passed: {current_results}")

    def test_integration_pipeline(self):
        """Test the complete integration pipeline"""
        # This simulates running the complete test pipeline
        test_results = {
            'unit_tests': {'passed': 0, 'failed': 0, 'total': 0},
            'integration_tests': {'passed': 0, 'failed': 0, 'total': 0},
            'performance_tests': {'passed': 0, 'failed': 0, 'total': 0},
            'system_tests': {'passed': 0, 'failed': 0, 'total': 0}
        }

        # Simulate running different test suites
        from vision_integration_tests import run_vision_integration_tests
        from language_integration_tests import run_language_integration_tests
        from action_integration_tests import run_action_integration_tests
        from end_to_end_tests import run_end_to_end_tests
        from performance_tests import run_performance_tests

        # Run unit tests (mock implementation)
        unit_result = unittest.TextTestRunner(stream=open(os.devnull, 'w')).run(
            unittest.TestLoader().loadTestsFromName('test_suite')
        )
        test_results['unit_tests']['total'] = unit_result.testsRun
        test_results['unit_tests']['failed'] = len(unit_result.failures) + len(unit_result.errors)
        test_results['unit_tests']['passed'] = test_results['unit_tests']['total'] - test_results['unit_tests']['failed']

        # Run other test suites would go here
        # For simulation, we'll use mock results
        test_results['integration_tests'] = {'passed': 45, 'failed': 2, 'total': 47}
        test_results['performance_tests'] = {'passed': 12, 'failed': 1, 'total': 13}
        test_results['system_tests'] = {'passed': 8, 'failed': 0, 'total': 8}

        # Validate pipeline results
        total_tests = sum(result['total'] for result in test_results.values())
        total_failed = sum(result['failed'] for result in test_results.values())
        success_rate = (total_tests - total_failed) / total_tests if total_tests > 0 else 0

        self.assertGreaterEqual(success_rate, 0.95, f"Overall test success rate {success_rate:.2%} is below 95%")

        print(f"Integration pipeline: {success_rate:.2%} success rate ({total_failed}/{total_tests} failed)")

    def test_cross_platform_compatibility(self):
        """Test compatibility across different platforms"""
        import platform

        current_platform = platform.system().lower()
        supported_platforms = ['linux', 'darwin', 'windows']  # Linux, macOS, Windows

        self.assertIn(
            current_platform, supported_platforms,
            f"Platform {current_platform} not supported. Supported: {supported_platforms}"
        )

        # Test platform-specific functionality
        if current_platform == 'windows':
            # Windows-specific tests
            import ctypes
            is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
            print(f"Windows admin privileges: {is_admin}")
        elif current_platform in ['linux', 'darwin']:
            # Unix-specific tests
            import pwd
            current_user = pwd.getpwuid(os.getuid()).pw_name
            print(f"Current user: {current_user}")

        print(f"Platform compatibility test passed for {current_platform}")

    def test_documentation_generation(self):
        """Test that documentation can be generated"""
        # Check if documentation generation tools are available
        try:
            # Try to import and test documentation tools
            import sphinx
            print(f"Sphinx version: {sphinx.__version__}")
        except ImportError:
            print("Sphinx not available, skipping documentation test")
            self.skipTest("Sphinx not available")

        # In a real system, you would test actual documentation generation
        # For now, just verify the capability exists
        docs_dir = 'docs/'
        if os.path.exists(docs_dir):
            doc_files = [f for f in os.listdir(docs_dir) if f.endswith(('.md', '.rst', '.txt'))]
            self.assertGreaterEqual(len(doc_files), 1, "Should have documentation files")
            print(f"Documentation files found: {len(doc_files)}")


def create_ci_suite():
    """Create a test suite for CI tests"""
    suite = unittest.TestSuite()

    # Add all test methods
    loader = unittest.TestLoader()
    suite.addTests(loader.loadTestsFromTestCase(ContinuousIntegrationTests))

    return suite


def run_ci_tests():
    """Run CI tests and generate reports"""
    # Create results directory
    results_dir = 'test-results'
    os.makedirs(results_dir, exist_ok=True)

    # Create test suite
    suite = create_ci_suite()

    # Run tests and capture results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(results_dir, f'ci_results_{timestamp}.xml')

    # Use XMLTestRunner if available for CI/CD integration
    try:
        from xmlrunner import XMLTestRunner
        runner = XMLTestRunner(output=results_dir, outsuffix=f'_{timestamp}')
        result = runner.run(suite)
    except ImportError:
        # Fallback to regular runner
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

    # Generate summary
    print(f"\n=== CI Test Summary ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")

    # Save results summary
    summary = {
        'timestamp': timestamp,
        'tests_run': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun,
        'results_file': results_file
    }

    summary_file = os.path.join(results_dir, f'summary_{timestamp}.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to: {summary_file}")

    return result


if __name__ == '__main__':
    run_ci_tests()
```

## Key Takeaways and Best Practices

### Integration Testing Best Practices

Based on the comprehensive testing framework developed above, here are the key best practices for AI robot brain integration testing:

1. **Layered Testing Approach**: Test at multiple levels - unit, component, integration, and system
2. **Mock-Based Testing**: Use mocks extensively to isolate components during integration testing
3. **Performance Validation**: Always test performance under realistic loads
4. **Real-time Constraints**: Validate that integrated systems meet real-time requirements
5. **Error Recovery**: Test error handling and recovery mechanisms
6. **Cross-Component Dependencies**: Validate data flow between components
7. **Safety Integration**: Ensure safety systems work with all action components
8. **Continuous Testing**: Integrate testing into CI/CD pipelines

### Validation Strategies

1. **Functional Validation**: Ensure the integrated system performs intended functions
2. **Performance Validation**: Verify system meets speed and throughput requirements
3. **Robustness Validation**: Test system behavior under stress and edge cases
4. **Safety Validation**: Confirm safety mechanisms function correctly
5. **Reliability Validation**: Test system reliability over extended periods

These integration tests provide comprehensive coverage for validating perception pipeline functionality and multimodal integration in AI robot brain systems.