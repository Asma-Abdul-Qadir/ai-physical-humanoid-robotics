---
sidebar_position: 3
---

# Perception Pipeline: Vision-Language-Action Integration

Welcome to the Perception Pipeline module, which focuses on creating robust perception systems that integrate computer vision, audio processing, and multimodal understanding for humanoid robots. This chapter covers the implementation of perception systems that can process visual and auditory inputs to understand the environment and support decision-making.

## Learning Objectives

By the end of this section, you will be able to:
- Design and implement multimodal perception pipelines for humanoid robots
- Integrate computer vision and audio processing systems
- Create fusion mechanisms for combining different sensory inputs
- Implement real-time perception for dynamic environments
- Evaluate perception system performance and robustness
- Handle edge cases and failure scenarios in perception
- Troubleshoot common perception system issues

## Introduction to Perception Systems

Perception systems form the sensory foundation of humanoid robots, enabling them to understand and interact with their environment. For humanoid robots, perception must handle complex, dynamic environments with multiple objects, people, and changing conditions.

### Key Components of Perception Systems

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Visual        │    │   Audio         │    │   Multimodal    │
│   Perception    │───▶│   Processing    │───▶│   Fusion        │
│   • Cameras     │    │   • Microphones │    │   • Integration │
│   • LIDAR       │    │   • Speech      │    │   • Context     │
│   • Depth       │    │   • Noise       │    │   • Attention   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Perception Pipeline                          │
│            (Real-time Processing & Understanding)               │
└─────────────────────────────────────────────────────────────────┘
```

### Perception Architecture for Humanoid Robots

Humanoid robots require specialized perception systems that:
- Handle 3D environments with complex spatial relationships
- Support social interaction with humans
- Provide real-time processing for responsive behavior
- Integrate multiple sensory modalities
- Adapt to changing environmental conditions

## Visual Perception Pipeline

### Camera Systems and Image Processing

Humanoid robots typically employ multiple cameras for comprehensive visual perception:

```python
#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header


class VisualPerceptionNode(Node):
    def __init__(self):
        super().__init__('visual_perception')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Camera subscriptions for different viewpoints
        self.front_camera_sub = self.create_subscription(
            Image, '/camera/front/image_raw', self.front_camera_callback, 10)
        self.head_camera_sub = self.create_subscription(
            Image, '/camera/head/image_raw', self.head_camera_callback, 10)
        self.depth_camera_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_camera_callback, 10)

        # Publishers for processed data
        self.object_detection_pub = self.create_publisher(
            MarkerArray, '/detected_objects', 10)
        self.feature_map_pub = self.create_publisher(
            Image, '/feature_maps', 10)
        self.scene_description_pub = self.create_publisher(
            MarkerArray, '/scene_description', 10)

        # TF buffer for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Vision processing parameters
        self.detection_threshold = 0.5
        self.tracking_threshold = 0.3
        self.class_names = [
            'person', 'chair', 'table', 'door', 'obstacle',
            'robot', 'cup', 'phone', 'book', 'plant', 'laptop'
        ]

        # Processing pipeline components
        self.object_detector = self.initialize_object_detector()
        self.feature_extractor = self.initialize_feature_extractor()
        self.scene_analyzer = SceneAnalyzer()
        self.object_tracker = ObjectTracker()

        # State variables
        self.tracked_objects = {}
        self.scene_context = {}

        self.get_logger().info('Visual perception pipeline initialized')

    def initialize_object_detector(self):
        """Initialize object detection model"""
        # In a real implementation, this would load a YOLO, SSD, or similar model
        # For this example, we'll simulate detection
        return ObjectDetector()

    def initialize_feature_extractor(self):
        """Initialize feature extraction model"""
        # This could be a CNN, ViT, or other feature extractor
        return FeatureExtractor()

    def front_camera_callback(self, msg):
        """Process front-facing camera feed for navigation and obstacle detection"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Run object detection
            detections = self.object_detector.detect(cv_image, self.detection_threshold)

            # Update object tracking
            tracked_objects = self.object_tracker.update(detections, msg.header.stamp)

            # Extract features for deeper analysis
            features = self.feature_extractor.extract(cv_image)

            # Analyze scene context
            scene_description = self.scene_analyzer.analyze(cv_image, tracked_objects)

            # Publish results
            self.publish_detections(tracked_objects, msg.header)
            self.publish_features(features, msg)
            self.publish_scene_description(scene_description, msg.header)

            # Update internal state
            self.tracked_objects = tracked_objects
            self.scene_context = scene_description

            # Log scene description
            self.get_logger().info(f'Scene: {scene_description["summary"]}')

        except Exception as e:
            self.get_logger().error(f'Front camera processing error: {e}')

    def head_camera_callback(self, msg):
        """Process head-mounted camera feed for social interaction"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Face detection for social interaction
            faces = self.detect_faces(cv_image)

            # Person identification and tracking
            identities = self.identify_people(faces, cv_image, msg.header.stamp)

            # Update interaction context
            self.update_interaction_context(identities, msg.header)

        except Exception as e:
            self.get_logger().error(f'Head camera processing error: {e}')

    def depth_camera_callback(self, msg):
        """Process depth camera feed for 3D understanding"""
        try:
            # Convert depth image to CV format
            depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")

            # Process depth information
            obstacles_3d = self.extract_3d_obstacles(depth_image)

            # Update spatial understanding
            self.update_spatial_map(obstacles_3d, msg.header)

        except Exception as e:
            self.get_logger().error(f'Depth camera processing error: {e}')

    def detect_faces(self, image):
        """Detect faces in image for social interaction"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Load face cascade classifier
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        return faces

    def identify_people(self, faces, image, timestamp):
        """Identify and track people in the scene"""
        identities = []

        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = image[y:y+h, x:x+w]

            # In a real system, this would use face recognition
            # For simulation, assign temporary IDs based on position
            person_id = f'person_{x}_{y}_{int(timestamp.nanoseconds / 1e9)}'

            identity = {
                'id': person_id,
                'bbox': (x, y, w, h),
                'confidence': 0.9,
                'timestamp': timestamp,
                'position': (x, y)  # 2D position in image
            }
            identities.append(identity)

        return identities

    def extract_3d_obstacles(self, depth_image):
        """Extract 3D obstacles from depth image"""
        # Convert depth to point cloud or extract obstacle regions
        # This is a simplified approach - in practice, use proper depth processing

        # Threshold to identify obstacles (objects closer than threshold)
        obstacle_threshold = 2.0  # meters
        obstacle_mask = depth_image < obstacle_threshold
        obstacle_regions = cv2.connectedComponents(obstacle_mask.astype(np.uint8))

        obstacles_3d = []
        for i in range(1, obstacle_regions[0]):  # Skip background (0)
            region_mask = (obstacle_regions[1] == i)
            if np.sum(region_mask) > 100:  # Minimum size filter
                # Calculate 3D position (simplified)
                y_coords, x_coords = np.where(region_mask)
                avg_depth = np.mean(depth_image[region_mask])

                obstacle_3d = {
                    'region_mask': region_mask,
                    'avg_depth': avg_depth,
                    'center': (np.mean(x_coords), np.mean(y_coords)),
                    'size': np.sum(region_mask)
                }
                obstacles_3d.append(obstacle_3d)

        return obstacles_3d

    def update_spatial_map(self, obstacles_3d, header):
        """Update spatial understanding based on 3D obstacles"""
        # In a real system, this would update a 3D occupancy grid or similar map
        # For simulation, just log the information
        self.get_logger().info(f'Detected {len(obstacles_3d)} 3D obstacles')

    def publish_detections(self, tracked_objects, header):
        """Publish detected objects as visualization markers"""
        marker_array = MarkerArray()

        for obj_id, obj_data in tracked_objects.items():
            marker = Marker()
            marker.header = header
            marker.ns = "objects"
            marker.id = hash(obj_id) % 1000  # Convert string to int
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            # Position in camera frame (would need to transform to world)
            marker.pose.position.x = obj_data['center'][0] / 100.0  # Scale down for visualization
            marker.pose.position.y = obj_data['center'][1] / 100.0
            marker.pose.position.z = 0.0  # Simplified

            marker.pose.orientation.w = 1.0
            marker.scale.x = obj_data['bbox'][2] / 100.0  # Scale down
            marker.scale.y = obj_data['bbox'][3] / 100.0
            marker.scale.z = 0.5  # Height of marker

            # Color based on class
            class_colors = {
                'person': (0, 1, 0),      # Green
                'chair': (1, 0, 0),       # Red
                'table': (0, 0, 1),       # Blue
                'door': (1, 1, 0),        # Yellow
                'obstacle': (1, 0, 1),    # Magenta
                'robot': (0, 1, 1)        # Cyan
            }

            color = class_colors.get(obj_data['class'], (0.5, 0.5, 0.5))
            marker.color.r, marker.color.g, marker.color.b = color
            marker.color.a = 0.7

            marker.text = f"{obj_data['class']}: {obj_data['confidence']:.2f}"

            marker_array.markers.append(marker)

        self.object_detection_pub.publish(marker_array)

    def publish_features(self, features, original_msg):
        """Publish feature maps for downstream processing"""
        # Convert features back to image format for visualization
        if features is not None and len(features.shape) >= 2:
            # This is a simplified representation
            feature_image = np.abs(features)  # Take absolute values
            feature_image = (feature_image / np.max(feature_image) * 255).astype(np.uint8)

            # If it's multi-channel, take the first channel or average
            if len(feature_image.shape) == 3:
                feature_image = np.mean(feature_image, axis=2).astype(np.uint8)

            # Publish as grayscale image
            feature_msg = self.bridge.cv2_to_imgmsg(feature_image, "mono8")
            feature_msg.header = original_msg.header

            self.feature_map_pub.publish(feature_msg)

    def publish_scene_description(self, scene_description, header):
        """Publish scene description as markers"""
        marker_array = MarkerArray()

        # Create a text marker for the scene summary
        text_marker = Marker()
        text_marker.header = header
        text_marker.ns = "scene_summary"
        text_marker.id = 0
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD

        text_marker.pose.position.x = 0.0
        text_marker.pose.position.y = 0.0
        text_marker.pose.position.z = 2.0  # Above the robot

        text_marker.pose.orientation.w = 1.0
        text_marker.scale.z = 0.3  # Text size
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 1.0
        text_marker.text = scene_description['summary']

        marker_array.markers.append(text_marker)

        self.scene_description_pub.publish(marker_array)

    def update_interaction_context(self, identities, header):
        """Update interaction context based on detected people"""
        # Update the interaction context with detected people
        interaction_context = {
            'people_count': len(identities),
            'people_positions': [id_data['position'] for id_data in identities],
            'header': header
        }

        # This could trigger social interaction behaviors
        if len(identities) > 0:
            self.get_logger().info(f'Detected {len(identities)} people for interaction')


class ObjectDetector:
    def __init__(self):
        # In a real implementation, load a pre-trained model
        # e.g., YOLOv8, SSD MobileNet, etc.
        self.model_loaded = False
        pass

    def detect(self, image, threshold=0.5):
        """Detect objects in image"""
        # Simulate detection results
        h, w = image.shape[:2]

        # Generate some mock detections with realistic patterns
        detections = []

        # Simulate detecting various objects with different probabilities
        object_probs = {
            'person': 0.4,    # People are common in human environments
            'chair': 0.3,     # Furniture is common
            'table': 0.2,     # Tables are common
            'door': 0.15,     # Doors are less frequent in frame
            'obstacle': 0.2,  # Random obstacles
            'robot': 0.05,    # Other robots are rare
            'cup': 0.2,       # Objects on tables
            'phone': 0.15,    # Personal items
            'book': 0.1       # Books/reading materials
        }

        for obj_class, prob in object_probs.items():
            if np.random.random() < prob:
                # Random bounding box
                x = np.random.randint(0, max(1, w - w//4))
                y = np.random.randint(0, max(1, h - h//4))
                width = np.random.randint(w//8, min(w//3, w - x))
                height = np.random.randint(h//8, min(h//3, h - y))

                detection = {
                    'class': obj_class,
                    'bbox': (x, y, width, height),
                    'center': (x + width//2, y + height//2),
                    'confidence': np.random.uniform(threshold, 0.95)
                }

                detections.append(detection)

        return detections


class FeatureExtractor:
    def __init__(self):
        # In a real implementation, this would be a CNN or transformer
        pass

    def extract(self, image):
        """Extract features from image"""
        # Simulate feature extraction
        # In reality, this would run through a neural network
        h, w = image.shape[:2]

        # Return a simplified feature representation
        # Shape would typically be (channels, height, width) or (features,)
        return np.random.rand(h//16, w//16, 64).astype(np.float32)


class SceneAnalyzer:
    def __init__(self):
        self.context = {}
        self.room_types = ['office', 'kitchen', 'living room', 'bedroom', 'hallway']

    def analyze(self, image, detections):
        """Analyze scene and generate description"""
        if not detections:
            return {
                'summary': 'Empty scene detected',
                'room_type': 'unknown',
                'activity': 'none',
                'objects': [],
                'people': 0
            }

        # Count objects by class
        object_counts = {}
        for det in detections:
            cls = det['class']
            object_counts[cls] = object_counts.get(cls, 0) + 1

        # Infer room type based on objects present
        room_inference = self.infer_room_type(object_counts)

        # Infer activity based on objects
        activity = self.infer_activity(object_counts)

        # Generate scene description
        description_parts = []
        for obj_class, count in object_counts.items():
            if count == 1:
                description_parts.append(f"1 {obj_class}")
            else:
                description_parts.append(f"{count} {obj_class}s")

        scene_summary = f"Scene contains: {', '.join(description_parts)}"

        return {
            'summary': scene_summary,
            'room_type': room_inference,
            'activity': activity,
            'objects': object_counts,
            'people': object_counts.get('person', 0)
        }

    def infer_room_type(self, object_counts):
        """Infer room type based on object composition"""
        room_scores = {}

        for room in self.room_types:
            score = 0
            if room == 'kitchen':
                score += object_counts.get('cup', 0) * 2
                score += object_counts.get('table', 0) * 1.5
                score += object_counts.get('chair', 0) * 1
            elif room == 'office':
                score += object_counts.get('laptop', 0) * 3
                score += object_counts.get('chair', 0) * 1.5
                score += object_counts.get('table', 0) * 2
            elif room == 'living room':
                score += object_counts.get('chair', 0) * 2
                score += object_counts.get('table', 0) * 1.5
                score += object_counts.get('book', 0) * 1
            elif room == 'bedroom':
                score += object_counts.get('chair', 0) * 0.5
                score += object_counts.get('table', 0) * 1
            elif room == 'hallway':
                score += object_counts.get('door', 0) * 3
                score += object_counts.get('person', 0) * 1

            room_scores[room] = score

        # Return the room with highest score
        return max(room_scores, key=room_scores.get) if room_scores else 'unknown'

    def infer_activity(self, object_counts):
        """Infer current activity based on objects"""
        if object_counts.get('person', 0) > 0:
            if object_counts.get('laptop', 0) > 0 or object_counts.get('book', 0) > 0:
                return 'working/studying'
            elif object_counts.get('cup', 0) > 0:
                return 'relaxing'
            else:
                return 'socializing'
        else:
            return 'inactive'


class ObjectTracker:
    def __init__(self):
        self.tracked_objects = {}
        self.next_id = 0
        self.max_displacement = 50  # pixels for matching

    def update(self, detections, timestamp):
        """Update object tracking with new detections"""
        new_tracked = {}

        for det in detections:
            matched = False

            # Try to match with existing tracked objects
            for obj_id, obj_data in self.tracked_objects.items():
                # Calculate distance to last known position
                dist = np.sqrt((det['center'][0] - obj_data['center'][0])**2 +
                              (det['center'][1] - obj_data['center'][1])**2)

                # If close enough and same class, update tracking
                if (dist < self.max_displacement and
                    det['class'] == obj_data['class'] and
                    det['confidence'] > obj_data['confidence'] * 0.7):
                    new_tracked[obj_id] = {
                        'class': det['class'],
                        'bbox': det['bbox'],
                        'center': det['center'],
                        'confidence': det['confidence'],
                        'timestamp': timestamp,
                        'velocity': self.calculate_velocity(obj_data, det)
                    }
                    matched = True
                    break

            # If no match found, create new track
            if not matched:
                new_id = f"obj_{self.next_id}"
                self.next_id += 1
                new_tracked[new_id] = {
                    'class': det['class'],
                    'bbox': det['bbox'],
                    'center': det['center'],
                    'confidence': det['confidence'],
                    'timestamp': timestamp,
                    'velocity': (0, 0)
                }

        # Update tracked objects
        self.tracked_objects = new_tracked
        return new_tracked

    def calculate_velocity(self, old_obj, new_det):
        """Calculate velocity based on position change"""
        # This would use timestamp information in a real implementation
        dx = new_det['center'][0] - old_obj['center'][0]
        dy = new_det['center'][1] - old_obj['center'][1]
        return (dx, dy)


def main(args=None):
    rclpy.init(args=args)
    visual_node = VisualPerceptionNode()

    try:
        rclpy.spin(visual_node)
    except KeyboardInterrupt:
        pass
    finally:
        visual_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Advanced Visual Processing

For humanoid robots, advanced visual processing includes:

```python
#!/usr/bin/env python3

import numpy as np
import cv2
from scipy.spatial.distance import cdist
import math


class AdvancedVisualProcessor:
    def __init__(self):
        # Feature matching and tracking
        self.feature_detector = cv2.SIFT_create()  # or ORB, AKAZE, etc.
        self.bf_matcher = cv2.BFMatcher()

        # Background subtraction
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()

        # Optical flow
        self.lk_params = dict(winSize=(15, 15),
                             maxLevel=2,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def detect_keypoints_descriptors(self, image):
        """Detect keypoints and compute descriptors"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_features(self, desc1, desc2):
        """Match features between two images"""
        if desc1 is not None and desc2 is not None:
            matches = self.bf_matcher.knnMatch(desc1, desc2, k=2)
            # Apply Lowe's ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            return good_matches
        return []

    def background_subtraction(self, image):
        """Subtract background to detect moving objects"""
        fg_mask = self.bg_subtractor.apply(image)
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        return fg_mask

    def optical_flow_tracking(self, prev_image, curr_image, prev_points):
        """Track points using optical flow"""
        gray_prev = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)

        if len(prev_points) > 0:
            curr_points, status, err = cv2.calcOpticalFlowPyrLK(
                gray_prev, gray_curr, prev_points, None, **self.lk_params)

            # Filter out bad points
            good_curr_points = []
            good_prev_points = []

            for i, st in enumerate(status):
                if st == 1:
                    good_curr_points.append(curr_points[i])
                    good_prev_points.append(prev_points[i])

            return np.array(good_prev_points), np.array(good_curr_points)

        return np.array([]), np.array([])

    def estimate_motion(self, prev_points, curr_points):
        """Estimate camera/robot motion from point correspondences"""
        if len(prev_points) >= 4 and len(curr_points) >= 4:
            # Estimate homography (for planar scenes) or fundamental matrix
            H, mask = cv2.findHomography(prev_points, curr_points, cv2.RANSAC, 5.0)
            return H
        return None
```

## Audio Processing Pipeline

### Microphone Array and Audio Capture

For humanoid robots, audio processing is crucial for voice interaction and environmental awareness:

```python
#!/usr/bin/env python3

import pyaudio
import numpy as np
import webrtcvad
import threading
import queue
import wave
import collections
import time
from scipy import signal


class AudioProcessingNode:
    def __init__(self):
        # Audio parameters
        self.sample_rate = 16000  # Standard for speech processing
        self.chunk_size = 1024
        self.channels = 1
        self.format = pyaudio.paInt16

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()

        # Initialize WebRTC VAD (Voice Activity Detection)
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(1)  # Aggressiveness mode (0-3)

        # Audio processing parameters
        self.silence_threshold = 0.01
        self.min_speech_duration = 0.3  # seconds
        self.max_buffer_duration = 5.0  # seconds

        # Audio buffers and queues
        self.audio_buffer = collections.deque(maxlen=int(self.max_buffer_duration * self.sample_rate))
        self.speech_queue = queue.Queue()

        # Processing threads
        self.capture_thread = None
        self.processing_thread = None
        self.is_recording = False

        # Audio processing components
        self.noise_reducer = NoiseReducer(self.sample_rate)
        self.speech_enhancer = SpeechEnhancer(self.sample_rate)

        self.get_logger().info('Audio processing pipeline initialized')

    def start_capture(self):
        """Start audio capture from microphone"""
        # Open audio stream
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        # Start capture thread
        self.capture_thread = threading.Thread(target=self.capture_audio)
        self.capture_thread.daemon = True
        self.capture_thread.start()

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_audio_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.get_logger().info('Audio capture started')

    def capture_audio(self):
        """Capture audio in a separate thread"""
        while True:
            try:
                # Read audio data
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32)

                # Normalize audio
                audio_array /= 32768.0

                # Add to buffer
                self.audio_buffer.extend(audio_array)

                # Check for voice activity
                if self.is_voice_activity(audio_array):
                    if not self.is_recording:
                        self.start_recording()
                else:
                    if self.is_recording:
                        self.check_stop_recording()

            except Exception as e:
                self.get_logger().error(f'Audio capture error: {e}')
                break

    def is_voice_activity(self, audio_chunk):
        """Check for voice activity using WebRTC VAD"""
        # Convert to 16-bit PCM for VAD
        chunk_16bit = (audio_chunk * 32768).astype(np.int16)

        # VAD requires 10, 20, or 30 ms frames
        frame_size = int(self.sample_rate * 0.01)  # 10ms frame
        if len(chunk_16bit) >= frame_size:
            frame = chunk_16bit[:frame_size].tobytes()
            return self.vad.is_speech(frame, self.sample_rate)

        return False

    def start_recording(self):
        """Start recording audio for processing"""
        self.is_recording = True
        self.recording_start_time = time.time()
        self.recording_buffer = []

    def check_stop_recording(self):
        """Check if we should stop recording"""
        if (time.time() - self.recording_start_time) >= self.min_speech_duration:
            # Queue the recording for processing
            if hasattr(self, 'recording_buffer') and len(self.recording_buffer) > 0:
                recording_data = np.concatenate(self.recording_buffer)
                self.speech_queue.put(recording_data)
            self.is_recording = False

    def process_audio_queue(self):
        """Process speech data from queue"""
        while True:
            try:
                audio_data = self.speech_queue.get(timeout=1.0)

                # Preprocess audio
                processed_audio = self.preprocess_audio(audio_data)

                # Perform speech enhancement
                enhanced_audio = self.speech_enhancer.enhance(processed_audio)

                # Publish for further processing (e.g., to speech recognition)
                self.publish_speech_data(enhanced_audio)

            except queue.Empty:
                continue  # Timeout, continue loop
            except Exception as e:
                self.get_logger().error(f'Audio queue processing error: {e}')

    def preprocess_audio(self, audio_data):
        """Preprocess audio data"""
        # Apply noise reduction
        denoised_audio = self.noise_reducer.reduce_noise(audio_data)

        # Apply AGC (Automatic Gain Control)
        agc_audio = self.apply_agc(denoised_audio)

        # Apply high-pass filter to remove DC offset
        b, a = signal.butter(2, 100 / (self.sample_rate / 2), 'high')
        filtered_audio = signal.filtfilt(b, a, agc_audio)

        return filtered_audio

    def apply_agc(self, audio_data):
        """Apply Automatic Gain Control"""
        # Simple AGC implementation
        target_level = 0.5
        current_rms = np.sqrt(np.mean(audio_data ** 2))

        if current_rms > 0:
            gain = target_level / current_rms
            # Limit gain to prevent excessive amplification
            gain = min(gain, 10.0)
            return audio_data * gain

        return audio_data

    def publish_speech_data(self, audio_data):
        """Publish processed speech data for higher-level processing"""
        # This would publish to ROS topic in real implementation
        # For simulation, just log
        duration = len(audio_data) / self.sample_rate
        self.get_logger().info(f'Processed speech segment: {duration:.2f}s')

    def get_logger(self):
        """Simple logger for simulation"""
        class Logger:
            def info(self, msg):
                print(f'[INFO] {msg}')
            def error(self, msg):
                print(f'[ERROR] {msg}')
        return Logger()

    def stop_capture(self):
        """Stop audio capture"""
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()


class NoiseReducer:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.noise_buffer = collections.deque(maxlen=10000)  # 10 seconds of noise samples
        self.noise_estimate = None

    def reduce_noise(self, audio_data):
        """Reduce noise using spectral subtraction"""
        # Update noise estimate if we have silence
        if self.is_silence(audio_data):
            self.noise_buffer.extend(audio_data)
            if len(self.noise_buffer) == self.noise_buffer.maxlen:
                self.noise_estimate = np.array(self.noise_buffer)

        # Apply noise reduction if we have an estimate
        if self.noise_estimate is not None and len(self.noise_estimate) > 0:
            # Simple spectral subtraction approach
            audio_fft = np.fft.fft(audio_data)
            noise_fft = np.fft.fft(self.noise_estimate[:len(audio_data)])

            # Calculate magnitude spectra
            audio_mag = np.abs(audio_fft)
            noise_mag = np.abs(noise_fft)

            # Subtract noise (with flooring to prevent negative values)
            enhanced_mag = np.maximum(audio_mag - 0.5 * noise_mag, 0.1 * audio_mag)

            # Apply phase from original signal
            enhanced_fft = enhanced_mag * np.exp(1j * np.angle(audio_fft))

            # Inverse FFT
            enhanced_audio = np.real(np.fft.ifft(enhanced_fft))
            return enhanced_audio.astype(np.float32)

        return audio_data

    def is_silence(self, audio_data):
        """Check if audio segment is silence"""
        rms = np.sqrt(np.mean(audio_data ** 2))
        return rms < 0.01  # Adjust threshold as needed


class SpeechEnhancer:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def enhance(self, audio_data):
        """Enhance speech quality"""
        # Apply pre-emphasis filter
        enhanced = self.pre_emphasis_filter(audio_data)

        # Apply spectral enhancement
        enhanced = self.spectral_enhancement(enhanced)

        return enhanced

    def pre_emphasis_filter(self, audio_data, coeff=0.97):
        """Apply pre-emphasis filter"""
        return np.append(audio_data[0], audio_data[1:] - coeff * audio_data[:-1])

    def spectral_enhancement(self, audio_data):
        """Apply basic spectral enhancement"""
        # This is a simplified version
        # In practice, use more sophisticated methods like Wiener filtering
        return audio_data
```

## Multimodal Fusion

### Combining Visual and Audio Information

The fusion of visual and audio information creates a more robust perception system:

```python
#!/usr/bin/env python3

import numpy as np
from scipy.spatial.distance import cdist
import threading
import time
from collections import defaultdict, deque


class MultimodalFusionNode:
    def __init__(self):
        # Buffers for different modalities
        self.visual_buffer = deque(maxlen=10)  # Recent visual observations
        self.audio_buffer = deque(maxlen=10)   # Recent audio observations
        self.temporal_window = 2.0  # seconds to consider for fusion

        # Fusion parameters
        self.confidence_weights = {
            'visual': 0.7,
            'audio': 0.3
        }

        # Event detection
        self.event_detector = EventDetector()

        # Attention mechanisms
        self.visual_attention = VisualAttention()
        self.audio_attention = AudioAttention()

        # State variables
        self.fusion_lock = threading.Lock()
        self.last_fusion_time = time.time()

        # Confidence thresholds
        self.detection_confidence_threshold = 0.6
        self.event_confidence_threshold = 0.7

        self.get_logger().info('Multimodal fusion system initialized')

    def fuse_perception_data(self, visual_data, audio_data, timestamp):
        """Fuse visual and audio data into coherent understanding"""
        with self.fusion_lock:
            # Add new data to buffers
            self.visual_buffer.append({
                'data': visual_data,
                'timestamp': timestamp,
                'modality': 'visual'
            })

            self.audio_buffer.append({
                'data': audio_data,
                'timestamp': timestamp,
                'modality': 'audio'
            })

            # Perform fusion
            fused_result = self.perform_fusion(visual_data, audio_data, timestamp)

            # Detect events
            events = self.event_detector.detect_events(
                self.visual_buffer, self.audio_buffer, timestamp
            )

            # Apply attention mechanisms
            attended_result = self.apply_attention(fused_result, events)

            return attended_result

    def perform_fusion(self, visual_data, audio_data, timestamp):
        """Perform multimodal fusion"""
        # Simple weighted fusion approach
        # In practice, use more sophisticated methods like neural networks

        fusion_result = {
            'timestamp': timestamp,
            'fused_features': {},
            'confidence': 0.0,
            'attention_map': {},
            'spatial_context': {},
            'temporal_context': {}
        }

        # Extract relevant features from each modality
        visual_features = self.extract_visual_features(visual_data)
        audio_features = self.extract_audio_features(audio_data)

        # Combine features with weights
        combined_features = {}
        for key, value in visual_features.items():
            combined_features[f'visual_{key}'] = value * self.confidence_weights['visual']

        for key, value in audio_features.items():
            combined_features[f'audio_{key}'] = value * self.confidence_weights['audio']

        # Calculate overall confidence
        visual_conf = visual_features.get('confidence', 0.0)
        audio_conf = audio_features.get('confidence', 0.0)
        fusion_result['confidence'] = (
            visual_conf * self.confidence_weights['visual'] +
            audio_conf * self.confidence_weights['audio']
        )

        fusion_result['fused_features'] = combined_features
        fusion_result['spatial_context'] = self.create_spatial_context(visual_data)
        fusion_result['temporal_context'] = self.create_temporal_context(timestamp)

        return fusion_result

    def extract_visual_features(self, visual_data):
        """Extract features from visual data"""
        features = {
            'objects_count': len(visual_data.get('objects', [])),
            'people_count': visual_data.get('people_count', 0),
            'scene_type': visual_data.get('room_type', 'unknown'),
            'confidence': visual_data.get('confidence', 0.8),
            'dominant_colors': self.get_dominant_colors(visual_data),
            'motion_vectors': self.get_motion_vectors(visual_data)
        }
        return features

    def extract_audio_features(self, audio_data):
        """Extract features from audio data"""
        features = {
            'energy_level': self.calculate_audio_energy(audio_data),
            'spectral_features': self.calculate_spectral_features(audio_data),
            'voice_activity': self.detect_voice_activity(audio_data),
            'noise_level': self.estimate_noise_level(audio_data),
            'confidence': 0.8  # Placeholder
        }
        return features

    def calculate_audio_energy(self, audio_data):
        """Calculate audio energy"""
        return np.mean(audio_data ** 2) if len(audio_data) > 0 else 0.0

    def calculate_spectral_features(self, audio_data):
        """Calculate basic spectral features"""
        if len(audio_data) == 0:
            return {'centroid': 0.0, 'bandwidth': 0.0}

        # Simple spectral centroid calculation
        fft = np.fft.fft(audio_data)
        magnitude = np.abs(fft[:len(fft)//2])  # Only positive frequencies
        freqs = np.arange(len(magnitude))

        if np.sum(magnitude) > 0:
            centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        else:
            centroid = 0.0

        return {'centroid': centroid, 'bandwidth': np.std(freqs) if len(freqs) > 1 else 0.0}

    def detect_voice_activity(self, audio_data):
        """Detect voice activity in audio"""
        # Placeholder - in practice, use VAD
        energy = self.calculate_audio_energy(audio_data)
        return energy > 0.01  # Simple threshold

    def estimate_noise_level(self, audio_data):
        """Estimate noise level"""
        return np.std(audio_data) if len(audio_data) > 0 else 0.0

    def get_dominant_colors(self, visual_data):
        """Get dominant colors from visual scene"""
        # Placeholder implementation
        return ['white', 'black', 'gray']

    def get_motion_vectors(self, visual_data):
        """Get motion vectors from visual data"""
        # Placeholder implementation
        return []

    def create_spatial_context(self, visual_data):
        """Create spatial context from visual data"""
        context = {
            'objects_positions': {},
            'navigation_clearance': True,
            'interaction_zones': [],
            'obstacle_density': 0.0
        }

        # Process object positions if available
        objects = visual_data.get('objects', {})
        for obj_id, obj_data in objects.items():
            if 'center' in obj_data:
                context['objects_positions'][obj_id] = obj_data['center']

        return context

    def create_temporal_context(self, timestamp):
        """Create temporal context"""
        return {
            'current_time': timestamp,
            'time_since_last_fusion': timestamp - self.last_fusion_time
        }

    def apply_attention(self, fusion_result, events):
        """Apply attention mechanisms to focus on relevant information"""
        # Visual attention based on saliency
        attended_visual = self.visual_attention.focus(
            fusion_result['spatial_context']
        )

        # Audio attention based on activity
        attended_audio = self.audio_attention.focus(
            fusion_result['fused_features']
        )

        # Combine attended information
        attended_result = {
            'main_focus': self.determine_main_focus(attended_visual, attended_audio),
            'attention_weights': self.calculate_attention_weights(
                attended_visual, attended_audio
            ),
            'relevance_score': self.calculate_relevance_score(
                fusion_result, events
            )
        }

        # Update last fusion time
        self.last_fusion_time = fusion_result['timestamp']

        return {**fusion_result, **attended_result}

    def determine_main_focus(self, attended_visual, attended_audio):
        """Determine main focus based on attended modalities"""
        # In practice, this would use more sophisticated attention mechanisms
        if attended_visual.get('relevance', 0.0) > attended_audio.get('relevance', 0.0):
            return 'visual'
        else:
            return 'audio'

    def calculate_attention_weights(self, attended_visual, attended_audio):
        """Calculate attention weights"""
        weights = {
            'visual': attended_visual.get('relevance', 0.5),
            'audio': attended_audio.get('relevance', 0.5)
        }
        return weights

    def calculate_relevance_score(self, fusion_result, events):
        """Calculate relevance score for the fused result"""
        # Consider confidence, presence of events, and temporal factors
        base_score = fusion_result['confidence']

        # Boost if significant events detected
        if events:
            event_score = max([event.get('confidence', 0.0) for event in events])
            base_score = max(base_score, event_score)

        # Consider temporal context
        time_factor = min(1.0, fusion_result['temporal_context']['time_since_last_fusion'] / 5.0)
        base_score *= (1.0 + time_factor * 0.2)  # Slight boost for infrequent updates

        return min(1.0, base_score)

    def get_logger(self):
        """Simple logger for simulation"""
        class Logger:
            def info(self, msg):
                print(f'[INFO] {msg}')
            def error(self, msg):
                print(f'[ERROR] {msg}')
        return Logger()


class EventDetector:
    def __init__(self):
        self.event_templates = {
            'person_approaching': {
                'visual_cues': ['person', 'motion_toward_robot'],
                'audio_cues': ['footsteps', 'voice'],
                'temporal_pattern': 'approaching_motion'
            },
            'object_interaction': {
                'visual_cues': ['hand_movement', 'object_grab'],
                'audio_cues': ['grasp_sound'],
                'temporal_pattern': 'interaction_sequence'
            },
            'environment_change': {
                'visual_cues': ['light_change', 'object_appears'],
                'audio_cues': ['door_open', 'new_sound'],
                'temporal_pattern': 'abrupt_change'
            }
        }

    def detect_events(self, visual_buffer, audio_buffer, current_time):
        """Detect events from multimodal buffers"""
        events = []

        # Analyze recent visual and audio data for events
        recent_visual = [item for item in visual_buffer
                        if current_time - item['timestamp'] < 2.0]  # Last 2 seconds
        recent_audio = [item for item in audio_buffer
                       if current_time - item['timestamp'] < 2.0]

        # Detect person approaching
        if self.detect_person_approaching(recent_visual, recent_audio):
            events.append({
                'type': 'person_approaching',
                'confidence': 0.8,
                'timestamp': current_time,
                'details': {'distance': 'unknown', 'direction': 'unknown'}
            })

        # Detect other events based on patterns
        # ... additional event detection logic

        return events

    def detect_person_approaching(self, visual_buffer, audio_buffer):
        """Detect if a person is approaching the robot"""
        # Look for person detection with increasing proximity over time
        # and potential audio cues like footsteps
        person_detections = [v for v in visual_buffer
                           if v['data'].get('people_count', 0) > 0]

        if len(person_detections) >= 2:
            # Check if person count is increasing or person is getting closer
            # (in a real system, track person positions over time)
            return True

        return False


class VisualAttention:
    def focus(self, spatial_context):
        """Apply visual attention to spatial context"""
        attention_map = {}
        relevance = 0.0

        # Focus on objects in interaction zones
        interaction_zones = spatial_context.get('interaction_zones', [])
        objects_positions = spatial_context.get('objects_positions', {})

        for obj_id, position in objects_positions.items():
            if self.is_in_interaction_zone(position, interaction_zones):
                attention_map[obj_id] = 1.0  # High attention
                relevance += 0.3
            else:
                attention_map[obj_id] = 0.1  # Low attention

        return {
            'attention_map': attention_map,
            'relevance': min(1.0, relevance),
            'focus_objects': [k for k, v in attention_map.items() if v > 0.5]
        }

    def is_in_interaction_zone(self, position, zones):
        """Check if position is in interaction zone"""
        # Placeholder implementation
        return True  # For simulation, consider all as relevant


class AudioAttention:
    def focus(self, fused_features):
        """Apply audio attention to fused features"""
        relevance = fused_features.get('audio_energy_level', 0.0)
        return {
            'relevance': min(1.0, relevance * 5.0),  # Scale up for relevance
            'focus_cues': ['voice_activity', 'energy_peaks']
        }
```

## Real-time Performance Optimization

### Efficient Processing Pipelines

For humanoid robots, real-time performance is critical:

```python
#!/usr/bin/env python3

import time
import threading
import queue
import numpy as np
from functools import wraps
import psutil
import gc


class PerceptionOptimizer:
    def __init__(self):
        self.component_timings = {}
        self.resource_usage = {}
        self.throughput_stats = {}
        self.fps_counters = {}

        # Processing queues with size limits
        self.processing_queues = {}
        self.max_queue_size = 5

        # Performance thresholds
        self.cpu_threshold = 80.0  # percent
        self.memory_threshold = 80.0  # percent

        # Adaptive processing parameters
        self.processing_quality = 1.0  # 0.0 to 1.0, 1.0 = full quality
        self.downscale_factor = 1.0   # For image processing

        # Threading and multiprocessing setup
        self.worker_threads = {}
        self.processing_pool = None

        self.get_logger().info('Perception optimizer initialized')

    def adaptive_resource_allocation(self):
        """Dynamically adjust processing parameters based on system load"""
        # Monitor system resources
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent

        self.resource_usage = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'timestamp': time.time()
        }

        # Adjust processing quality based on load
        if cpu_percent > self.cpu_threshold or memory_percent > self.memory_threshold:
            # Reduce processing quality to maintain performance
            self.processing_quality = max(0.3, self.processing_quality * 0.9)
            self.downscale_factor = max(0.5, self.downscale_factor * 0.95)
        else:
            # Gradually increase quality if resources allow
            self.processing_quality = min(1.0, self.processing_quality * 1.01)
            self.downscale_factor = min(1.0, self.downscale_factor * 1.005)

        self.get_logger().info(
            f'Resources: CPU={cpu_percent:.1f}%, Memory={memory_percent:.1f}%, '
            f'Quality={self.processing_quality:.2f}, Downscale={self.downscale_factor:.2f}'
        )

    def optimize_processing_pipeline(self):
        """Optimize the processing pipeline for real-time performance"""
        # 1. Asynchronous processing where possible
        # 2. Batch processing for ML models
        # 3. Caching of expensive computations
        # 4. Early termination of low-confidence detections

        # Example: Create optimized processing queues
        self.processing_queues['vision'] = queue.Queue(maxsize=self.max_queue_size)
        self.processing_queues['audio'] = queue.Queue(maxsize=self.max_queue_size)
        self.processing_queues['fusion'] = queue.Queue(maxsize=self.max_queue_size)

        # Start worker threads
        self.start_worker_threads()

    def start_worker_threads(self):
        """Start worker threads for different processing components"""
        # Vision processing thread
        self.worker_threads['vision'] = threading.Thread(
            target=self.vision_worker, daemon=True)
        self.worker_threads['vision'].start()

        # Audio processing thread
        self.worker_threads['audio'] = threading.Thread(
            target=self.audio_worker, daemon=True)
        self.worker_threads['audio'].start()

        # Fusion processing thread
        self.worker_threads['fusion'] = threading.Thread(
            target=self.fusion_worker, daemon=True)
        self.worker_threads['fusion'].start()

    def vision_worker(self):
        """Worker thread for vision processing"""
        while True:
            try:
                # Get vision data with timeout
                vision_data = self.processing_queues['vision'].get(timeout=0.1)

                # Process with adaptive parameters
                result = self.optimized_vision_process(vision_data)

                # Add to fusion queue
                try:
                    self.processing_queues['fusion'].put_nowait(result)
                except queue.Full:
                    # Drop frame if fusion queue is full
                    pass

                self.processing_queues['vision'].task_done()

            except queue.Empty:
                continue  # Timeout, continue loop
            except Exception as e:
                self.get_logger().error(f'Vision worker error: {e}')

    def audio_worker(self):
        """Worker thread for audio processing"""
        while True:
            try:
                # Get audio data with timeout
                audio_data = self.processing_queues['audio'].get(timeout=0.1)

                # Process with adaptive parameters
                result = self.optimized_audio_process(audio_data)

                # Add to fusion queue
                try:
                    self.processing_queues['fusion'].put_nowait(result)
                except queue.Full:
                    # Drop frame if fusion queue is full
                    pass

                self.processing_queues['audio'].task_done()

            except queue.Empty:
                continue  # Timeout, continue loop
            except Exception as e:
                self.get_logger().error(f'Audio worker error: {e}')

    def fusion_worker(self):
        """Worker thread for fusion processing"""
        while True:
            try:
                # Get fusion data with timeout
                fusion_data = self.processing_queues['fusion'].get(timeout=0.1)

                # Process fusion
                result = self.optimized_fusion_process(fusion_data)

                # For simulation, just log the result
                self.get_logger().info(f'Fusion result processed: {type(result)}')

                self.processing_queues['fusion'].task_done()

            except queue.Empty:
                continue  # Timeout, continue loop
            except Exception as e:
                self.get_logger().error(f'Fusion worker error: {e}')

    def optimized_vision_process(self, data):
        """Optimized vision processing with adaptive parameters"""
        start_time = time.time()

        # Apply downscaling if needed
        if self.downscale_factor < 1.0:
            height, width = data.get('image_shape', (480, 640))
            new_height = int(height * self.downscale_factor)
            new_width = int(width * self.downscale_factor)
            # Simulate image resizing
            data['processed_resolution'] = (new_width, new_height)

        # Perform vision processing with quality adjustment
        if self.processing_quality > 0.5:
            # Full processing
            result = self.full_vision_processing(data)
        else:
            # Simplified processing
            result = self.simplified_vision_processing(data)

        # Record timing
        processing_time = time.time() - start_time
        self.update_fps_counter('vision', processing_time)

        return result

    def optimized_audio_process(self, data):
        """Optimized audio processing with adaptive parameters"""
        start_time = time.time()

        # Perform audio processing
        if self.processing_quality > 0.3:
            # Full processing
            result = self.full_audio_processing(data)
        else:
            # Simplified processing
            result = self.simplified_audio_processing(data)

        # Record timing
        processing_time = time.time() - start_time
        self.update_fps_counter('audio', processing_time)

        return result

    def optimized_fusion_process(self, data):
        """Optimized fusion processing with adaptive parameters"""
        start_time = time.time()

        # Perform fusion processing
        if self.processing_quality > 0.4:
            # Full processing
            result = self.full_fusion_processing(data)
        else:
            # Simplified processing
            result = self.simplified_fusion_processing(data)

        # Record timing
        processing_time = time.time() - start_time
        self.update_fps_counter('fusion', processing_time)

        return result

    def full_vision_processing(self, data):
        """Full vision processing pipeline"""
        # Simulate comprehensive vision processing
        result = {
            'type': 'vision_result',
            'detections': self.simulate_detections(),
            'features': self.simulate_features(),
            'timestamp': time.time()
        }
        return result

    def simplified_vision_processing(self, data):
        """Simplified vision processing pipeline"""
        # Simulate basic vision processing
        result = {
            'type': 'vision_result',
            'detections': self.simulate_simple_detections(),
            'features': None,
            'timestamp': time.time()
        }
        return result

    def full_audio_processing(self, data):
        """Full audio processing pipeline"""
        # Simulate comprehensive audio processing
        result = {
            'type': 'audio_result',
            'features': self.simulate_audio_features(),
            'voice_detected': True,
            'timestamp': time.time()
        }
        return result

    def simplified_audio_processing(self, data):
        """Simplified audio processing pipeline"""
        # Simulate basic audio processing
        result = {
            'type': 'audio_result',
            'energy': self.simulate_audio_energy(),
            'voice_detected': False,
            'timestamp': time.time()
        }
        return result

    def full_fusion_processing(self, data):
        """Full fusion processing pipeline"""
        # Simulate comprehensive fusion processing
        result = {
            'type': 'fusion_result',
            'confidence': 0.8,
            'context': {'fused': True},
            'timestamp': time.time()
        }
        return result

    def simplified_fusion_processing(self, data):
        """Simplified fusion processing pipeline"""
        # Simulate basic fusion processing
        result = {
            'type': 'fusion_result',
            'confidence': 0.6,
            'context': {'fused': False},
            'timestamp': time.time()
        }
        return result

    def simulate_detections(self):
        """Simulate object detections"""
        return [
            {'class': 'person', 'confidence': 0.85, 'bbox': [100, 100, 200, 300]},
            {'class': 'chair', 'confidence': 0.72, 'bbox': [300, 200, 150, 150]}
        ]

    def simulate_simple_detections(self):
        """Simulate simple object detections"""
        return [
            {'class': 'person', 'confidence': 0.85},
            {'class': 'object', 'confidence': 0.65}
        ]

    def simulate_features(self):
        """Simulate feature extraction"""
        return np.random.rand(128).astype(np.float32)

    def simulate_audio_features(self):
        """Simulate audio features"""
        return {'mfcc': np.random.rand(13), 'energy': 0.45, 'spectral_centroid': 1200}

    def simulate_audio_energy(self):
        """Simulate audio energy"""
        return np.random.random()

    def update_fps_counter(self, component, processing_time):
        """Update FPS counter for a component"""
        if component not in self.fps_counters:
            self.fps_counters[component] = {'times': []}

        self.fps_counters[component]['times'].append(processing_time)

        # Keep only last 10 measurements
        if len(self.fps_counters[component]['times']) > 10:
            self.fps_counters[component]['times'] = \
                self.fps_counters[component]['times'][-10:]

        # Calculate FPS
        if len(self.fps_counters[component]['times']) > 0:
            avg_time = np.mean(self.fps_counters[component]['times'])
            fps = 1.0 / avg_time if avg_time > 0 else 0
            self.fps_counters[component]['fps'] = fps

    def get_performance_metrics(self):
        """Get current performance metrics"""
        metrics = {
            'resource_usage': self.resource_usage,
            'processing_quality': self.processing_quality,
            'downscale_factor': self.downscale_factor,
            'fps': {comp: data.get('fps', 0) for comp, data in self.fps_counters.items()}
        }
        return metrics

    def get_logger(self):
        """Simple logger for simulation"""
        class Logger:
            def info(self, msg):
                print(f'[INFO] {msg}')
            def error(self, msg):
                print(f'[ERROR] {msg}')
        return Logger()


def timing_decorator(func):
    """Decorator to measure execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper
```

## Testing and Validation

### Perception System Testing

Comprehensive testing of perception systems is crucial:

```python
#!/usr/bin/env python3

import unittest
import numpy as np
import cv2
from unittest.mock import Mock, patch


class TestPerceptionPipeline(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.test_audio = np.random.random(16000).astype(np.float32)  # 1 second at 16kHz

    def test_visual_perception_initialization(self):
        """Test that visual perception node initializes correctly."""
        node = VisualPerceptionNode()

        # Check that required attributes are initialized
        self.assertIsNotNone(node.bridge)
        self.assertIsNotNone(node.object_detector)
        self.assertIsNotNone(node.feature_extractor)
        self.assertIsNotNone(node.scene_analyzer)

        # Check default parameters
        self.assertEqual(node.detection_threshold, 0.5)
        self.assertEqual(node.tracking_threshold, 0.3)

    def test_object_detection(self):
        """Test object detection functionality."""
        detector = ObjectDetector()

        # Test detection on random image
        detections = detector.detect(self.test_image, threshold=0.5)

        # Should return a list
        self.assertIsInstance(detections, list)

        # Each detection should have required fields
        for detection in detections:
            self.assertIn('class', detection)
            self.assertIn('bbox', detection)
            self.assertIn('center', detection)
            self.assertIn('confidence', detection)
            self.assertGreaterEqual(detection['confidence'], 0.5)

    def test_feature_extraction(self):
        """Test feature extraction functionality."""
        extractor = FeatureExtractor()

        # Extract features from test image
        features = extractor.extract(self.test_image)

        # Should return numpy array
        self.assertIsInstance(features, np.ndarray)

        # Should have expected shape (simplified for test)
        self.assertGreaterEqual(len(features.shape), 2)

    def test_scene_analysis(self):
        """Test scene analysis functionality."""
        analyzer = SceneAnalyzer()

        # Create mock detections
        mock_detections = [
            {'class': 'person', 'bbox': (100, 100, 50, 100), 'confidence': 0.8},
            {'class': 'chair', 'bbox': (200, 200, 80, 80), 'confidence': 0.7}
        ]

        # Analyze scene
        result = analyzer.analyze(self.test_image, mock_detections)

        # Should return expected structure
        self.assertIn('summary', result)
        self.assertIn('room_type', result)
        self.assertIn('activity', result)
        self.assertIn('objects', result)
        self.assertIn('people', result)

    def test_object_tracking(self):
        """Test object tracking functionality."""
        tracker = ObjectTracker()

        # Create initial detections
        initial_detections = [
            {'class': 'person', 'bbox': (100, 100, 50, 100), 'confidence': 0.8, 'center': (125, 150)},
            {'class': 'chair', 'bbox': (200, 200, 80, 80), 'confidence': 0.7, 'center': (240, 240)}
        ]

        # Update tracker with initial detections
        tracked1 = tracker.update(initial_detections, Mock())

        # Create updated detections (simulating next frame)
        updated_detections = [
            {'class': 'person', 'bbox': (105, 105, 50, 100), 'confidence': 0.8, 'center': (130, 155)},
            {'class': 'chair', 'bbox': (205, 205, 80, 80), 'confidence': 0.7, 'center': (245, 245)}
        ]

        # Update tracker with new detections
        tracked2 = tracker.update(updated_detections, Mock())

        # Should maintain object identities
        self.assertEqual(len(tracked1), len(tracked2))

    def test_audio_processing(self):
        """Test audio processing functionality."""
        processor = AudioProcessingNode()

        # Test preprocessing
        processed = processor.preprocess_audio(self.test_audio)

        # Should return numpy array of same length
        self.assertIsInstance(processed, np.ndarray)
        self.assertEqual(len(processed), len(self.test_audio))

    def test_noise_reduction(self):
        """Test noise reduction functionality."""
        reducer = NoiseReducer(16000)

        # Add some noise to test signal
        noisy_signal = self.test_audio + 0.1 * np.random.randn(len(self.test_audio))

        # Reduce noise
        denoised = reducer.reduce_noise(noisy_signal)

        # Should return numpy array of same length
        self.assertIsInstance(denoised, np.ndarray)
        self.assertEqual(len(denoised), len(noisy_signal))

    def test_multimodal_fusion(self):
        """Test multimodal fusion functionality."""
        fusion_node = MultimodalFusionNode()

        # Create mock visual and audio data
        visual_data = {
            'objects': {'obj1': {'class': 'person', 'center': (100, 100)}},
            'people_count': 1,
            'room_type': 'office',
            'confidence': 0.8
        }

        audio_data = self.test_audio

        # Perform fusion
        result = fusion_node.perform_fusion(visual_data, audio_data, time.time())

        # Should return expected structure
        self.assertIn('fused_features', result)
        self.assertIn('confidence', result)
        self.assertIn('spatial_context', result)
        self.assertIn('temporal_context', result)

        # Confidence should be reasonable
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)

    def test_performance_optimization(self):
        """Test performance optimization functionality."""
        optimizer = PerceptionOptimizer()

        # Test adaptive resource allocation
        optimizer.adaptive_resource_allocation()

        # Should update resource usage
        self.assertIn('cpu_percent', optimizer.resource_usage)
        self.assertIn('memory_percent', optimizer.resource_usage)

        # Should have reasonable processing quality
        self.assertGreaterEqual(optimizer.processing_quality, 0.0)
        self.assertLessEqual(optimizer.processing_quality, 1.0)

    def test_event_detection(self):
        """Test event detection functionality."""
        detector = EventDetector()

        # Create mock visual and audio buffers
        visual_buffer = deque(maxlen=10)
        audio_buffer = deque(maxlen=10)

        # Add mock data
        for i in range(5):
            visual_buffer.append({
                'data': {'people_count': 1},
                'timestamp': time.time() - i * 0.1
            })
            audio_buffer.append({
                'data': np.random.random(1000),
                'timestamp': time.time() - i * 0.1
            })

        # Detect events
        events = detector.detect_events(visual_buffer, audio_buffer, time.time())

        # Should return a list
        self.assertIsInstance(events, list)

    def tearDown(self):
        """Clean up after each test method."""
        pass


def run_tests():
    """Run all perception pipeline tests."""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == '__main__':
    # Run tests
    run_tests()
```

## Troubleshooting Common Issues

### Common Perception System Problems and Solutions

#### Issue 1: High CPU Usage in Visual Processing
**Symptoms**: System becomes unresponsive, dropped frames, slow processing
**Causes**:
- High-resolution image processing
- Complex ML models
- Synchronous processing
- Lack of optimization
**Solutions**:
- Implement image downscaling for processing
- Use optimized OpenCV functions
- Implement asynchronous processing
- Use GPU acceleration where available
- Reduce processing frequency for non-critical tasks

#### Issue 2: Audio Noise and Interference
**Symptoms**: Poor speech recognition, false triggers, audio artifacts
**Causes**:
- Environmental noise
- Electrical interference
- Poor microphone placement
- Inadequate filtering
**Solutions**:
- Implement noise reduction algorithms
- Use directional microphones
- Apply appropriate filtering
- Implement voice activity detection
- Use microphone arrays for beamforming

#### Issue 3: Inconsistent Object Tracking
**Symptoms**: Objects disappearing/appearing randomly, incorrect identities
**Causes**:
- Occlusions
- Fast motion
- Similar objects
- Lighting changes
**Solutions**:
- Implement robust feature matching
- Use temporal consistency checks
- Apply Kalman filtering for prediction
- Use appearance models for re-identification
- Implement multi-object tracking algorithms

#### Issue 4: Multimodal Fusion Inconsistencies
**Symptoms**: Contradictory information from different modalities
**Causes**:
- Temporal misalignment
- Different confidence levels
- Sensor calibration issues
- Integration problems
**Solutions**:
- Implement proper temporal synchronization
- Use confidence-based weighting
- Regular sensor calibration
- Implement consistency checks
- Use probabilistic fusion methods

## Best Practices for Perception Systems

### 1. Robustness and Reliability
- Implement graceful degradation when sensors fail
- Use redundant sensors where critical
- Handle edge cases and unexpected inputs
- Include error recovery mechanisms
- Validate sensor data before processing

### 2. Real-time Performance
- Optimize critical processing paths
- Use asynchronous processing where possible
- Implement adaptive quality based on system load
- Monitor and log performance metrics
- Use efficient algorithms and data structures

### 3. Calibration and Maintenance
- Regular sensor calibration procedures
- Environmental adaptation mechanisms
- Self-diagnostic capabilities
- Easy maintenance and replacement
- Documentation of calibration procedures

### 4. Privacy and Security
- Protect sensitive visual and audio data
- Implement data anonymization where needed
- Secure communication channels
- Access control for perception data
- Compliance with privacy regulations

## Key Takeaways

- Perception systems are the sensory foundation of humanoid robots
- Multimodal fusion combines visual, audio, and other sensory inputs
- Real-time performance optimization is critical for responsive behavior
- Robust tracking and recognition algorithms handle dynamic environments
- Proper testing and validation ensure system reliability
- Privacy and security considerations are important for deployed systems
- Adaptive processing maintains performance under varying conditions

In the next section, we'll explore multimodal pipeline examples that connect vision, language, and action in practical humanoid robot applications.