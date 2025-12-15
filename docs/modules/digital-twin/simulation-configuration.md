---
sidebar_position: 5
---

# Simulation Configuration: Hardware Requirements and Optimization

Welcome to the Simulation Configuration module, which focuses on properly configuring simulation environments for optimal performance and ensuring hardware requirements are met for different simulation tasks. This chapter covers the configuration of Gazebo, Unity, and Isaac Sim with attention to hardware requirements and performance optimization.

## Learning Objectives

By the end of this section, you will be able to:
- Determine appropriate hardware requirements for different simulation tasks
- Configure simulation environments for optimal performance
- Optimize simulation parameters based on available hardware
- Understand the trade-offs between simulation fidelity and performance
- Set up multi-machine simulation deployments
- Troubleshoot common performance issues in simulation environments

## Hardware Requirements Overview

### Tiered Hardware Classification

The simulation environments have different hardware requirements depending on the complexity of the simulation and desired performance levels:

#### Basic Tier (Development and Testing)
- **CPU**: 8+ cores (Intel i7 or AMD Ryzen 7 equivalent)
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: NVIDIA RTX 2060/AMD RX 6700 XT or equivalent
- **VRAM**: 8GB minimum
- **Storage**: 500GB SSD
- **OS**: Ubuntu 22.04 LTS or Windows 11
- **Network**: Gigabit Ethernet

#### Recommended Tier (Production and Training)
- **CPU**: 16+ cores (Intel i9 or AMD Threadripper)
- **RAM**: 64GB
- **GPU**: NVIDIA RTX 4080/RTX 6000 Ada or equivalent
- **VRAM**: 16GB+
- **Storage**: 1TB+ NVMe SSD
- **OS**: Ubuntu 22.04 LTS (preferred for server deployments)
- **Network**: 10GbE for multi-node setups

#### Optimal Tier (Large-Scale Deployment)
- **CPU**: 32+ cores (Dual Xeon or EPYC)
- **RAM**: 128GB+
- **GPU**: NVIDIA RTX 6000 Ada, A40, or H100
- **VRAM**: 48GB+ (multiple GPUs for complex scenes)
- **Storage**: 2TB+ NVMe RAID array
- **Network**: 25/40GbE for distributed simulation

## Gazebo Garden Configuration

### Hardware Requirements for Gazebo

Gazebo's performance is primarily CPU-bound for physics simulation and GPU-bound for rendering:

#### Physics Simulation Requirements
- **CPU**: Multi-core processor with good single-thread performance
- **RAM**: 8-16GB for typical robot simulation
- **GPU**: Not critical for pure physics simulation

#### Rendering Requirements
- **GPU**: Modern GPU with OpenGL 3.3+ support
- **VRAM**: 4GB+ for complex scenes
- **VRAM**: 8GB+ for photorealistic rendering

### Configuration Files

#### Physics Engine Configuration
```xml
<!-- physics_config.sdf -->
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="simulation_world">
    <physics name="ode_physics" type="ode">
      <!-- Time stepping -->
      <max_step_size>0.001</max_step_size>  <!-- Smaller = more accurate but slower -->
      <real_time_factor>1.0</real_time_factor>  <!-- 1.0 = real-time -->
      <real_time_update_rate>1000</real_time_update_rate>  <!-- Steps per second -->

      <!-- Gravity -->
      <gravity>0 0 -9.8</gravity>

      <!-- Solver settings -->
      <ode>
        <solver>
          <type>quick</type>  <!-- quick or pgslcp -->
          <iters>20</iters>  <!-- More iterations = more stable but slower -->
          <sor>1.3</sor>  <!-- Successive Over-Relaxation parameter -->
        </solver>
        <constraints>
          <cfm>0.0</cfm>  <!-- Constraint Force Mixing -->
          <erp>0.2</erp>  <!-- Error Reduction Parameter -->
          <contact_max_correcting_vel>100</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>
  </world>
</sdf>
```

#### Performance Configuration
```xml
<!-- performance_config.sdf -->
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="performance_world">
    <!-- Physics settings for better performance -->
    <physics name="fast_physics" type="ode">
      <max_step_size>0.01</max_step_size>  <!-- Larger step for better performance -->
      <real_time_factor>2.0</real_time_factor>  <!-- Allow simulation to run faster than real-time -->
      <real_time_update_rate>100</real_time_update_rate>  <!-- Lower update rate -->

      <ode>
        <solver>
          <iters>10</iters>  <!-- Fewer iterations for better performance -->
          <sor>1.0</sor>
        </solver>
        <constraints>
          <contact_surface_layer>0.01</contact_surface_layer>  <!-- Larger surface layer -->
        </constraints>
      </ode>
    </physics>

    <!-- Threading configuration -->
    <engine>ogre2</engine>  <!-- Rendering engine -->

    <!-- Plugin for performance monitoring -->
    <plugin filename="gz-sim-perf-system" name="gz::sim::systems::PerformanceMonitor">
      <report_frequency>1.0</report_frequency>  <!-- Report every second -->
    </plugin>
  </world>
</sdf>
```

### Environment Variables and Optimization

```bash
# Gazebo environment variables for performance
export GZ_SIM_RESOURCE_PATH=/path/to/models:/path/to/worlds
export GZ_SIM_SYSTEM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/gazebo-11/plugins
export GAZEBO_MODEL_PATH=/path/to/custom/models:$GAZEBO_MODEL_PATH

# Graphics optimization
export MESA_GL_VERSION_OVERRIDE=3.3  # For older graphics cards
export __GL_SYNC_TO_VBLANK=0  # Disable vsync for better performance
export OGRE_RESOURCEMANAGER_STRICT=0  # Less strict resource management

# Physics optimization
export ODE_THREAD_COUNT=4  # Number of threads for ODE physics
export GAZEBO_THREADS=4  # Number of Gazebo threads
```

## Unity Simulation Configuration

### Hardware Requirements for Unity

Unity's simulation performance depends heavily on the rendering pipeline and scene complexity:

#### Rendering Pipeline Requirements
- **Universal Render Pipeline (URP)**: Mid-range GPU with 6GB+ VRAM
- **High Definition Render Pipeline (HDRP)**: High-end GPU with 8GB+ VRAM
- **Built-in Render Pipeline**: Mid-range GPU with 4GB+ VRAM

#### Scene Complexity Factors
- **Poly count**: Higher poly counts require more GPU power
- **Lighting**: Dynamic lighting is GPU-intensive
- **Shadows**: Shadow calculations are computationally expensive
- **Effects**: Post-processing effects impact performance

### Unity Project Configuration

#### Player Settings Optimization
```csharp
// PlayerSettings.cs - Unity configuration for simulation
using UnityEditor;

public class SimulationPlayerSettings
{
    [MenuItem("Tools/Simulation/Configure Player Settings")]
    public static void ConfigureForSimulation()
    {
        // Performance settings
        PlayerSettings.runInBackground = true;
        PlayerSettings.captureSingleScreen = false;
        PlayerSettings.resizableWindow = true;

        // Graphics settings
        PlayerSettings.graphicsJobs = true;  // Enable graphics jobs
        PlayerSettings.streamingMipmapsActive = true;
        PlayerSettings.streamingMipmapsMaxLevelReduction = 2;
        PlayerSettings.streamingMipmapsMaxFileIORequests = 1024;

        // Memory settings
        PlayerSettings.virtualTexturingEnabled = false;  // Disable for performance

        // Quality settings for simulation
        QualitySettings.vSyncCount = 0;  // Disable VSync for performance
        QualitySettings.antiAliasing = 0;  // Disable AA for performance
        QualitySettings.shadowResolution = ShadowResolution.Low;
        QualitySettings.shadowDistance = 50f;  // Limit shadow distance

        Debug.Log("Player settings configured for simulation performance");
    }
}
```

#### Camera and Rendering Optimization
```csharp
// CameraOptimizer.cs
using UnityEngine;

public class CameraOptimizer : MonoBehaviour
{
    [Header("Performance Settings")]
    public bool enableLOD = true;
    public int targetFrameRate = 60;
    public bool disableShadows = false;
    public bool reduceResolution = false;

    private Camera mainCamera;
    private RenderTexture lowResTexture;

    void Start()
    {
        mainCamera = GetComponent<Camera>();
        ConfigureCameraForPerformance();
    }

    void ConfigureCameraForPerformance()
    {
        // Frame rate
        Application.targetFrameRate = targetFrameRate;

        // Shadow settings
        if (disableShadows)
        {
            QualitySettings.shadows = ShadowQuality.Disable;
            QualitySettings.shadowResolution = ShadowResolution.Low;
        }

        // Resolution scaling for performance
        if (reduceResolution)
        {
            int width = Screen.width / 2;
            int height = Screen.height / 2;

            lowResTexture = new RenderTexture(width, height, 24);
            mainCamera.targetTexture = lowResTexture;
        }

        // Occlusion culling for complex scenes
        mainCamera.layerCullDistances = new float[32];  // Configure per-layer culling
    }

    void Update()
    {
        // Dynamic LOD adjustment based on performance
        if (enableLOD)
        {
            AdjustLODBasedOnPerformance();
        }
    }

    void AdjustLODBasedOnPerformance()
    {
        float frameTime = Time.deltaTime;
        float targetFrameTime = 1.0f / targetFrameRate;

        if (frameTime > targetFrameTime * 1.2f)  // Running slow
        {
            // Reduce detail level
            QualitySettings.lodBias = Mathf.Max(0.5f, QualitySettings.lodBias - 0.1f);
        }
        else if (frameTime < targetFrameTime * 0.8f)  // Running fast
        {
            // Increase detail level
            QualitySettings.lodBias = Mathf.Min(2.0f, QualitySettings.lodBias + 0.1f);
        }
    }
}
```

#### Physics Optimization
```csharp
// PhysicsOptimizer.cs
using UnityEngine;

[CreateAssetMenu(fileName = "PhysicsConfig", menuName = "Simulation/Physics Configuration")]
public class PhysicsConfig : ScriptableObject
{
    [Header("Physics Performance Settings")]
    public int solverIterations = 6;
    public int solverVelocityIterations = 1;
    public float sleepThreshold = 0.005f;
    public float defaultContactOffset = 0.01f;
    public float bounceThreshold = 2.0f;

    [Header("Broadphase Settings")]
    public int worldSubdivisions = 8;
    public Vector3 worldBounds = new Vector3(250, 250, 250);

    public void ApplySettings()
    {
        Physics.defaultSolverIterations = solverIterations;
        Physics.defaultSolverVelocityIterations = solverVelocityIterations;
        Physics.sleepThreshold = sleepThreshold;
        Physics.defaultContactOffset = defaultContactOffset;
        Physics.bounceThreshold = bounceThreshold;

        // Set broadphase bounds
        Physics.defaultWorldBounds = new Bounds(Vector3.zero, worldBounds);
        Physics.worldSubdivisions = worldSubdivisions;

        Debug.Log("Physics settings applied for simulation performance");
    }
}
```

## NVIDIA Isaac Sim Configuration

### Hardware Requirements for Isaac Sim

Isaac Sim is the most demanding simulation environment, especially for AI training:

#### Minimum Requirements
- **GPU**: NVIDIA RTX 3080 or A40
- **VRAM**: 12GB minimum
- **CPU**: 8+ cores with high single-thread performance
- **RAM**: 32GB minimum

#### Recommended Requirements
- **GPU**: NVIDIA RTX 4090, RTX 6000 Ada, or A6000
- **VRAM**: 24GB+ for complex scenes
- **CPU**: 16+ cores (Intel i9 or AMD Threadripper)
- **RAM**: 64GB+

### Isaac Sim Configuration Files

#### USD Stage Configuration
```python
# isaac_config.py
import carb
from omni.isaac.core.utils.stage import set_stage_up_axis
from pxr import UsdPhysics, PhysxSchema, Gf
import omni.physx

def configure_isaac_stage():
    """Configure Isaac Sim stage for optimal performance"""

    # Set up axis (Z-up is common for robotics)
    set_stage_up_axis("Z")

    # Get physics scene
    scene = UsdPhysics.Scene.GetAtPath("/physicsScene")

    # Configure PhysX settings
    physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())

    # Performance-oriented settings
    physx_scene_api.CreateSubdivisionsPerFrameAttr(2)  # Default: 4
    physx_scene_api.CreateMaxDeltaTimeAttr(1.0/60.0)  # 60 FPS
    physx_scene_api.CreateMinDeltaTimeAttr(1.0/240.0)  # Minimum time step

    # Solver settings for performance
    physx_scene_api.CreateMaxPositionIterationsAttr(4)  # Default: 8
    physx_scene_api.CreateMaxVelocityIterationsAttr(1)  # Default: 1

    # Contact settings
    physx_scene_api.CreateMaxDepenetrationVelocityAttr(10.0)
    physx_scene_api.CreateMaxContactOffsetAttr(0.01)

    carb.log_info("Isaac Sim stage configured for performance")

def configure_rendering_settings():
    """Configure rendering settings for simulation"""
    settings = carb.settings.get_settings()

    # Disable heavy rendering features for simulation
    settings.set("/rtx/ambientOcclusion/enabled", False)
    settings.set("/rtx/reflections/enable", False)
    settings.set("/rtx/globalillumination/enable", False)
    settings.set("/rtx/pathtracing/enable", False)

    # Enable performance features
    settings.set("/rtx/dlss/enable", True)  # Enable DLSS if supported
    settings.set("/rtx/upscaling/enable", True)

    # Reduce resolution scaling for performance
    settings.set("/app/window/scaleToFrame", 0.75)  # 75% scaling

    carb.log_info("Rendering settings optimized for performance")

def configure_simulation_settings():
    """Configure core simulation settings"""
    # Physics settings
    omni.physx.get_physx_interface().set_num_threads(8)  # Adjust based on CPU

    # Simulation settings
    settings = carb.settings.get_settings()
    settings.set("/app/runLoops/updateLoop.frequency", 60.0)  # 60 Hz update rate
    settings.set("/app/runLoops/renderLoop.frequency", 60.0)  # 60 Hz render rate

    carb.log_info("Simulation settings configured")
```

#### Replicator Configuration for AI Training
```python
# replicator_config.py
import omni.replicator.core as rep

def configure_replicator_for_training():
    """Configure Omniverse Replicator for AI training data generation"""

    # Initialize replicator
    rep.orchestrator.init_async_orchestrator()

    # Set up randomization ranges for synthetic data diversity
    with rep.new_layer():
        # Randomize lighting
        lights = rep.get.light()
        with lights.randomize.uniform_illumination(
            color_temperature=rep.distribution.normal(6500, 500),
            intensity=rep.distribution.uniform(100, 1000),
            scale=rep.distribution.uniform(0.5, 2.0)
        ):
            lights.light_component.position = rep.distribution.uniform((-10, -10, 5), (10, 10, 15))

        # Randomize materials
        materials = rep.get.material()
        with materials.randomize.diffuse_reflection_weight(
            rep.distribution.uniform(0.2, 1.0)
        ):
            materials.material_component.diffuse_reflection_color = rep.distribution.uniform((0, 0, 0), (1, 1, 1))

        # Randomize object placement
        objects = rep.get.prims()
        with objects.randomize.position(
            position_range=rep.distribution.uniform((-5, -5, 0), (5, 5, 2)),
            scale=rep.distribution.uniform((0.5, 0.5, 0.5), (2.0, 2.0, 2.0))
        ):
            pass

def configure_output_settings():
    """Configure output settings for training data"""

    # Set up output writers
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir="./training_data",
        rgb=True,
        semantic_segmentation=True,
        depth=True,
        bounding_box_2d_tight=True,
        instance_segmentation=True
    )

    return writer
```

### GPU Memory Management

```python
# gpu_memory_optimizer.py
import gc
import torch
import omni
from pxr import UsdLux, UsdGeom
import carb

class GPUMemoryOptimizer:
    def __init__(self):
        self.current_memory_usage = 0
        self.max_memory_threshold = 0.8  # 80% of available memory

    def optimize_memory_usage(self):
        """Optimize GPU memory usage during simulation"""

        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # Force garbage collection
        gc.collect()

        # Check current memory usage
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()

            if current_memory > self.max_memory_threshold:
                carb.log_warn(f"GPU memory usage high: {current_memory:.2%}")
                self.reduce_quality_settings()

    def reduce_quality_settings(self):
        """Reduce quality settings to conserve memory"""
        settings = carb.settings.get_settings()

        # Reduce texture streaming pool
        settings.set("/renderer/texturePoolSize", 256)  # Reduce from default

        # Simplify rendering
        settings.set("/rtx/indirectDiffuse/enable", False)
        settings.set("/rtx/directLighting/enable", False)

        carb.log_info("Quality settings reduced to conserve GPU memory")

    def adaptive_texture_streaming(self):
        """Implement adaptive texture streaming based on memory usage"""
        settings = carb.settings.get_settings()

        # Monitor texture memory usage and adjust streaming settings
        if torch.cuda.is_available():
            texture_memory_ratio = torch.cuda.memory_reserved() / torch.cuda.max_memory_allocated()

            if texture_memory_ratio > 0.7:
                # Reduce texture streaming quality
                settings.set("/renderer/textureStreaming/targetMemoryRatio", 0.5)
            elif texture_memory_ratio < 0.3:
                # Increase texture streaming quality if memory allows
                settings.set("/renderer/textureStreaming/targetMemoryRatio", 0.8)
```

## Multi-Machine Simulation Configuration

### Distributed Simulation Setup

For large-scale simulations, distribute across multiple machines:

#### Master Node Configuration
```bash
# master_config.sh
#!/bin/bash

# Master node configuration for distributed simulation
export MASTER_IP="192.168.1.10"
export MASTER_PORT="11345"
export ROS_MASTER_URI="http://$MASTER_IP:11311"

# Isaac Sim multi-node settings
export ISAACSIM_MULTI_GPU=true
export ISAACSIM_NUM_NODES=4
export ISAACSIM_NODES="192.168.1.10,192.168.1.11,192.168.1.12,192.168.1.13"

# Network optimization
export OMNIVERSE_HEADLESS=true  # Run without GUI on slave nodes
export ISAACSIM_DISABLE_RENDERING=false  # Keep rendering on master
```

#### Slave Node Configuration
```bash
# slave_config.sh
#!/bin/bash

# Slave node configuration
export MASTER_IP="192.168.1.10"
export ROS_MASTER_URI="http://$MASTER_IP:11311"
export ROS_HOSTNAME=$(hostname -I | cut -d' ' -f1)

# Isaac Sim slave settings
export ISAACSIM_MULTI_GPU=true
export ISAACSIM_HEADLESS=true  # No GUI on slaves
export ISAACSIM_DISABLE_PHYSICS=false  # Still do physics
```

### Load Balancing Configuration

```python
# load_balancer.py
import subprocess
import psutil
import socket
import json
from typing import Dict, List

class SimulationLoadBalancer:
    def __init__(self, nodes: List[str]):
        self.nodes = nodes
        self.node_loads = {node: 0 for node in nodes}
        self.node_capabilities = {node: self.get_node_capabilities(node) for node in nodes}

    def get_node_capabilities(self, node_ip: str) -> Dict:
        """Get hardware capabilities of a node"""
        # In a real implementation, this would query the node
        # For now, we'll simulate based on common configurations
        if "high" in node_ip.lower() or "powerful" in node_ip.lower():
            return {
                "cpu_cores": 32,
                "gpu_memory_gb": 24,
                "ram_gb": 64,
                "network_bandwidth": 1000  # Mbps
            }
        else:
            return {
                "cpu_cores": 16,
                "gpu_memory_gb": 12,
                "ram_gb": 32,
                "network_bandwidth": 1000
            }

    def assign_simulation_task(self, task_requirements: Dict) -> str:
        """Assign a simulation task to the most appropriate node"""
        available_nodes = []

        for node in self.nodes:
            capabilities = self.node_capabilities[node]

            # Check if node meets requirements
            if (capabilities["cpu_cores"] >= task_requirements.get("min_cpu_cores", 0) and
                capabilities["gpu_memory_gb"] >= task_requirements.get("min_gpu_memory_gb", 0) and
                capabilities["ram_gb"] >= task_requirements.get("min_ram_gb", 0)):

                # Calculate load score (lower is better)
                load_score = self.node_loads[node]
                available_nodes.append((node, load_score))

        if not available_nodes:
            raise Exception("No suitable node available for task")

        # Choose node with lowest load
        best_node = min(available_nodes, key=lambda x: x[1])[0]

        # Update load
        self.node_loads[best_node] += task_requirements.get("estimated_load", 1)

        return best_node

    def get_load_distribution(self) -> Dict:
        """Get current load distribution across nodes"""
        return self.node_loads

# Example usage
def setup_distributed_simulation():
    nodes = ["192.168.1.10", "192.168.1.11", "192.168.1.12", "192.168.1.13"]
    balancer = SimulationLoadBalancer(nodes)

    # Example task requirements
    tasks = [
        {"task_type": "physics_only", "min_cpu_cores": 8, "min_gpu_memory_gb": 0, "estimated_load": 2},
        {"task_type": "rendering_heavy", "min_cpu_cores": 16, "min_gpu_memory_gb": 12, "estimated_load": 4},
        {"task_type": "ai_training", "min_cpu_cores": 16, "min_gpu_memory_gb": 24, "estimated_load": 5}
    ]

    for task in tasks:
        assigned_node = balancer.assign_simulation_task(task)
        print(f"Assigned {task['task_type']} to {assigned_node}")

    print("Load distribution:", balancer.get_load_distribution())
```

## Performance Monitoring and Profiling

### System Monitoring
```bash
# performance_monitor.sh
#!/bin/bash

# Monitor system resources during simulation
LOG_FILE="/tmp/simulation_performance.log"

echo "Starting performance monitoring..." > $LOG_FILE

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # CPU usage
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')

    # Memory usage
    MEMORY_USAGE=$(free | grep Mem | awk '{printf("%.2f", $3/$2 * 100.0)}')

    # GPU usage (if nvidia-smi available)
    if command -v nvidia-smi &> /dev/null; then
        GPU_USAGE=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
        GPU_MEMORY=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits)
    fi

    # Log performance data
    echo "$TIMESTAMP - CPU: ${CPU_USAGE}%, MEM: ${MEMORY_USAGE}%" >> $LOG_FILE
    if [ ! -z "$GPU_USAGE" ]; then
        echo "$TIMESTAMP - GPU: ${GPU_USAGE}%, VRAM: ${GPU_MEMORY}" >> $LOG_FILE
    fi

    sleep 5
done
```

### Simulation Profiling
```python
# simulation_profiler.py
import time
import psutil
import threading
from dataclasses import dataclass
from typing import Dict, List
import json

@dataclass
class PerformanceMetrics:
    timestamp: float
    cpu_percent: float
    memory_percent: float
    gpu_utilization: float
    gpu_memory_used: float
    gpu_memory_total: float
    simulation_fps: float
    physics_steps_per_second: float
    active_joints: int
    active_colliders: int

class SimulationProfiler:
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.profiling_active = False
        self.profile_thread = None
        self.start_time = None
        self.last_physics_step = None
        self.physics_step_count = 0

    def start_profiling(self):
        """Start profiling simulation performance"""
        self.profiling_active = True
        self.start_time = time.time()
        self.profile_thread = threading.Thread(target=self._collect_metrics)
        self.profile_thread.daemon = True
        self.profile_thread.start()

    def stop_profiling(self):
        """Stop profiling and return metrics"""
        self.profiling_active = False
        if self.profile_thread:
            self.profile_thread.join()
        return self.metrics_history

    def _collect_metrics(self):
        """Collect performance metrics in a separate thread"""
        while self.profiling_active:
            # Collect system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent

            # Collect GPU metrics (simplified - in real implementation would use pynvml)
            gpu_utilization = self._get_gpu_utilization()
            gpu_memory = self._get_gpu_memory()

            # Calculate FPS if possible
            current_time = time.time()
            simulation_fps = 0
            physics_steps_ps = 0

            if self.last_physics_step:
                time_delta = current_time - self.last_physics_step
                if time_delta > 0:
                    simulation_fps = 1.0 / time_delta
                    physics_steps_ps = self.physics_step_count / (current_time - self.start_time)

            # Create metrics object
            metrics = PerformanceMetrics(
                timestamp=current_time,
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                gpu_utilization=gpu_utilization,
                gpu_memory_used=gpu_memory[0],
                gpu_memory_total=gpu_memory[1],
                simulation_fps=simulation_fps,
                physics_steps_per_second=physics_steps_ps,
                active_joints=self._get_active_joints(),
                active_colliders=self._get_active_colliders()
            )

            self.metrics_history.append(metrics)

            time.sleep(0.1)  # Collect metrics every 100ms

    def _get_gpu_utilization(self):
        """Get GPU utilization percentage"""
        # In real implementation, use pynvml to get actual GPU stats
        return 0.0

    def _get_gpu_memory(self):
        """Get GPU memory usage"""
        # In real implementation, use pynvml to get actual GPU memory stats
        return 0.0, 0.0

    def _get_active_joints(self):
        """Get number of active joints in simulation"""
        # This would interface with the simulation engine
        return 0

    def _get_active_colliders(self):
        """Get number of active colliders in simulation"""
        # This would interface with the simulation engine
        return 0

    def get_performance_report(self):
        """Generate a performance report from collected metrics"""
        if not self.metrics_history:
            return "No metrics collected"

        # Calculate averages
        avg_cpu = sum(m.cpu_percent for m in self.metrics_history) / len(self.metrics_history)
        avg_memory = sum(m.memory_percent for m in self.metrics_history) / len(self.metrics_history)
        avg_gpu = sum(m.gpu_utilization for m in self.metrics_history) / len(self.metrics_history) if self.metrics_history[0].gpu_utilization > 0 else 0

        peak_cpu = max(m.cpu_percent for m in self.metrics_history)
        peak_memory = max(m.memory_percent for m in self.metrics_history)
        peak_gpu = max(m.gpu_utilization for m in self.metrics_history) if self.metrics_history[0].gpu_utilization > 0 else 0

        report = f"""
Simulation Performance Report:
==============================
Duration: {time.time() - self.start_time:.2f} seconds
Total Metrics Collected: {len(self.metrics_history)}

Average Resource Usage:
- CPU: {avg_cpu:.1f}% (Peak: {peak_cpu:.1f}%)
- Memory: {avg_memory:.1f}% (Peak: {peak_memory:.1f}%)
- GPU: {avg_gpu:.1f}% (Peak: {peak_gpu:.1f}%)

Performance Metrics:
- Average Simulation FPS: {sum(m.simulation_fps for m in self.metrics_history) / len(self.metrics_history):.2f}
- Average Physics Steps/sec: {sum(m.physics_steps_per_second for m in self.metrics_history) / len(self.metrics_history):.2f}
- Active Joints: {self.metrics_history[-1].active_joints}
- Active Colliders: {self.metrics_history[-1].active_colliders}
        """

        return report

# Usage example
def profile_simulation():
    profiler = SimulationProfiler()
    profiler.start_profiling()

    # Run simulation here
    time.sleep(10)  # Simulate running simulation

    profiler.stop_profiling()
    report = profiler.get_performance_report()
    print(report)
```

## Troubleshooting Common Configuration Issues

### Performance Issues
- **Low FPS**: Check GPU VRAM usage, reduce scene complexity
- **Physics instability**: Increase solver iterations or reduce time step
- **Memory exhaustion**: Implement proper asset streaming and cleanup

### Hardware Compatibility
- **Graphics errors**: Update GPU drivers, check OpenGL/DirectX support
- **Physics errors**: Verify CPU architecture compatibility
- **Network issues**: Check firewall settings, network bandwidth

### Simulation-Specific Issues
- **Gazebo**: Check ODE configuration, model complexity
- **Unity**: Verify rendering pipeline compatibility, shader issues
- **Isaac Sim**: Ensure CUDA/RTX compatibility, Omniverse connectivity

## Key Takeaways

- Hardware requirements vary significantly between simulation environments
- Proper configuration can dramatically improve simulation performance
- Multi-machine setups enable large-scale simulation scenarios
- Performance monitoring is essential for optimization
- Each simulation environment has unique configuration requirements

In the next section, we'll explore exercises to reinforce the concepts covered in this module.