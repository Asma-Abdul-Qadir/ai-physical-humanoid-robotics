---
sidebar_position: 2
---

# Hardware Requirements

This section outlines the hardware requirements for working with the Physical AI and Humanoid Robotics book. We've designed three tiers to accommodate different budgets and computational needs while ensuring optimal performance for advanced features.

## Three Tiers of Hardware Requirements

### Basic Tier
- **CPU**: Modern quad-core processor
- **RAM**: 8GB minimum
- **GPU**: Integrated graphics or entry-level discrete GPU (e.g., GTX 1660)
- **OS**: Ubuntu 22.04 LTS (primary support)
- **Storage**: 50GB free space
- **Use Case**: Basic ROS 2 operations, simple simulations, reading and exercises

### Recommended Tier
- **CPU**: Modern hexa-core or octa-core processor
- **RAM**: 16GB
- **GPU**: Mid-range discrete GPU (e.g., RTX 3060) for simulation and AI tasks
- **OS**: Ubuntu 22.04 LTS (primary support)
- **Storage**: 100GB free space
- **Use Case**: Full simulation capabilities, Isaac Sim operations, moderate complexity projects

### Optimal Tier
- **CPU**: High-end multi-core processor
- **RAM**: 32GB or more
- **GPU**: High-end discrete GPU (e.g., RTX 4080) for complex simulations
- **OS**: Ubuntu 22.04 LTS (primary support)
- **Storage**: 200GB+ free space
- **Use Case**: Complex simulations, real-time perception, advanced AI training

## Why These Requirements?

### ROS 2 (Robot Operating System 2)
ROS 2 Humble Hawksbill requires substantial computational resources for:
- Real-time processing of sensor data
- Coordination of multiple robot subsystems
- Simulation environments
- Development tools and visualization

### Simulation Environments
Different simulation environments have varying requirements:
- **Gazebo Garden**: Physics simulation requires significant GPU and CPU resources
- **Unity 2023.2 LTS**: High-quality visualization needs modern graphics capabilities
- **NVIDIA Isaac Sim**: AI and perception training requires powerful GPUs

### Development Tools
Additional tools for development and debugging also require resources:
- IDEs and development environments
- Docker containers for isolated testing
- Multiple processes running simultaneously

## Operating System Considerations

While Ubuntu 22.04 LTS is our primary support target, here are considerations for other platforms:

- **Ubuntu 22.04 LTS**: Primary development and testing platform
- **Other Linux distributions**: May work but with reduced support
- **Windows**: Possible via WSL2 but with performance limitations
- **macOS**: Limited support due to hardware requirements for simulation

## Network Requirements

- **Internet connection**: Required for initial setup, package downloads, and updates
- **Bandwidth**: At least 10 Mbps recommended for large simulation downloads
- **Stability**: Important for remote simulation environments

## Preparation Checklist

Before proceeding with software installation, ensure your system meets:
- [ ] Sufficient free storage space
- [ ] Compatible operating system
- [ ] Adequate RAM and CPU
- [ ] Compatible GPU for simulation needs
- [ ] Stable internet connection

In the next section, we'll cover the software prerequisites needed for this course.