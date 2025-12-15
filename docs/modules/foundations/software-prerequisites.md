---
sidebar_position: 3
---

# Software Prerequisites

This section outlines the software requirements and setup procedures for the Physical AI and Humanoid Robotics course. We'll cover the installation and configuration of all necessary tools and frameworks.

## Required Software Versions

For consistency and compatibility, we specify exact versions for all tools:

- **ROS 2**: Humble Hawksbill (LTS version)
- **Gazebo**: Garden
- **Unity**: 2023.2 LTS
- **NVIDIA Isaac Sim**: 2023.2
- **Node.js**: 18.x or higher
- **Docker**: 20.x or higher

## Operating System Setup

We primarily support **Ubuntu 22.04 LTS** as our development environment. While other platforms may work, Ubuntu 22.04 provides the best compatibility with our target tools.

### Installing Ubuntu 22.04 LTS

If you're not already running Ubuntu 22.04 LTS:

1. Download the ISO from the official Ubuntu website
2. Create a bootable USB drive or dual-boot setup
3. Perform a clean installation
4. Update the system: `sudo apt update && sudo apt upgrade`

## ROS 2 Humble Hawksbill Installation

ROS 2 (Robot Operating System 2) is the middleware that connects all components of your robot system.

### Installation Steps

1. Set up your sources list:
   ```bash
   sudo apt update && sudo apt install -y software-properties-common
   sudo add-apt-repository universe
   ```

2. Add the ROS 2 GPG key and repository:
   ```bash
   sudo apt update && sudo apt install -y curl gnupg lsb-release
   sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
   ```

3. Install ROS 2 Humble:
   ```bash
   sudo apt update
   sudo apt install -y ros-humble-desktop
   ```

4. Install colcon build tool:
   ```bash
   sudo apt install -y python3-colcon-common-extensions
   ```

5. Source the ROS 2 environment:
   ```bash
   echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
   source ~/.bashrc
   ```

## Gazebo Garden Installation

Gazebo Garden provides physics-based simulation for testing your robot designs.

### Installation Steps

1. Add the Gazebo GPG key and repository:
   ```bash
   sudo apt update && sudo apt install wget lsb-release gnupg
   sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/gazebo-archive-keyring.gpg
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/gazebo-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
   ```

2. Install Gazebo Garden:
   ```bash
   sudo apt update
   sudo apt install gazebo-garden
   ```

## Node.js and NPM Setup

For the Docusaurus-based documentation system:

1. Install Node.js 18.x:
   ```bash
   curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
   sudo apt install -y nodejs
   ```

2. Verify installation:
   ```bash
   node --version
   npm --version
   ```

## Docker Installation

Docker will be used for isolated testing and development environments:

1. Set up Docker repository:
   ```bash
   sudo apt update
   sudo apt install ca-certificates curl gnupg lsb-release
   sudo mkdir -p /etc/apt/keyrings
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   ```

2. Install Docker Engine:
   ```bash
   sudo apt update
   sudo apt install docker-ce docker-ce-cli containerd.io docker-compose-plugin
   ```

3. Add your user to the docker group:
   ```bash
   sudo usermod -aG docker $USER
   ```

4. Log out and log back in for the changes to take effect.

## Verification Steps

After completing the installation, verify your setup:

1. Test ROS 2:
   ```bash
   ros2 topic list
   ```

2. Test Gazebo:
   ```bash
   gazebo --version
   ```

3. Test Node.js:
   ```bash
   node --version
   npm run build
   ```

4. Test Docker:
   ```bash
   docker run hello-world
   ```

## Troubleshooting Common Issues

### ROS 2 Installation Issues
- If you encounter locale errors, run: `export LC_ALL=C`
- If ROS packages aren't found, ensure you've sourced the setup.bash file

### Gazebo Graphics Issues
- If Gazebo doesn't start properly, ensure you have proper graphics drivers installed
- For virtual machines, ensure 3D acceleration is enabled

### Node.js Permission Issues
- If you encounter permission errors with npm, consider using a Node.js version manager like nvm

## Next Steps

Once you've successfully installed all prerequisites, you're ready to move on to the Basic Robotics Concepts section where we'll explore fundamental principles of robotics and how they apply to humanoid systems.