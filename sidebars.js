// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Modules',
      items: [
        {
          type: 'category',
          label: 'Foundations',
          items: ['modules/foundations/intro', 'modules/foundations/hardware-requirements', 'modules/foundations/software-prerequisites', 'modules/foundations/basic-robotics-concepts', 'modules/foundations/safety-ethics', 'modules/foundations/exercises'],
        },
        {
          type: 'category',
          label: 'ROS 2 + URDF',
          items: ['modules/ros2-urdf/intro', 'modules/ros2-urdf/nodes-topics-services', 'modules/ros2-urdf/urdf-modeling', 'modules/ros2-urdf/launch-files', 'modules/ros2-urdf/actions-parameters', 'modules/ros2-urdf/exercises'],
        },
        {
          type: 'category',
          label: 'Digital Twin',
          items: ['modules/digital-twin/gazebo', 'modules/digital-twin/unity', 'modules/digital-twin/isaac-sim', 'modules/digital-twin/simulation-configuration', 'modules/digital-twin/exercises'],
        },
        {
          type: 'category',
          label: 'AI Robot Brain',
          items: ['modules/ai-robot-brain/nav2-navigation', 'modules/ai-robot-brain/vision-language-action-pipeline'],
        },
        {
          type: 'category',
          label: 'Capstone Project',
          items: [
            'modules/capstone/overview',
            'modules/capstone/planning',
            'modules/capstone/implementation',
            'modules/capstone/testing',
            'modules/capstone/assessment',
            'modules/capstone/integration_tests',
            'modules/capstone/verification'
          ],
        },
      ],
    },
  ],
};

module.exports = sidebars;