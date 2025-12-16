// @ts-check
// `@type` JSDoc annotations allow IDEs and type checkers to understand what
// is expected in this project and check for errors.

const lightCodeTheme = require('prism-react-renderer').themes.github;
const darkCodeTheme = require('prism-react-renderer').themes.dracula;

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'A comprehensive guide to building intelligent humanoid robots',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: process.env.DEPLOYMENT_TARGET === 'vercel' || process.env.VERCEL === '1'
    ? 'https://ai-physical-humanoid-robotics.vercel.app'
    : 'https://asma-abdul-qadir.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub Pages: https://<USERNAME>.github.io/<REPO>/
  baseUrl: process.env.DEPLOYMENT_TARGET === 'vercel' || process.env.VERCEL === '1'
    ? '/'
    : '/physical-ai-humanoid-robotics-book/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'Asma-Abdul-Qadir', // Usually your GitHub org/user name.
  projectName: 'physical-ai-humanoid-robotics-book', // Usually your repo name.

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'ur'], // Adding Urdu for translation support
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        blog: false, // Disable blog if not needed
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      colorMode: {
        defaultMode: 'dark',
        disableSwitch: false,
        respectPrefersColorScheme: true,
      },
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: 'Physical AI & Humanoid Robotics',
        logo: {
          alt: 'Physical AI Logo',
          src: 'logo.png',
        },
        items: [
          {
            to: '/',
            label: 'Home',
            position: 'left'
          },
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Modules',
          },
          {
            to: '/docs/modules/foundations/intro',
            label: 'Foundations',
            position: 'left',
          },
          {
            to: '/docs/modules/ros2-urdf/intro',
            label: 'ROS 2 + URDF',
            position: 'left',
          },
          {
            to: '/docs/modules/digital-twin/gazebo',
            label: 'Digital Twin',
            position: 'left',
          },
          {
            to: '/docs/modules/ai-robot-brain/nav2-navigation',
            label: 'AI Robot Brain',
            position: 'left',
          },
          {
            href: 'https://github.com/Asma-Abdul-Qadir/ai-physical-humanoid-robotics',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Modules',
            items: [
              {
                label: 'Foundations',
                to: '/docs/modules/foundations/intro',
              },
              {
                label: 'ROS 2 + URDF',
                to: '/docs/modules/ros2-urdf/intro',
              },
              {
                label: 'Digital Twin',
                to: '/docs/modules/digital-twin/gazebo',
              },
              {
                label: 'AI Robot Brain',
                to: '/docs/modules/ai-robot-brain/nav2-navigation',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Stack Overflow',
                href: 'https://stackoverflow.com/questions/tagged/docusaurus',
              },
              {
                label: 'Discord',
                href: 'https://discordapp.com/invite/docusaurus',
              },
              {
                label: 'Twitter',
                href: 'https://twitter.com/docusaurus',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/facebook/docusaurus',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Book. Built with Docusaurus.`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
    }),
};

module.exports = config;
