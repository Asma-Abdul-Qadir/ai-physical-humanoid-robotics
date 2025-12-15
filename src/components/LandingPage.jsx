import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './LandingPage.module.css';

export default function LandingPage() {
  const { siteConfig } = useDocusaurusContext();

  // Predefined modules (NO sidebar runtime import)
  const modules = [
    { title: 'Foundations', href: '/docs/modules/foundations/intro' },
    { title: 'ROS 2 + URDF', href: '/docs/modules/ros2-urdf/intro' },
    { title: 'Digital Twin', href: '/docs/modules/digital-twin/gazebo' },
    { title: 'AI Robot Brain', href: '/docs/modules/ai-robot-brain/nav2-navigation' },
  ];

  return (
    <div className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className={clsx('container', styles.container)}>

        {/* ================= HERO SECTION ================= */}
        <section className={styles.heroSection}>
          <div className={styles.heroContent}>
            <div className={styles.textContent}>
              <h1 className={clsx('hero__title', styles.title)}>
                {siteConfig.title}
              </h1>

              <p className={clsx('hero__subtitle', styles.subtitle)}>
                {siteConfig.tagline}
              </p>

              <div className={styles.buttons}>
                <Link
                   to="/docs/modules/foundations/intro"
                  className={clsx(
                    'button button--secondary button--lg',
                    styles.getStartedButton
                  )}
                >
                  Explore Book
                </Link>
              </div>
            </div>

            <div className={styles.robotImage}>
              <img
                src="ai.webp"
                alt="Humanoid Robot"
                className={styles.robotImageContent}
              />
            </div>
          </div>
        </section>

        {/* ================= MODULES SECTION ================= */}
        <section className={styles.modulesSection}>
          <h2 className={styles.modulesTitle}>Core Modules</h2>

          <div className={styles.modulesGrid}>
            {modules.map((module, index) => (
              <Link
                key={index}
                to={module.href}
                className={styles.moduleCard}
              >
                <h3 className={styles.moduleTitle}>{module.title}</h3>
              </Link>
            ))}
          </div>
        </section>

      </div>
    </div>
  );
}
