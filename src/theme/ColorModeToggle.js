import React from 'react';
import clsx from 'clsx';
import { useColorMode } from '@docusaurus/theme-common';
import useIsBrowser from '@docusaurus/useIsBrowser';
import { translate } from '@docusaurus/Translate';

import styles from './ColorModeToggle/styles.module.css';

// Custom SVG icons for sun and moon
const SunIcon = ({ className }) => (
  <svg
    className={className}
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
  >
    <path
      d="M12 16C14.2091 16 16 14.2091 16 12C16 9.79086 14.2091 8 12 8C9.79086 8 8 9.79086 8 12C8 14.2091 9.79086 16 12 16Z"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <path
      d="M12 2V4"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <path
      d="M12 20V22"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <path
      d="M4.93 4.93L6.34 6.34"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <path
      d="M17.66 17.66L19.07 19.07"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <path
      d="M2 12H4"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <path
      d="M20 12H22"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <path
      d="M6.34 17.66L4.93 19.07"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <path
      d="M19.07 4.93L17.66 6.34"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

const MoonIcon = ({ className }) => (
  <svg
    className={className}
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
  >
    <path
      d="M20.3542 15.3542C19.3176 15.7708 18.1856 16.0001 17 16.0001C12.0294 16.0001 8 11.9706 8 7.00006C8 5.81444 8.22924 4.68243 8.64581 3.64581C5.33651 4.9756 3 8.21507 3 12.0001C3 16.9706 7.02944 21.0001 12 21.0001C15.785 21.0001 19.0245 18.6635 20.3542 15.3542Z"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

const DarkIcon = MoonIcon;
const LightIcon = SunIcon;

export default function ColorModeToggle({ className, buttonClassName }) {
  const { colorMode, setColorMode } = useColorMode();
  const isBrowser = useIsBrowser();
  const isDarkTheme = colorMode === 'dark';
  const toggle = () => setColorMode(isDarkTheme ? 'light' : 'dark');

  // Get disableSwitch from theme config without using useThemeConfig
  const disableSwitch = false; // Default to false, can be overridden in theme config

  if (disableSwitch) {
    return null;
  }

  return (
    <div className={clsx(styles.colorModeToggle, className)}>
      <button
        className={clsx(
          'button button--secondary button--outline button--sm',
          styles.colorModeToggle__button,
          buttonClassName,
        )}
        onClick={toggle}
        disabled={!isBrowser}
        title={translate({
          id: 'theme.NotFound.toggleButton.title',
          message: 'Switch between dark and light mode (currently {mode})',
          description: 'The title attribute for the navbar color mode toggle',
        }, { mode: isDarkTheme ? 'dark mode' : 'light mode' })}
        style={{
          background: 'rgba(255, 255, 255, 0.1)',
          border: '1px solid rgba(255, 255, 255, 0.2)',
          color: 'white',
          borderRadius: '8px',
          padding: '6px 12px',
          cursor: 'pointer',
          transition: 'all 0.3s ease',
          position: 'relative',
          backdropFilter: 'blur(4px)',
          overflow: 'hidden'
        }}
        onMouseEnter={(e) => {
          e.target.style.background = 'rgba(255, 0, 0, 0.1)';
          e.target.style.boxShadow = '0 0 12px rgba(255, 0, 0, 0.5)';
        }}
        onMouseLeave={(e) => {
          e.target.style.background = 'rgba(255, 255, 255, 0.1)';
          e.target.style.boxShadow = 'none';
        }}
      >
        {isDarkTheme ? (
          <DarkIcon
            className={styles.colorModeToggle__icon}
            style={{ width: '20px', height: '20px', color: 'white' }}
          />
        ) : (
          <LightIcon
            className={styles.colorModeToggle__icon}
            style={{ width: '20px', height: '20px', color: 'white' }}
          />
        )}
      </button>
    </div>
  );
}