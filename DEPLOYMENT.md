# Deployment Guide

This document explains how to deploy the Physical AI & Humanoid Robotics book to both GitHub Pages and Vercel.

## GitHub Pages Deployment

The project is already configured for GitHub Pages deployment using GitHub Actions.

### Automatic Deployment
- When you push to the `main` or `1-physical-ai-book` branches, the GitHub Action workflow automatically builds and deploys the site to GitHub Pages.
- The workflow is defined in `.github/workflows/deploy.yml`.

### Manual Deployment
To manually deploy to GitHub Pages:

```bash
npm run build:github
```

Then follow Docusaurus' standard GitHub Pages deployment process.

## Vercel Deployment

The project is configured for Vercel deployment.

### Setup Vercel Project
1. Sign up/Log in to [Vercel](https://vercel.com)
2. Import your GitHub repository
3. Vercel will automatically use the build command from vercel.json: `npm run vercel-build`
4. The output directory is set to: `build`
5. Vercel will automatically use the vercel.json configuration for clean URLs

### Environment Variables
- No special environment variables are needed for Vercel deployment
- The site will automatically detect the Vercel environment and use the correct URL/basePath

## Configuration Details

The `docusaurus.config.js` file contains conditional logic to handle different deployment targets:

- When `DEPLOYMENT_TARGET=vercel`:
  - URL: `https://ai-physical-humanoid-robotics.vercel.app`
  - Base URL: `/`

- When deploying to GitHub Pages (default or `DEPLOYMENT_TARGET=github`):
  - URL: `https://asma-abdul-qadir.github.io`
  - Base URL: `/physical-ai-humanoid-robotics-book/`

## Build Scripts

Two custom build scripts are available:

- `npm run build:github` - Builds the site for GitHub Pages deployment
- `npm run build:vercel` - Builds the site for Vercel deployment

## Verifying Your Deployment

After deployment, verify that:
- All pages load correctly
- Navigation works as expected
- Images and assets are loading properly
- Search functionality works
- Links to GitHub repository are correct