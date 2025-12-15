# Quickstart Guide: Physical AI & Humanoid Robotics Book

**Feature**: 1-physical-ai-book | **Date**: 2025-12-10

## Getting Started

This guide provides a quick overview of how to set up and start working with the Physical AI & Humanoid Robotics Book project.

## Prerequisites

### System Requirements
- **OS**: Ubuntu 22.04 LTS (primary support), Windows 10/11 or macOS with WSL2
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 20GB free space
- **Node.js**: v18.x or higher
- **Git**: Latest version

### Development Environment Setup

1. **Install ROS 2 Humble Hawksbill** (for robotics examples):
   ```bash
   # Follow official ROS 2 Humble installation guide for Ubuntu 22.04
   # http://docs.ros.org/en/humble/Installation.html
   ```

2. **Install Node.js and npm**:
   ```bash
   # Using nvm (recommended)
   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
   nvm install 18
   nvm use 18
   ```

3. **Clone the repository**:
   ```bash
   git clone https://github.com/[your-org]/physical-ai-humanoid-robotics-book.git
   cd physical-ai-humanoid-robotics-book
   ```

4. **Install project dependencies**:
   ```bash
   npm install
   ```

## Local Development

### Running the Docusaurus Book Locally
```bash
# Install Docusaurus CLI if not already installed
npm install -g @docusaurus/cli

# Start the development server
npm run start
```

The book will be available at `http://localhost:3000`

### Building the Book
```bash
# Build static files
npm run build

# Serve the built site locally
npm run serve
```

## Adding New Content

### Creating a New Book Module
1. Create a new directory under `docs/modules/`
2. Add your content in Markdown format
3. Update the sidebar configuration in `sidebars.js`

### Content Structure
```
docs/modules/[module-name]/
├── intro.md
├── concepts.md
├── examples.md
├── exercises.md
└── summary.md
```

## Backend Development (Optional Features)

### Running the Backend Server
```bash
# Navigate to backend directory
cd backend

# Install dependencies
npm install

# Start development server
npm run dev
```

### Environment Variables
Create a `.env` file in the backend directory:
```
DATABASE_URL="your-database-connection-string"
AUTH_SECRET="your-auth-secret"
GOOGLE_TRANSLATE_API_KEY="your-google-translate-api-key"
```

## Testing

### Running Tests
```bash
# Run all tests
npm test

# Run E2E tests with Playwright
npm run test:e2e

# Run accessibility tests
npm run test:accessibility
```

### Docker Testing for ROS2/Isaac Examples
```bash
# Build and run ROS2 examples in Docker
docker-compose -f docker/ros2/docker-compose.yml up --build

# Run Isaac examples in Docker
docker-compose -f docker/isaac/docker-compose.yml up --build
```

## Deployment

### GitHub Pages Deployment
The site is automatically deployed via GitHub Actions when changes are pushed to the main branch.

To manually trigger a deployment:
1. Ensure all changes are committed and pushed
2. The GitHub Actions workflow in `.github/workflows/deploy.yml` will handle the build and deployment

## Key Technologies

- **Docusaurus**: Static site generation for the book
- **React**: Frontend components and interactivity
- **ROS 2 Humble Hawksbill**: Robotics framework
- **Gazebo Garden**: Physics simulation
- **Hono**: Backend web framework (optional)
- **Better-Auth**: Authentication system (optional)

## Troubleshooting

### Common Issues
1. **Docusaurus build fails**: Ensure Node.js version is 18.x or higher
2. **ROS2 examples don't work**: Verify ROS2 Humble is properly installed and sourced
3. **Links broken in local dev**: Use absolute paths for internal links

### Getting Help
- Check the detailed specification in `specs/1-physical-ai-book/spec.md`
- Review the implementation plan in `specs/1-physical-ai-book/plan.md`
- Consult the research findings in `specs/1-physical-ai-book/research.md`