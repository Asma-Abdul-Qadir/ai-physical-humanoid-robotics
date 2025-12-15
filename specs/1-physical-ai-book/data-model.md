# Data Model: Physical AI & Humanoid Robotics Book

**Feature**: 1-physical-ai-book | **Date**: 2025-12-10

## Core Entities

### User
- **Fields**:
  - id (string, primary key)
  - email (string, unique, required)
  - name (string, required)
  - software_background (string, optional) - Programming experience level
  - hardware_background (string, optional) - Robotics/hardware experience level
  - created_at (timestamp, required)
  - updated_at (timestamp, required)
  - preferences (object, optional) - Personalization settings
- **Relationships**:
  - One-to-many with UserProgress
  - One-to-many with UserSettings
- **Validation**:
  - Email must be valid email format
  - Name must be 1-100 characters
  - Background fields must be from predefined options

### BookModule
- **Fields**:
  - id (string, primary key)
  - title (string, required)
  - slug (string, unique, required) - URL-friendly identifier
  - description (string, required)
  - order (integer, required) - Position in the book sequence
  - prerequisites (array of strings, optional) - Skills/knowledge needed
  - learning_objectives (array of strings, required)
  - content_path (string, required) - Path to the module content
  - created_at (timestamp, required)
  - updated_at (timestamp, required)
- **Relationships**:
  - One-to-many with ModuleContent
  - One-to-many with UserProgress
- **Validation**:
  - Title must be 1-200 characters
  - Slug must be URL-friendly format
  - Order must be positive integer

### ModuleContent
- **Fields**:
  - id (string, primary key)
  - module_id (string, foreign key to BookModule)
  - content_type (string, required) - 'text', 'code', 'image', 'exercise', 'example'
  - content (string, required) - The actual content in Markdown format
  - title (string, required)
  - order (integer, required) - Position within the module
  - language (string, optional) - For code examples
  - created_at (timestamp, required)
  - updated_at (timestamp, required)
- **Relationships**:
  - Many-to-one with BookModule
  - One-to-many with ContentPersonalization
- **Validation**:
  - Content type must be from predefined enum
  - Order must be positive integer
  - Content must not be empty

### UserProgress
- **Fields**:
  - id (string, primary key)
  - user_id (string, foreign key to User)
  - module_id (string, foreign key to BookModule)
  - progress_percentage (integer, required) - 0-100
  - started_at (timestamp, required)
  - completed_at (timestamp, optional)
  - time_spent (integer, optional) - Time spent in seconds
  - last_accessed_at (timestamp, required)
- **Relationships**:
  - Many-to-one with User
  - Many-to-one with BookModule
- **Validation**:
  - Progress percentage must be 0-100
  - Completed_at must be after started_at if present

### UserSettings
- **Fields**:
  - id (string, primary key)
  - user_id (string, foreign key to User)
  - language_preference (string, optional) - Default: 'en'
  - translate_to_urdu (boolean, optional) - Default: false
  - personalization_enabled (boolean, optional) - Default: true
  - created_at (timestamp, required)
  - updated_at (timestamp, required)
- **Relationships**:
  - Many-to-one with User
- **Validation**:
  - Language must be from supported languages list

### ContentPersonalization
- **Fields**:
  - id (string, primary key)
  - content_id (string, foreign key to ModuleContent)
  - user_background (string, required) - Software/hardware background
  - personalized_content (string, required) - Personalized version of content
  - difficulty_level (string, required) - 'beginner', 'intermediate', 'advanced'
  - created_at (timestamp, required)
  - updated_at (timestamp, required)
- **Relationships**:
  - Many-to-one with ModuleContent
- **Validation**:
  - Difficulty level must be from predefined enum
  - Personalized content must not be empty

### Exercise
- **Fields**:
  - id (string, primary key)
  - module_id (string, foreign key to BookModule)
  - title (string, required)
  - description (string, required)
  - instructions (string, required)
  - expected_output (string, optional)
  - difficulty (string, required) - 'easy', 'medium', 'hard'
  - estimated_time (integer, required) - Time in minutes
  - created_at (timestamp, required)
  - updated_at (timestamp, required)
- **Relationships**:
  - Many-to-one with BookModule
  - One-to-many with ExerciseSubmission
- **Validation**:
  - Difficulty must be from predefined enum
  - Estimated time must be positive integer

### ExerciseSubmission
- **Fields**:
  - id (string, primary key)
  - exercise_id (string, foreign key to Exercise)
  - user_id (string, foreign key to User)
  - submission_content (string, required)
  - submitted_at (timestamp, required)
  - is_correct (boolean, optional)
  - feedback (string, optional)
- **Relationships**:
  - Many-to-one with Exercise
  - Many-to-one with User
- **Validation**:
  - Submission content must not be empty

## State Transitions

### User Registration Flow
1. User → Registered (email verification)
2. User → Profile Completed (background information added)

### Module Progression Flow
1. ModuleContent → Started (user begins reading)
2. ModuleContent → In Progress (user continues engagement)
3. ModuleContent → Completed (user finishes content)

### Exercise Submission Flow
1. Exercise → Started (user begins exercise)
2. Exercise → Submitted (user submits solution)
3. Exercise → Evaluated (feedback provided, correctness determined)

## Indexes
- User.email (unique)
- BookModule.slug (unique)
- UserProgress.user_id + UserProgress.module_id (composite unique)
- UserSettings.user_id (unique)