# API Contracts: Physical AI & Humanoid Robotics Book Backend

**Feature**: 1-physical-ai-book | **Date**: 2025-12-10

## Authentication Contracts

### POST /api/auth/signup
- **Purpose**: Register a new user
- **Request**:
  - Headers: `Content-Type: application/json`
  - Body:
    ```json
    {
      "email": "string (required, valid email)",
      "name": "string (required, 1-100 chars)",
      "software_background": "string (optional)",
      "hardware_background": "string (optional)"
    }
    ```
- **Response**:
  - 201 Created:
    ```json
    {
      "user": {
        "id": "string",
        "email": "string",
        "name": "string",
        "software_background": "string",
        "hardware_background": "string",
        "created_at": "timestamp"
      },
      "token": "string"
    }
    ```
  - 400 Bad Request: Invalid input data
  - 409 Conflict: User already exists

### POST /api/auth/signin
- **Purpose**: Authenticate existing user
- **Request**:
  - Headers: `Content-Type: application/json`
  - Body:
    ```json
    {
      "email": "string (required, valid email)",
      "password": "string (required)"
    }
    ```
- **Response**:
  - 200 OK:
    ```json
    {
      "user": {
        "id": "string",
        "email": "string",
        "name": "string",
        "software_background": "string",
        "hardware_background": "string",
        "created_at": "timestamp"
      },
      "token": "string"
    }
    ```
  - 401 Unauthorized: Invalid credentials

### GET /api/auth/me
- **Purpose**: Get current user profile
- **Request**:
  - Headers: `Authorization: Bearer {token}`
- **Response**:
  - 200 OK:
    ```json
    {
      "user": {
        "id": "string",
        "email": "string",
        "name": "string",
        "software_background": "string",
        "hardware_background": "string",
        "created_at": "timestamp",
        "updated_at": "timestamp"
      }
    }
    ```
  - 401 Unauthorized: Invalid token

## User Settings Contracts

### GET /api/user/settings
- **Purpose**: Get user settings
- **Request**:
  - Headers: `Authorization: Bearer {token}`
- **Response**:
  - 200 OK:
    ```json
    {
      "settings": {
        "language_preference": "string",
        "translate_to_urdu": "boolean",
        "personalization_enabled": "boolean"
      }
    }
    ```
  - 401 Unauthorized: Invalid token

### PUT /api/user/settings
- **Purpose**: Update user settings
- **Request**:
  - Headers: `Authorization: Bearer {token}`, `Content-Type: application/json`
  - Body:
    ```json
    {
      "language_preference": "string (optional)",
      "translate_to_urdu": "boolean (optional)",
      "personalization_enabled": "boolean (optional)"
    }
    ```
- **Response**:
  - 200 OK:
    ```json
    {
      "settings": {
        "language_preference": "string",
        "translate_to_urdu": "boolean",
        "personalization_enabled": "boolean"
      }
    }
    ```
  - 401 Unauthorized: Invalid token

## Content Personalization Contracts

### GET /api/content/personalize/{moduleId}
- **Purpose**: Get personalized content for a specific module based on user background
- **Request**:
  - Headers: `Authorization: Bearer {token}`
  - Path params: `moduleId` - ID of the book module
- **Response**:
  - 200 OK:
    ```json
    {
      "module_id": "string",
      "personalized_content": [
        {
          "content_id": "string",
          "original_content": "string",
          "personalized_content": "string",
          "difficulty_level": "string"
        }
      ]
    }
    ```
  - 401 Unauthorized: Invalid token
  - 404 Not Found: Module not found

### POST /api/translate/to-urdu
- **Purpose**: Translate content to Urdu using Google Translate API
- **Request**:
  - Headers: `Authorization: Bearer {token}`, `Content-Type: application/json`
  - Body:
    ```json
    {
      "content": "string (required) - Content to translate",
      "module_id": "string (optional) - Module context"
    }
    ```
- **Response**:
  - 200 OK:
    ```json
    {
      "original_content": "string",
      "translated_content": "string",
      "module_id": "string"
    }
    ```
  - 401 Unauthorized: Invalid token
  - 429 Too Many Requests: Rate limit exceeded

## User Progress Contracts

### POST /api/user/progress
- **Purpose**: Update user progress for a module
- **Request**:
  - Headers: `Authorization: Bearer {token}`, `Content-Type: application/json`
  - Body:
    ```json
    {
      "module_id": "string (required)",
      "progress_percentage": "integer (0-100, required)",
      "time_spent": "integer (optional) - seconds"
    }
    ```
- **Response**:
  - 200 OK:
    ```json
    {
      "progress": {
        "id": "string",
        "user_id": "string",
        "module_id": "string",
        "progress_percentage": "integer",
        "time_spent": "integer",
        "started_at": "timestamp",
        "last_accessed_at": "timestamp"
      }
    }
    ```
  - 401 Unauthorized: Invalid token

### GET /api/user/progress/{moduleId}
- **Purpose**: Get user progress for a specific module
- **Request**:
  - Headers: `Authorization: Bearer {token}`
  - Path params: `moduleId` - ID of the book module
- **Response**:
  - 200 OK:
    ```json
    {
      "progress": {
        "id": "string",
        "user_id": "string",
        "module_id": "string",
        "progress_percentage": "integer",
        "time_spent": "integer",
        "started_at": "timestamp",
        "completed_at": "timestamp",
        "last_accessed_at": "timestamp"
      }
    }
    ```
  - 401 Unauthorized: Invalid token
  - 404 Not Found: Progress record not found

## Book Content Contracts

### GET /api/book/modules
- **Purpose**: Get all book modules with metadata
- **Request**:
  - Headers: `Authorization: Bearer {token}` (optional)
- **Response**:
  - 200 OK:
    ```json
    {
      "modules": [
        {
          "id": "string",
          "title": "string",
          "slug": "string",
          "description": "string",
          "order": "integer",
          "prerequisites": ["string"],
          "learning_objectives": ["string"]
        }
      ]
    }
    ```

### GET /api/book/modules/{moduleId}/content
- **Purpose**: Get content for a specific module
- **Request**:
  - Headers: `Authorization: Bearer {token}` (optional)
  - Path params: `moduleId` - ID of the book module
- **Response**:
  - 200 OK:
    ```json
    {
      "module": {
        "id": "string",
        "title": "string",
        "slug": "string"
      },
      "content": [
        {
          "id": "string",
          "title": "string",
          "content_type": "string",
          "content": "string",
          "order": "integer"
        }
      ]
    }
    ```
  - 404 Not Found: Module not found