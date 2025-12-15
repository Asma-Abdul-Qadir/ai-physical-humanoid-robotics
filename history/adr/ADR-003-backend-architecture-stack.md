# ADR-003: Backend Architecture and API Framework Stack

> **Scope**: Document decision clusters, not individual technology choices. Group related decisions that work together (e.g., "Frontend Stack" not separate ADRs for framework, styling, deployment).

- **Status:** Accepted
- **Date:** 2025-12-10
- **Feature:** Physical AI & Humanoid Robotics Book
- **Context:** Need to select a backend architecture to support optional features like user authentication, content personalization, and translation capabilities for the educational book platform. The solution should be lightweight, fast, and integrate well with the static frontend.

<!-- Significance checklist (ALL must be true to justify this ADR)
     1) Impact: Long-term consequence for architecture/platform/security?
     2) Alternatives: Multiple viable options considered with tradeoffs?
     3) Scope: Cross-cutting concern (not an isolated detail)?
     If any are false, prefer capturing as a PHR note instead of an ADR. -->

## Decision

Use Hono as the API framework with Better-Auth for authentication in an optional backend architecture for the Physical AI & Humanoid Robotics educational platform.

- API Framework: Hono (for lightweight, fast API routes)
- Authentication: Better-Auth (for secure, easy-to-implement auth)
- Database: Optional PostgreSQL for user profiles and preferences
- Middleware: Hono native middleware for request handling
- Deployment: Compatible with multiple platforms (Cloudflare Workers, Node.js, Bun)

## Consequences

### Positive

- Lightweight and fast API framework suitable for educational platform
- Better-Auth provides secure, well-tested authentication without reinventing security
- Hono's compatibility with multiple deployment targets provides flexibility
- Small bundle size and fast response times
- Good TypeScript support for type safety
- Minimal overhead for optional backend features
- Integration-friendly with the static Docusaurus frontend

### Negative

- Additional complexity compared to purely static site
- Need for database management and security considerations
- Hono is relatively newer framework with smaller community than Express
- Operational overhead for managing backend infrastructure
- Potential scaling considerations if user base grows significantly
- Additional security surface area requiring ongoing attention

## Alternatives Considered

Alternative Stack A: Express.js + Passport.js
- Why rejected: Heavier framework with larger bundle size, more complex setup than needed for this use case

Alternative Stack B: Next.js API Routes
- Why rejected: Would require migration from Docusaurus to Next.js, losing documentation-focused features

Alternative Stack C: Serverless functions (AWS Lambda, Vercel)
- Why rejected: Would fragment the architecture and potentially increase costs with usage

## References

- Feature Spec: ./specs/1-physical-ai-book/spec.md
- Implementation Plan: ./specs/1-physical-ai-book/plan.md
- Related ADRs:
- Evaluator Evidence: ./history/prompts/1-physical-ai-book/