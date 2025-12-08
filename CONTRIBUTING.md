# Contributing Guide

## Branching Strategy
- `main` — stable, production-ready
- `dev` — integration branch
- Feature branches:
  - `feature/rag-ritwik`
  - `feature/prompts-akshaya`
  - `feature/ui-akshaya`
  - `feature/eval-ritwik`

## Workflow
1. Pull `dev`
2. Create feature branch
3. Commit frequently
4. Open PR → review by other member
5. Merge into `dev`

## Code Style
- Use type hints
- Add docstrings
- Avoid circular imports
- Keep modules single-responsibility
- No secrets in repo (`.env.example` used instead)

## Testing
- All new features must have tests where possible.
