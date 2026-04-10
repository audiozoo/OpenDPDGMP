# OpenDPDGMP — Claude Code Instructions

## Git & GitHub Workflow

This project uses Git for version control. **All work must be committed and pushed to GitHub regularly** so that progress is never lost and changes can be reverted at any time.

### Rules

- After completing any meaningful unit of work (new feature, bug fix, refactor, config change), commit it immediately.
- Write clean, descriptive commit messages in the imperative mood (e.g. "Add user authentication", "Fix off-by-one error in score calculation").
- Push to the `main` branch on GitHub after every commit (or logical batch of commits).
- Never let uncommitted changes accumulate — commit early, commit often.
- Before starting a new task, ensure the working tree is clean (`git status`).

### Commit Message Format

```
<short summary in imperative mood> (50 chars or less)

<optional body: explain what and why, not how>
```

Examples of good commit messages:
- `Initialize project structure`
- `Add CLAUDE.md with project conventions`
- `Fix calculation error in damage multiplier`

### GitHub Repository

- Remote: `https://github.com/audiozoo/OpenDPDGMP`
- Branch: `main`

### Workflow Steps

1. Make changes to files.
2. Stage relevant files: `git add <files>`
3. Commit with a clean message: `git commit -m "..."`
4. Push: `git push origin main`

This ensures there is always a saved, recoverable state of the project on GitHub.
