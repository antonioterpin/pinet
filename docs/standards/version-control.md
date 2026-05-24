# Version control discipline

## Commit strategy

- **Commit code often**, in small, logically coherent units.
- Do **not** commit unfinished or broken code.
- **Always follow the `.gitmessage` template** for commit messages (Conventional Commits style).
- Configure git once with:
  ```bash
  git config commit.template .gitmessage
  ```

## Remote repository

- **Never push code to the remote repository.** (Pushed centrally).
