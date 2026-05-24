// Commitlint configuration (auto-discovered by wagoid/commitlint-github-action).
//
// Extends the Conventional Commits preset but relaxes body-max-line-length:
// Dependabot's auto-generated bodies (changelog/compare URLs) and trailers
// like Co-authored-by routinely exceed 100 columns, and wrapping URLs is not
// useful. All other conventional rules (type-enum, subject-case, ...) stay on.
export default {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'body-max-line-length': [0, 'always'],
  },
};
