# Coverage integrations

The [`Coverage` workflow](workflows/coverage.yml) creates one `lcov.info`
report from the Rust 1.98.1 all-feature test suite. It reports the same file to
Codecov and Codacy. The workflow is informational while a baseline is being
established; it does not replace the required CI, package, or semver jobs.

## Codecov

1. An organization administrator installs or authorizes the [Codecov GitHub
   App](https://github.com/apps/codecov) for `Limen-Neural/nir-rs`.
2. In Codecov, enable GitHub OIDC uploads for this repository. The workflow uses
   `id-token: write` and `use_oidc: true`, so it does **not** need a
   `CODECOV_TOKEN` repository secret.
3. Confirm a `rust`-flagged upload appears for a `Coverage` workflow run. Review
   the first baseline before enabling Codecov required status checks or numeric
   coverage thresholds.

## Codacy

1. An organization administrator connects `Limen-Neural/nir-rs` in
   [Codacy](https://app.codacy.com/).
2. Create a **project API token** in Codacy and store it as the repository
   Actions secret `CODACY_PROJECT_TOKEN`. Do not add it to a workflow, file,
   issue, or pull-request comment.
3. The next `Coverage` workflow sends `lcov.info` to Codacy. Until that secret
   exists, the workflow prints a notice and safely skips only the Codacy upload.
4. Review the first baseline before enabling Codacy coverage or quality status
   checks as required GitHub checks.

Coverage uploads are intentionally advisory until both providers have an
accepted baseline. The required test, Clippy, rustdoc, package, and semver gates
remain authoritative throughout setup.
