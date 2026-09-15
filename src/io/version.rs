// SPDX-License-Identifier: MIT OR Apache-2.0

//! Opt-in `/version` compatibility policy for HDF5 reads.
//!
//! Default [`super::read`] stays permissive — missing or arbitrary version
//! strings are stored verbatim, matching Python `nir.read`. Callers that need
//! a fail-closed envelope check set [`VersionPolicy`] on [`super::ReadOptions`].

use crate::error::{NirError, Result};
use std::fmt;

/// How [`super::read_with`] treats the root `/version` dataset.
///
/// The default is [`Self::Permissive`]. Majors accepted by
/// [`Self::CompatibleMajor`] are supplied by the caller; this crate does not
/// keep a hidden compatibility matrix. Paper fixtures vendored under
/// `tests/fixtures/` embed `0.1.1` / `0.2.0`, and [`super::DEFAULT_NIR_VERSION`]
/// is `1.0.8` — a typical importer therefore passes `[0, 1]`.
///
/// ```
/// use nir_rs::io::{ReadOptions, VersionPolicy};
///
/// let tool = ReadOptions::default();
/// assert_eq!(tool.version_policy, VersionPolicy::Permissive);
///
/// let importer = ReadOptions::default()
///     .with_version_policy(VersionPolicy::compatible_major([0, 1]));
/// assert!(matches!(
///     importer.version_policy,
///     VersionPolicy::CompatibleMajor { .. }
/// ));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum VersionPolicy {
    /// Accept a missing `/version` or any present string and store it verbatim.
    ///
    /// Matches Python `nir.read` and the default [`super::read`] entry point.
    #[default]
    Permissive,
    /// Reject a missing `/version`. Any present string is stored verbatim
    /// without SemVer parsing.
    RequirePresent,
    /// Parse `/version` as `MAJOR.MINOR.PATCH` with optional SemVer prerelease
    /// (`-…`) and build (`+…`) suffixes, and accept only the listed majors.
    ///
    /// Construct with [`Self::compatible_major`] so the allow-list is explicit.
    /// A missing `/version` is a policy failure.
    ///
    /// # Parsing
    ///
    /// The core must be exactly three decimal components. `1.0`, `v1.0.0`, and
    /// `01.0.0` are malformed. Numeric components must not have leading zeros
    /// (`0` itself is allowed).
    ///
    /// Prerelease and build suffixes are **parsed, then ignored for the major
    /// check**: `1.0.0-rc.1+exp.sha` has major `1` and is accepted when `1` is
    /// listed. They are not stripped from [`crate::NirGraph::version`] — the
    /// original wire string is stored. An empty suffix (`1.0.0-` / `1.0.0+`)
    /// is malformed. Identifiers may contain ASCII alphanumerics and `-`,
    /// separated by `.`.
    CompatibleMajor {
        /// Accepted major numbers, for example `[0, 1]`.
        majors: Vec<u64>,
    },
}

impl VersionPolicy {
    /// Accept `/version` strings whose SemVer major is one of `majors`.
    ///
    /// Duplicates are dropped and the list is sorted so [`Display`] is stable.
    /// An empty list rejects every well-formed version (fail-closed).
    #[must_use]
    pub fn compatible_major(majors: impl IntoIterator<Item = u64>) -> Self {
        let mut majors: Vec<u64> = majors.into_iter().collect();
        majors.sort_unstable();
        majors.dedup();
        Self::CompatibleMajor { majors }
    }
}

impl fmt::Display for VersionPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Permissive => f.write_str("permissive"),
            Self::RequirePresent => f.write_str("require-present"),
            Self::CompatibleMajor { majors } => {
                write!(f, "compatible-major majors=[")?;
                for (i, major) in majors.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{major}")?;
                }
                write!(f, "]")
            }
        }
    }
}

/// Core `MAJOR.MINOR.PATCH` extracted from a SemVer-compatible `/version`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ParsedVersion {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
}

/// Parse the NIR `/version` subset used by [`VersionPolicy::CompatibleMajor`].
pub(super) fn parse_nir_version(raw: &str) -> Option<ParsedVersion> {
    let (core_and_pre, build) = match raw.split_once('+') {
        Some((left, right)) => (left, Some(right)),
        None => (raw, None),
    };
    if let Some(build) = build
        && !is_valid_suffix(build)
    {
        return None;
    }
    let (core, pre) = match core_and_pre.split_once('-') {
        Some((left, right)) => (left, Some(right)),
        None => (core_and_pre, None),
    };
    if let Some(pre) = pre
        && !is_valid_suffix(pre)
    {
        return None;
    }
    let mut parts = core.split('.');
    let major = parse_component(parts.next()?)?;
    let minor = parse_component(parts.next()?)?;
    let patch = parse_component(parts.next()?)?;
    if parts.next().is_some() {
        return None;
    }
    Some(ParsedVersion {
        major,
        minor,
        patch,
    })
}

fn parse_component(s: &str) -> Option<u64> {
    if s.is_empty() {
        return None;
    }
    if s.len() > 1 && s.starts_with('0') {
        return None;
    }
    if !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

fn is_valid_suffix(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    s.split('.').all(|ident| {
        !ident.is_empty()
            && ident
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-')
    })
}

/// Apply `policy` to an already-decoded `/version` string (or its absence).
///
/// Shared by the full graph reader and [`super::read_version_with`] so the two
/// cannot disagree about which strings a policy accepts. Dataset-shape errors
/// (missing vs group vs string) are resolved before this is called.
pub(super) fn enforce_version_policy(observed: Option<&str>, policy: &VersionPolicy) -> Result<()> {
    match policy {
        VersionPolicy::Permissive => Ok(()),
        VersionPolicy::RequirePresent => {
            if observed.is_none() {
                Err(incompatible(None, policy))
            } else {
                Ok(())
            }
        }
        VersionPolicy::CompatibleMajor { majors } => {
            let Some(raw) = observed else {
                return Err(incompatible(None, policy));
            };
            match parse_nir_version(raw) {
                Some(parsed) if majors.contains(&parsed.major) => Ok(()),
                _ => Err(incompatible(Some(raw.to_owned()), policy)),
            }
        }
    }
}

fn incompatible(observed: Option<String>, policy: &VersionPolicy) -> NirError {
    NirError::IncompatibleVersion {
        observed,
        policy: policy.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compatible_major_canonicalizes_the_allow_list() {
        let policy = VersionPolicy::compatible_major([1, 0, 1, 0]);
        assert_eq!(
            policy,
            VersionPolicy::CompatibleMajor { majors: vec![0, 1] }
        );
        assert_eq!(policy.to_string(), "compatible-major majors=[0, 1]");
    }

    #[test]
    fn parse_accepts_core_and_suffixes() {
        assert_eq!(
            parse_nir_version("0.1.1"),
            Some(ParsedVersion {
                major: 0,
                minor: 1,
                patch: 1
            })
        );
        assert_eq!(
            parse_nir_version("1.0.8"),
            Some(ParsedVersion {
                major: 1,
                minor: 0,
                patch: 8
            })
        );
        let pre = parse_nir_version("1.0.0-rc.1").unwrap();
        assert_eq!(pre.major, 1);
        let build = parse_nir_version("1.0.0+exp.sha.5114f85").unwrap();
        assert_eq!(build.major, 1);
        let both = parse_nir_version("1.2.3-alpha.1+build.9").unwrap();
        assert_eq!(
            both,
            ParsedVersion {
                major: 1,
                minor: 2,
                patch: 3
            }
        );
        // Hyphens inside identifiers are allowed; the first `-` / `+` split
        // still isolates the core.
        assert!(parse_nir_version("0.2.0-beta-1").is_some());
    }

    #[test]
    fn parse_rejects_malformed_strings() {
        for raw in [
            "",
            "latest",
            "1",
            "1.0",
            "v1.0.0",
            "01.0.0",
            "1.0.0.0",
            "1.0.0-",
            "1.0.0+",
            "1.0.0-+build",
            "1.0.0-rc.",
            "-1.0.0",
        ] {
            assert!(
                parse_nir_version(raw).is_none(),
                "{raw:?} should be malformed"
            );
        }
    }

    #[test]
    fn permissive_never_rejects() {
        let policy = VersionPolicy::Permissive;
        enforce_version_policy(None, &policy).unwrap();
        enforce_version_policy(Some("not-a-semver"), &policy).unwrap();
    }

    #[test]
    fn require_present_rejects_only_absence() {
        let policy = VersionPolicy::RequirePresent;
        match enforce_version_policy(None, &policy).unwrap_err() {
            NirError::IncompatibleVersion { observed, policy } => {
                assert_eq!(observed, None);
                assert_eq!(policy, "require-present");
            }
            other => panic!("unexpected {other:?}"),
        }
        enforce_version_policy(Some("not-a-semver"), &policy).unwrap();
    }

    #[test]
    fn compatible_major_covers_missing_malformed_and_majors() {
        let policy = VersionPolicy::compatible_major([0, 1]);
        assert!(enforce_version_policy(Some("0.1.1"), &policy).is_ok());
        assert!(enforce_version_policy(Some("0.2.0"), &policy).is_ok());
        assert!(enforce_version_policy(Some("1.0.8"), &policy).is_ok());
        assert!(enforce_version_policy(Some("1.0.0-rc.1"), &policy).is_ok());
        assert!(enforce_version_policy(Some("1.0.0+build"), &policy).is_ok());

        for (raw, expect_observed) in [
            (None, None),
            (Some("2.0.0"), Some("2.0.0")),
            (Some("not-a-semver"), Some("not-a-semver")),
            (Some("v1.0.0"), Some("v1.0.0")),
        ] {
            match enforce_version_policy(raw, &policy).unwrap_err() {
                NirError::IncompatibleVersion { observed, policy } => {
                    assert_eq!(observed.as_deref(), expect_observed);
                    assert_eq!(policy, "compatible-major majors=[0, 1]");
                }
                other => panic!("unexpected {other:?} for {raw:?}"),
            }
        }
    }

    #[test]
    fn empty_major_list_is_fail_closed() {
        let policy = VersionPolicy::compatible_major([]);
        match enforce_version_policy(Some("1.0.8"), &policy).unwrap_err() {
            NirError::IncompatibleVersion { observed, policy } => {
                assert_eq!(observed.as_deref(), Some("1.0.8"));
                assert_eq!(policy, "compatible-major majors=[]");
            }
            other => panic!("unexpected {other:?}"),
        }
    }
}
