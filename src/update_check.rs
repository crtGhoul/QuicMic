//! Best-effort startup check for a newer GitHub release.
//!
//! Runs once at startup on a background task, never blocks, and is **silent on any
//! failure** (offline, DNS/TLS error, an unknown repo, a malformed response). It
//! only ever reports a *strictly newer* version, so it can never produce a false
//! "update available". Opt out with `--no-update-check` / `QUICMIC_NO_UPDATE_CHECK`.
//!
//! No HTTP-client crate is pulled in: it reuses the crate's existing TLS stack
//! (`rustls` + the process-default `ring` provider + `tokio-rustls`) and the OS
//! trust store (`rustls-native-certs`) to validate GitHub's public certificate.
//! Rather than the REST API + JSON, it hits the `releases/latest` endpoint, which
//! 302-redirects to `…/releases/tag/<tag>`; only the `Location` header is read, so
//! there is no response body or JSON to parse.

use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio_rustls::TlsConnector;

/// GitHub `owner/repo` this build checks against. The casing must match the
/// canonical repository name exactly: GitHub 301-redirects a wrong-case path to
/// the canonical one, and `fetch_latest_tag` follows only a single redirect, so a
/// mismatch would capture the case-fix redirect instead of the release tag.
const REPO: &str = "crtGhoul/QuicMic";

/// Overall budget for the whole check, so a slow or black-holed network can never
/// keep the background task alive indefinitely.
const TIMEOUT: Duration = Duration::from_secs(5);

/// Public URL of the releases page, shown to the user (terminal + web UI).
pub fn releases_url() -> String {
    format!("https://github.com/{REPO}/releases")
}

/// Return `Some(tag)` only if GitHub's latest release is strictly newer than the
/// running binary. `None` on equal/older, or on any error (always silent).
pub async fn latest_if_newer() -> Option<String> {
    let tag = tokio::time::timeout(TIMEOUT, fetch_latest_tag())
        .await
        .ok()?
        .ok()?;
    let latest = parse_version(&tag)?;
    let current = parse_version(env!("CARGO_PKG_VERSION"))?;
    (latest > current).then_some(tag)
}

/// Parse `v1.2.3` / `1.2.3` (ignoring any `-pre` / `+build` suffix) into a
/// comparable tuple. Returns `None` for anything that isn't three numeric parts.
fn parse_version(s: &str) -> Option<(u64, u64, u64)> {
    let core = s.trim().trim_start_matches('v');
    let core = core.split(['-', '+']).next().unwrap_or(core);
    let mut parts = core.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch = parts.next()?.parse().ok()?;
    Some((major, minor, patch))
}

/// Fetch the latest release tag by reading the `Location` header of the
/// `releases/latest` redirect. No response body is parsed.
async fn fetch_latest_tag() -> anyhow::Result<String> {
    // Client TLS using the OS trust store — GitHub serves a publicly-trusted cert,
    // unlike our own self-signed LAN cert. The crypto provider is the process
    // default installed in `run` (ring).
    let mut roots = rustls::RootCertStore::empty();
    for cert in rustls_native_certs::load_native_certs().certs {
        let _ = roots.add(cert);
    }
    if roots.is_empty() {
        anyhow::bail!("no OS root certificates available");
    }
    let config = rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();
    let connector = TlsConnector::from(Arc::new(config));

    let tcp = TcpStream::connect("github.com:443").await?;
    let domain = rustls::pki_types::ServerName::try_from("github.com")?;
    let mut tls = connector.connect(domain, tcp).await?;

    let request = format!(
        "GET /{REPO}/releases/latest HTTP/1.1\r\n\
         Host: github.com\r\n\
         User-Agent: quicmic\r\n\
         Accept: */*\r\n\
         Connection: close\r\n\r\n"
    );
    tls.write_all(request.as_bytes()).await?;

    // Read only up to the end of the header block; the Location header carries the
    // resolved tag and we never need the body.
    let mut buf = Vec::new();
    let mut chunk = [0u8; 2048];
    loop {
        let n = match tls.read(&mut chunk).await {
            Ok(0) => break,
            Ok(n) => n,
            // A peer that closes without a TLS close_notify surfaces as UnexpectedEof;
            // treat it as a clean end since we already have what we need.
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        };
        buf.extend_from_slice(&chunk[..n]);
        if find_subslice(&buf, b"\r\n\r\n").is_some() || buf.len() > 16 * 1024 {
            break;
        }
    }

    let text = String::from_utf8_lossy(&buf);
    let location = text
        .lines()
        .take_while(|line| !line.is_empty()) // headers only — stop at the blank line
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.trim()
                .eq_ignore_ascii_case("location")
                .then(|| value.trim())
        })
        .ok_or_else(|| anyhow::anyhow!("no Location header (repo missing or no releases?)"))?;

    let tag = location.rsplit('/').next().unwrap_or_default().trim();
    if tag.is_empty() {
        anyhow::bail!("empty tag in Location header: {location}");
    }
    Ok(tag.to_string())
}

/// Index of the first occurrence of `needle` within `haystack`.
fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::parse_version;

    #[test]
    fn parses_and_orders_versions() {
        assert_eq!(parse_version("v1.2.3"), Some((1, 2, 3)));
        assert_eq!(parse_version("0.10.0"), Some((0, 10, 0)));
        // Pre-release / build metadata is ignored.
        assert_eq!(parse_version("v0.2.0-rc1"), Some((0, 2, 0)));
        // Malformed inputs never parse (so they can't trigger a false update).
        assert_eq!(parse_version("garbage"), None);
        assert_eq!(parse_version("v1.2"), None);

        // Ordering used by `latest_if_newer`.
        assert!(parse_version("v0.2.0") > parse_version("v0.1.9"));
        assert!(parse_version("v1.0.0") > parse_version("v0.9.9"));
        assert!(parse_version("v0.1.0") == parse_version("0.1.0"));
    }
}
