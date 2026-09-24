//! Shared server state and single-connection lifecycle management.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::audio::RingBuffer;
use crate::tls::TlsIdentity;

/// State shared between both transport handlers and the HTTP API.
/// Extracted as a separate struct to keep the transport layer decoupled
/// from HTTP-specific fields.
#[derive(Clone)]
pub struct StreamState {
    pub ring: Arc<RingBuffer>,
    pub is_connected: Arc<AtomicBool>,
    pub session_token: Arc<parking_lot::Mutex<Option<String>>>,
    pub noise_gate: Arc<AtomicU32>, // f32::to_bits(), 0.0 = disabled
    pub gain: Arc<AtomicU32>,       // f32::to_bits(), 1.0 = unity
    pub latency_threshold: Arc<AtomicU32>, // u32 milliseconds
    /// PC-side output volume multiplier (f32 bits, 1.0 = unity). Applied in the
    /// output stage after resampling, so it scales both the virtual-device
    /// stream and the hear-yourself monitor stream. Adjustable at runtime via
    /// `POST /api/settings` (and the phone's settings panel).
    pub output_volume: Arc<AtomicU32>, // f32::to_bits(), 1.0 = unity
    /// Second ring feeding the hear-yourself monitor stream, enabled at startup
    /// via `--monitor-device`. `None` when the monitor is off; the decode hot
    /// path pushes to both rings only when this is `Some`, keeping each ring's
    /// SPSC contract intact.
    pub monitor_ring: Option<Arc<RingBuffer>>,
    /// Runtime mute for the monitor stream, toggled from the phone UI via
    /// `POST /api/monitor`. Checked in the monitor output callback: `true` =
    /// audible. Defaults to `true` when the monitor stream is created.
    pub monitor_enabled: Arc<AtomicBool>,
    pub packets_received: Arc<AtomicU64>,
    pub packets_lost: Arc<AtomicU64>,
    pub source_sample_rate: Arc<AtomicU32>, // Client's actual capture rate
    pub cancel_tx: tokio::sync::broadcast::Sender<()>,
    pub is_shutdown: Arc<AtomicBool>,
    /// Whether a working audio output stream is currently active. Cleared while the
    /// output device is lost (disabled/removed) and the supervisor is rebuilding it;
    /// surfaced to the client via `/api/stats` so the UI can warn the user.
    pub device_ok: Arc<AtomicBool>,
}

/// Consecutive failed PIN attempts (per client IP) before a lockout kicks in.
const MAX_FAILED_ATTEMPTS: u32 = 5;
/// How long a client IP stays locked out after hitting the failure threshold.
const LOCKOUT_DURATION: Duration = Duration::from_secs(30);
/// A client IP's failure count is forgotten after this long with no new failed
/// attempt. This both decays the counter for a legitimate user who simply
/// mistyped and bounds the map's memory: idle entries are pruned away.
const FAILURE_DECAY: Duration = Duration::from_secs(60);

/// Per-IP brute-force counter for the pairing endpoint.
struct ThrottleEntry {
    failed_attempts: u32,
    locked_until: Option<Instant>,
    last_seen: Instant,
}

/// Brute-force protection state for the pairing endpoint, keyed by client IP so
/// one misbehaving host cannot lock everyone else out (the previous single global
/// counter did). Kept behind one lock (in `AppState`) so each IP's counter and
/// lockout deadline are read and updated together. Idle/expired entries are
/// pruned on every check, so the map stays bounded — and a completed HTTPS pair
/// request requires a real TCP+TLS handshake, so the key cannot be spoofed to
/// flood it.
#[derive(Default)]
pub struct PairingThrottle {
    entries: HashMap<IpAddr, ThrottleEntry>,
}

impl PairingThrottle {
    /// Remaining lockout seconds for `ip` if it is currently locked out. Clears an
    /// expired lockout and prunes stale entries as a side effect.
    pub(super) fn locked_remaining(&mut self, ip: IpAddr, now: Instant) -> Option<u64> {
        self.prune(now);
        let entry = self.entries.get_mut(&ip)?;
        match entry.locked_until {
            Some(until) if until > now => Some((until - now).as_secs()),
            _ => {
                entry.locked_until = None;
                None
            }
        }
    }

    /// Record a failed attempt for `ip`. Returns `(attempts, locked)` where
    /// `attempts` is the running failure count and `locked` is true if this
    /// attempt tripped the lockout (after which the count resets).
    pub(super) fn register_failure(&mut self, ip: IpAddr, now: Instant) -> (u32, bool) {
        let entry = self.entries.entry(ip).or_insert(ThrottleEntry {
            failed_attempts: 0,
            locked_until: None,
            last_seen: now,
        });
        entry.last_seen = now;
        entry.failed_attempts += 1;
        let attempts = entry.failed_attempts;
        let locked = attempts >= MAX_FAILED_ATTEMPTS;
        if locked {
            entry.locked_until = Some(now + LOCKOUT_DURATION);
            entry.failed_attempts = 0;
        }
        (attempts, locked)
    }

    /// Clear all failure state for `ip` after a successful pairing.
    pub(super) fn clear(&mut self, ip: IpAddr) {
        self.entries.remove(&ip);
    }

    /// Drop entries that are neither actively locked out nor recently active, so
    /// the map only ever holds IPs with live brute-force state.
    fn prune(&mut self, now: Instant) {
        self.entries.retain(|_, e| {
            matches!(e.locked_until, Some(until) if until > now)
                || now.duration_since(e.last_seen) < FAILURE_DECAY
        });
    }
}

/// Full application state accessible from all HTTP handlers.
#[derive(Clone)]
pub struct AppState {
    pub stream: StreamState,
    pub tls_identity: TlsIdentity,
    pub pairing_pin: String,
    pub wt_port: u16,
    pub lan_ip: String,
    pub pairing_throttle: Arc<parking_lot::Mutex<PairingThrottle>>,
    /// Latest newer release tag found by the startup update check, if any. Read by
    /// `/api/info` so the web UI can show a small "update available" banner.
    pub update_status: Arc<parking_lot::Mutex<Option<String>>>,
}

/// Try to atomically claim the single-connection slot, retrying briefly to
/// absorb a fast F5 handover. Returns `true` once acquired. Shared by both
/// transports so the CAS policy lives in one place.
pub(super) async fn acquire_connection_slot(is_connected: &AtomicBool) -> bool {
    for _ in 0..10 {
        if is_connected
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            return true;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    false
}

/// RAII guard that releases the single-connection slot when dropped, however
/// the session task ends (completion, cancellation, or panic).
pub(super) struct ConnectionGuard {
    is_connected: Arc<AtomicBool>,
}

impl ConnectionGuard {
    pub(super) fn new(is_connected: Arc<AtomicBool>) -> Self {
        Self { is_connected }
    }
}

impl Drop for ConnectionGuard {
    fn drop(&mut self) {
        self.is_connected.store(false, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::{acquire_connection_slot, ConnectionGuard};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    #[tokio::test]
    async fn slot_acquires_and_guard_releases() {
        let flag = Arc::new(AtomicBool::new(false));
        assert!(acquire_connection_slot(&flag).await);
        assert!(flag.load(Ordering::SeqCst));
        {
            let _guard = ConnectionGuard::new(flag.clone());
            assert!(flag.load(Ordering::SeqCst));
        }
        assert!(
            !flag.load(Ordering::SeqCst),
            "dropping the guard must release the slot"
        );
    }

    #[tokio::test]
    async fn slot_acquisition_fails_when_already_taken() {
        let flag = Arc::new(AtomicBool::new(true));
        assert!(!acquire_connection_slot(&flag).await);
    }

    #[test]
    fn throttle_locks_out_per_ip_independently() {
        use super::{PairingThrottle, MAX_FAILED_ATTEMPTS};
        use std::net::{IpAddr, Ipv4Addr};
        use std::time::Instant;

        let mut throttle = PairingThrottle::default();
        let attacker = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 50));
        let victim = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 51));
        let now = Instant::now();

        // The attacker burns through the failure budget and gets locked out.
        for _ in 0..MAX_FAILED_ATTEMPTS - 1 {
            assert!(!throttle.register_failure(attacker, now).1);
        }
        assert!(
            throttle.register_failure(attacker, now).1,
            "5th failure locks"
        );
        assert!(throttle.locked_remaining(attacker, now).is_some());

        // A different IP is unaffected — the old global counter would have locked
        // it too.
        assert!(throttle.locked_remaining(victim, now).is_none());

        // A successful pair clears the attacker's state.
        throttle.clear(attacker);
        assert!(throttle.locked_remaining(attacker, now).is_none());
    }

    #[test]
    fn throttle_prunes_idle_entries() {
        use super::{PairingThrottle, FAILURE_DECAY};
        use std::net::{IpAddr, Ipv4Addr};
        use std::time::Instant;

        let mut throttle = PairingThrottle::default();
        let ip = IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1));
        let t0 = Instant::now();

        // One failure, then no activity for longer than the decay window.
        throttle.register_failure(ip, t0);
        let later = t0 + FAILURE_DECAY + std::time::Duration::from_secs(1);

        // The stale entry is pruned, so the counter has effectively reset.
        assert!(throttle.locked_remaining(ip, later).is_none());
        assert_eq!(
            throttle.register_failure(ip, later).0,
            1,
            "decayed entry must restart the count at 1"
        );
    }
}
