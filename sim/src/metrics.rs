use std::sync::{Mutex, PoisonError};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use cpu_time::ProcessTime;
use serde::{Deserialize, Serialize};

/// One timed step on one node. Times are unix microseconds, so spans from
/// different nodes line up as long as the nodes share a clock (true for
/// containers on one host).
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct Span {
    pub(crate) name: String,
    pub(crate) start_us: u64,
    pub(crate) end_us: u64,
    /// Bytes sent or received during the step.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) bytes: Option<usize>,
    /// CPU time the whole process used during the step. Next to the wall time
    /// it shows how many cores the step kept busy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) cpu_us: Option<u64>,
}

pub(crate) fn now_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_micros() as u64)
}

pub(crate) struct Timer {
    name: String,
    start_us: u64,
    cpu: Option<ProcessTime>,
}

impl Timer {
    pub(crate) fn start(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            start_us: now_us(),
            cpu: None,
        }
    }

    /// Like [`Timer::start`], and also measures the process's cpu time.
    pub(crate) fn start_cpu(name: impl Into<String>) -> Self {
        Self {
            cpu: Some(ProcessTime::now()),
            ..Self::start(name)
        }
    }

    pub(crate) fn finish(self) -> Span {
        Span {
            name: self.name,
            start_us: self.start_us,
            end_us: now_us(),
            bytes: None,
            cpu_us: self.cpu.map(|c| c.elapsed().as_micros() as u64),
        }
    }

    pub(crate) fn finish_bytes(self, bytes: usize) -> Span {
        Span {
            bytes: Some(bytes),
            ..self.finish()
        }
    }
}

/// First and last time something happened, plus the bytes involved. Used
/// where per-event spans would be too many, like a server receiving one
/// opening from every client.
#[derive(Clone, Copy, Default)]
pub(crate) struct Window {
    first_us: Option<u64>,
    last_us: u64,
    bytes: usize,
}

impl Window {
    pub(crate) fn record(&mut self, bytes: usize) {
        let now = now_us();
        self.first_us.get_or_insert(now);
        self.last_us = now;
        self.bytes += bytes;
    }

    /// `None` if nothing was recorded.
    pub(crate) fn span(&self, name: &str) -> Option<Span> {
        Some(Span {
            name: name.to_string(),
            start_us: self.first_us?,
            end_us: self.last_us,
            bytes: Some(self.bytes),
            cpu_us: None,
        })
    }
}

/// One span covering all of `spans`, with their bytes added up.
pub(crate) fn merge(name: &str, spans: &[Span]) -> Span {
    Span {
        name: name.to_string(),
        start_us: spans.iter().map(|s| s.start_us).min().unwrap_or(0),
        end_us: spans.iter().map(|s| s.end_us).max().unwrap_or(0),
        bytes: Some(spans.iter().filter_map(|s| s.bytes).sum()),
        cpu_us: None,
    }
}

/// What the node's whole process used since its previous report, which is
/// one round from the second round on (the first also covers startup).
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub(crate) struct Usage {
    /// CPU time, user and system, including networking and serialization.
    pub(crate) cpu_us: u64,
    /// Most memory the process held at once. `None` without linux's `/proc`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) peak_rss_bytes: Option<u64>,
    /// Memory held at the time of the report.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) rss_bytes: Option<u64>,
    /// Bytes sent and received on the network interface, so with HTTP, TCP
    /// and IP headers and resent packets. `None` outside a container.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) sent_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) received_bytes: Option<u64>,
}

struct Reported {
    cpu: Duration,
    sent: u64,
    received: u64,
}

static REPORTED: Mutex<Reported> = Mutex::new(Reported {
    cpu: Duration::ZERO,
    sent: 0,
    received: 0,
});

/// Measures [`Usage`] and starts the next measurement. Covers the whole
/// process, so it only means something with one node per process.
pub(crate) fn take_usage() -> Usage {
    let cpu = ProcessTime::now().as_duration();
    let sent = interface_counter("tx_bytes");
    let received = interface_counter("rx_bytes");
    let mut reported = REPORTED.lock().unwrap_or_else(PoisonError::into_inner);
    let cpu_us = cpu.saturating_sub(reported.cpu).as_micros() as u64;
    reported.cpu = cpu;
    let sent_bytes = sent.map(|n| n.saturating_sub(std::mem::replace(&mut reported.sent, n)));
    let received_bytes =
        received.map(|n| n.saturating_sub(std::mem::replace(&mut reported.received, n)));

    let status = std::fs::read_to_string("/proc/self/status").ok();
    let field = |name: &str| {
        let line = status
            .as_deref()?
            .lines()
            .find_map(|l| l.strip_prefix(name))?;
        let kb: u64 = line.trim().strip_suffix("kB")?.trim().parse().ok()?;
        Some(kb * 1024)
    };
    let usage = Usage {
        cpu_us,
        peak_rss_bytes: field("VmHWM:"),
        rss_bytes: field("VmRSS:"),
        sent_bytes,
        received_bytes,
    };
    // resets the peak to the current size
    if let Err(e) = std::fs::write("/proc/self/clear_refs", "5") {
        tracing::debug!(%e, "could not reset peak memory");
    }
    usage
}

fn interface_counter(name: &str) -> Option<u64> {
    let path = format!("/sys/class/net/eth0/statistics/{name}");
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

/// The link and CPU budget `run.sh` gave this node's container.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub(crate) struct Link {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) delay: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) loss: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) up: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) down: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) cpus: Option<String>,
}

impl Link {
    pub(crate) fn from_env() -> Self {
        let var = |name| std::env::var(name).ok().filter(|v| !v.is_empty());
        Self {
            delay: var("NET_DELAY"),
            loss: var("NET_LOSS"),
            up: var("NET_UP"),
            down: var("NET_DOWN"),
            cpus: var("SIM_CPUS"),
        }
    }
}
