use std::cmp::max;
use std::fs;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug)]
pub struct LighterNonceManager {
    last: AtomicU64,
}

impl LighterNonceManager {
    pub fn new(initial: Option<u64>) -> Self {
        Self {
            last: AtomicU64::new(initial.unwrap_or(0)),
        }
    }

    pub fn next(&self, now_ms: u64) -> u64 {
        loop {
            let cur = self.last.load(Ordering::Relaxed);
            let candidate = max(now_ms, cur.saturating_add(1));
            if self
                .last
                .compare_exchange(cur, candidate, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                return candidate;
            }
        }
    }

    pub fn get(&self) -> u64 {
        self.last.load(Ordering::Relaxed)
    }
}

pub fn load_last_nonce(path: &Path) -> io::Result<Option<u64>> {
    if !path.exists() {
        return Ok(None);
    }
    let content = fs::read_to_string(path)?;
    let value: serde_json::Value = serde_json::from_str(&content)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
    let last = value
        .get("last_nonce")
        .or_else(|| value.get("nonce"))
        .and_then(|v| v.as_u64())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing nonce field"))?;
    Ok(Some(last))
}

pub fn store_last_nonce(path: &Path, nonce: u64) -> io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("lighter_nonce.json");
    let tmp_name = format!(".{}.tmp.{}", file_name, std::process::id());
    let tmp_path = parent.join(tmp_name);
    let payload = serde_json::json!({ "last_nonce": nonce });
    let content = serde_json::to_string(&payload)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
    fs::write(&tmp_path, content)?;
    fs::rename(&tmp_path, path).or_else(|err| {
        let _ = fs::remove_file(path);
        fs::rename(&tmp_path, path).map_err(|_| err)
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{load_last_nonce, store_last_nonce, LighterNonceManager};
    use std::collections::BTreeSet;
    use std::fs;
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(label: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let mut path = std::env::temp_dir();
        path.push(format!(
            "lighter_nonce_{}_{}_{}.json",
            label,
            std::process::id(),
            nanos
        ));
        path
    }

    #[test]
    fn nonce_monotonic_sequential() {
        let manager = LighterNonceManager::new(Some(0));
        let now_ms = 1_000;
        let first = manager.next(now_ms);
        let second = manager.next(now_ms);
        let third = manager.next(now_ms);
        assert_eq!(first + 1, second);
        assert_eq!(second + 1, third);
    }

    #[test]
    fn nonce_monotonic_with_time_jumps() {
        let manager = LighterNonceManager::new(Some(0));
        let n1 = manager.next(1_000);
        let n2 = manager.next(2_000);
        let n3 = manager.next(2_500);
        assert!(n1 >= 1_000);
        assert!(n2 >= 2_000);
        assert!(n3 >= 2_500);
        assert!(n1 < n2 && n2 < n3);
    }

    #[test]
    fn nonce_monotonic_concurrent() {
        let manager = Arc::new(LighterNonceManager::new(Some(0)));
        let results = Arc::new(Mutex::new(Vec::new()));
        let now_ms = 5_000;
        let threads = 32;
        let per_thread = 64;
        let mut handles = Vec::with_capacity(threads);
        for _ in 0..threads {
            let manager = Arc::clone(&manager);
            let results = Arc::clone(&results);
            handles.push(thread::spawn(move || {
                let mut local = Vec::with_capacity(per_thread);
                for _ in 0..per_thread {
                    local.push(manager.next(now_ms));
                }
                let mut guard = results.lock().expect("lock results");
                guard.extend(local);
            }));
        }
        for handle in handles {
            handle.join().expect("join");
        }
        let guard = results.lock().expect("lock results");
        assert_eq!(guard.len(), threads * per_thread);
        let uniques: BTreeSet<u64> = guard.iter().copied().collect();
        assert_eq!(uniques.len(), guard.len());
    }

    #[test]
    fn nonce_persist_roundtrip() {
        let path = temp_path("roundtrip");
        let _ = fs::remove_file(&path);
        store_last_nonce(&path, 12_345).expect("store");
        let loaded = load_last_nonce(&path).expect("load");
        assert_eq!(loaded, Some(12_345));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn nonce_load_missing_is_none() {
        let path = temp_path("missing");
        let _ = fs::remove_file(&path);
        let loaded = load_last_nonce(&path).expect("load");
        assert_eq!(loaded, None);
    }
}
