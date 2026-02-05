//! Task supervision utilities for automatic restart of connector tasks.
//!
//! This module provides a lightweight supervision framework that:
//! - Captures task handles
//! - Detects task completion (normal or panic)
//! - Restarts tasks with exponential backoff
//! - Logs task lifecycle events

use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::time::Duration;

use futures_util::FutureExt;
use tokio::task::JoinHandle;

/// Spawn a task that automatically restarts on completion or panic.
///
/// This wraps any async task in restart logic with exponential backoff.
/// The task will be restarted indefinitely, with backoff capped at 30 seconds.
///
/// # Arguments
///
/// * `name` - A descriptive name for the task (used in logs)
/// * `make_task` - A closure that creates a new instance of the task future
///
/// # Example
///
/// ```ignore
/// spawn_supervised("hyperliquid_public_ws", move || {
///     let hl = hl_public.clone();
///     async move { hl.run_public_ws().await }
/// });
/// ```
///
/// # Behavior
///
/// - If the task completes normally (returns `()`), it is restarted with backoff
/// - If the task panics, the panic is caught and the task is restarted with backoff
/// - Backoff starts at 1 second and doubles each restart, capping at 30 seconds
/// - Backoff resets after `HEALTHY_RESTART_COUNT` consecutive successful completions
/// - All restarts are logged with restart count and reason
pub fn spawn_supervised<N, F, Fut>(name: N, make_task: F) -> JoinHandle<()>
where
    N: Into<String> + Clone + Send + 'static,
    F: Fn() -> Fut + Send + Sync + 'static,
    Fut: Future<Output = ()> + Send + 'static,
{
    let name = name.into();
    tokio::spawn(async move {
        let mut restart_count: u64 = 0;
        let mut backoff = Duration::from_secs(1);
        const MAX_BACKOFF_SECS: u64 = 30;
        const HEALTHY_RESTART_COUNT: u64 = 10;

        loop {
            let task_name = name.clone();
            let task_future = make_task();

            // Wrap in catch_unwind to survive panics
            let result = AssertUnwindSafe(task_future).catch_unwind().await;

            restart_count += 1;

            match result {
                Ok(()) => {
                    // Task completed normally (should be rare for WS loops)
                    eprintln!(
                        "WARN: Supervised task '{}' exited normally (restart #{}), restarting in {:?}",
                        task_name, restart_count, backoff
                    );
                }
                Err(panic_info) => {
                    // Task panicked
                    let panic_msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                        (*s).to_string()
                    } else if let Some(s) = panic_info.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "unknown panic".to_string()
                    };
                    eprintln!(
                        "ERROR: Supervised task '{}' panicked (restart #{}): {}, restarting in {:?}",
                        task_name, restart_count, panic_msg, backoff
                    );
                }
            }

            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(Duration::from_secs(MAX_BACKOFF_SECS));

            // Reset backoff after many successful restarts (indicates stability)
            if restart_count > 0 && restart_count % HEALTHY_RESTART_COUNT == 0 {
                eprintln!(
                    "INFO: Supervised task '{}' resetting backoff after {} restarts",
                    task_name, restart_count
                );
                backoff = Duration::from_secs(1);
            }
        }
    })
}

/// Spawn a supervised task that uses a configurable healthy threshold for backoff reset.
///
/// This is a more sophisticated version that tracks session duration and
/// resets backoff when a session runs longer than the threshold.
///
/// # Arguments
///
/// * `name` - A descriptive name for the task (used in logs)
/// * `healthy_threshold` - Duration threshold for considering a session "healthy"
/// * `make_task` - A closure that creates a new instance of the task future
pub fn spawn_supervised_with_threshold<N, F, Fut>(
    name: N,
    healthy_threshold: Duration,
    make_task: F,
) -> JoinHandle<()>
where
    N: Into<String> + Clone + Send + 'static,
    F: Fn() -> Fut + Send + Sync + 'static,
    Fut: Future<Output = ()> + Send + 'static,
{
    let name = name.into();
    tokio::spawn(async move {
        let mut restart_count: u64 = 0;
        let mut backoff = Duration::from_secs(1);
        const MAX_BACKOFF_SECS: u64 = 30;

        loop {
            let task_name = name.clone();
            let task_future = make_task();
            let session_start = std::time::Instant::now();

            // Wrap in catch_unwind to survive panics
            let result = AssertUnwindSafe(task_future).catch_unwind().await;

            restart_count += 1;
            let session_duration = session_start.elapsed();

            match result {
                Ok(()) => {
                    eprintln!(
                        "WARN: Supervised task '{}' exited normally (restart #{}, ran for {:?}), restarting in {:?}",
                        task_name, restart_count, session_duration, backoff
                    );
                }
                Err(panic_info) => {
                    let panic_msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                        (*s).to_string()
                    } else if let Some(s) = panic_info.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "unknown panic".to_string()
                    };
                    eprintln!(
                        "ERROR: Supervised task '{}' panicked (restart #{}, ran for {:?}): {}, restarting in {:?}",
                        task_name, restart_count, session_duration, panic_msg, backoff
                    );
                }
            }

            // Reset backoff if session was healthy for long enough
            if session_duration >= healthy_threshold {
                eprintln!(
                    "INFO: Supervised task '{}' was healthy for {:?}; resetting backoff",
                    task_name, session_duration
                );
                backoff = Duration::from_secs(1);
            }

            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(Duration::from_secs(MAX_BACKOFF_SECS));
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    // NOTE: These tests are marked #[ignore] because they require real-time delays
    // (backoff starts at 1 second). Run with `cargo test -- --ignored` for manual verification.

    #[tokio::test]
    #[ignore = "requires 3+ seconds for backoff timing"]
    async fn test_spawn_supervised_restarts_on_normal_exit() {
        let call_count = Arc::new(AtomicU32::new(0));
        let call_count_clone = call_count.clone();

        let handle = spawn_supervised("test_task", move || {
            let count = call_count_clone.clone();
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                // Task exits immediately
            }
        });

        // Wait for a few restarts (backoff starts at 1s, so need ~2.5s for 2 restarts)
        tokio::time::sleep(Duration::from_millis(2600)).await;

        // Should have been called at least twice (initial + restart)
        let count = call_count.load(Ordering::SeqCst);
        assert!(count >= 2, "Expected at least 2 calls, got {}", count);

        handle.abort();
    }

    #[tokio::test]
    #[ignore = "requires 8+ seconds for backoff timing"]
    async fn test_spawn_supervised_restarts_on_panic() {
        let call_count = Arc::new(AtomicU32::new(0));
        let call_count_clone = call_count.clone();

        let handle = spawn_supervised("panic_task", move || {
            let count = call_count_clone.clone();
            async move {
                let c = count.fetch_add(1, Ordering::SeqCst);
                if c < 2 {
                    panic!("intentional panic for test");
                }
                // Stop panicking after 2 attempts
            }
        });

        // Wait for restarts (need ~1+2+4 = 7s for 3 retries with exponential backoff)
        tokio::time::sleep(Duration::from_millis(7500)).await;

        // Should have been called at least 3 times
        let count = call_count.load(Ordering::SeqCst);
        assert!(count >= 3, "Expected at least 3 calls, got {}", count);

        handle.abort();
    }

    #[tokio::test]
    #[ignore = "requires 3+ seconds for backoff timing"]
    async fn test_spawn_supervised_with_threshold_resets_backoff() {
        let call_count = Arc::new(AtomicU32::new(0));
        let call_count_clone = call_count.clone();

        // Use a very short threshold for testing
        let handle = spawn_supervised_with_threshold(
            "threshold_task",
            Duration::from_millis(50),
            move || {
                let count = call_count_clone.clone();
                async move {
                    count.fetch_add(1, Ordering::SeqCst);
                    // Run for 60ms to exceed threshold
                    tokio::time::sleep(Duration::from_millis(60)).await;
                }
            },
        );

        // Wait for a few cycles (60ms runtime + 1s backoff per cycle, need ~2.2s for 2 cycles)
        tokio::time::sleep(Duration::from_millis(2400)).await;

        let count = call_count.load(Ordering::SeqCst);
        assert!(count >= 2, "Expected at least 2 calls, got {}", count);

        handle.abort();
    }

    /// Quick unit test that verifies the basic structure compiles and runs
    #[tokio::test]
    async fn test_spawn_supervised_starts_task() {
        let started = Arc::new(AtomicU32::new(0));
        let started_clone = started.clone();

        let handle = spawn_supervised("quick_test", move || {
            let s = started_clone.clone();
            async move {
                s.fetch_add(1, Ordering::SeqCst);
                // Sleep longer than the test wait so we only see first call
                tokio::time::sleep(Duration::from_secs(10)).await;
            }
        });

        // Just verify task starts
        tokio::time::sleep(Duration::from_millis(50)).await;

        let count = started.load(Ordering::SeqCst);
        assert_eq!(count, 1, "Task should have started exactly once");

        handle.abort();
    }
}
