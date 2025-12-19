use paraphina::metrics::{DrawdownTracker, OnlineStats};

#[test]
fn online_stats_basic() {
    let mut s = OnlineStats::default();
    s.add(1.0);
    s.add(2.0);
    s.add(3.0);

    assert_eq!(s.n(), 3);
    assert!((s.mean() - 2.0).abs() < 1e-12);

    // Population variance for [1,2,3] is 2/3.
    let var_pop = s.variance_population();
    assert!((var_pop - (2.0 / 3.0)).abs() < 1e-12);

    // Sample variance for [1,2,3] is 1.
    let var_samp = s.variance_sample();
    assert!((var_samp - 1.0).abs() < 1e-12);

    assert_eq!(s.min(), 1.0);
    assert_eq!(s.max(), 3.0);
}

#[test]
fn drawdown_tracker_basic() {
    let mut dd = DrawdownTracker::new();

    // equity curve: 0 -> 2 -> 1 -> 3 -> 2
    dd.update(0.0);
    dd.update(2.0);
    dd.update(1.0); // drawdown 1.0 from peak 2.0
    dd.update(3.0); // new peak
    dd.update(2.0); // drawdown 1.0 from peak 3.0

    assert!((dd.peak() - 3.0).abs() < 1e-12);
    assert!((dd.max_drawdown() - 1.0).abs() < 1e-12);
}

#[test]
fn drawdown_tracker_handles_negative_equity() {
    let mut dd = DrawdownTracker::new();

    dd.update(0.0);
    dd.update(-1.0);
    dd.update(-2.0);

    // peak is 0.0, trough is -2.0 => drawdown is 2.0
    assert!((dd.max_drawdown() - 2.0).abs() < 1e-12);
}
