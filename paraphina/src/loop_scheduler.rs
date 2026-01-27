// src/loop_scheduler.rs
//
// Deterministic single-thread scheduler for §16 loop cadences.

#[derive(Debug, Clone)]
pub struct LoopScheduler {
    main_interval_ms: i64,
    hedge_interval_ms: i64,
    risk_interval_ms: i64,
    next_main_ms: i64,
    next_hedge_ms: i64,
    next_risk_ms: i64,
}

impl LoopScheduler {
    pub fn new(
        base_ms: i64,
        main_interval_ms: i64,
        hedge_interval_ms: i64,
        risk_interval_ms: i64,
    ) -> Self {
        let main = main_interval_ms.max(1);
        let hedge = hedge_interval_ms.max(1);
        let risk = risk_interval_ms.max(1);
        Self {
            main_interval_ms: main,
            hedge_interval_ms: hedge,
            risk_interval_ms: risk,
            next_main_ms: base_ms,
            next_hedge_ms: base_ms,
            next_risk_ms: base_ms,
        }
    }

    pub fn next_main_ms(&self) -> i64 {
        self.next_main_ms
    }

    pub fn advance_main(&mut self) -> i64 {
        let now_ms = self.next_main_ms;
        self.next_main_ms += self.main_interval_ms;
        now_ms
    }

    pub fn risk_due(&self, now_ms: i64) -> bool {
        now_ms >= self.next_risk_ms
    }

    pub fn hedge_due(&self, now_ms: i64) -> bool {
        now_ms >= self.next_hedge_ms
    }

    pub fn mark_risk_ran(&mut self) {
        self.next_risk_ms += self.risk_interval_ms;
    }

    pub fn mark_hedge_ran(&mut self) {
        self.next_hedge_ms += self.hedge_interval_ms;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn risk_and_hedge_cadences() {
        let mut sched = LoopScheduler::new(0, 1000, 2000, 3000);
        let mut risk_hits = 0;
        let mut hedge_hits = 0;
        for _ in 0..6 {
            let now = sched.advance_main();
            if sched.risk_due(now) {
                risk_hits += 1;
                sched.mark_risk_ran();
            }
            if sched.hedge_due(now) {
                hedge_hits += 1;
                sched.mark_hedge_ran();
            }
        }
        assert_eq!(risk_hits, 2); // at t=0 and t=3000
        assert_eq!(hedge_hits, 3); // at t=0, t=2000, t=4000
    }

    #[test]
    fn equal_intervals_fire_every_tick() {
        let mut sched = LoopScheduler::new(0, 1000, 1000, 1000);
        for _ in 0..3 {
            let now = sched.advance_main();
            assert!(sched.risk_due(now));
            assert!(sched.hedge_due(now));
            sched.mark_risk_ran();
            sched.mark_hedge_ran();
        }
    }
}
