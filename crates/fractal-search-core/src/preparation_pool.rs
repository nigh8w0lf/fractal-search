//! Bounded CPU preparation only. No model sessions, catalog or cache writes.
use anyhow::{anyhow, ensure, Context, Result};
use std::{
    collections::VecDeque,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Condvar, Mutex,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

pub(crate) const WORKERS: usize = 2;
// Conservative reservations include decoder scratch plus queued output. The
// caller's current batch and ONNX allocations are separate, as before.
pub(crate) const BUDGET: usize = 2 * 1024 * 1024 * 1024;
pub(crate) const IMAGE_RESERVATION: usize = 1024 * 1024 * 1024;
pub(crate) const MEDIA_RESERVATION: usize = 512 * 1024 * 1024;
pub(crate) const TEXT_RESERVATION: usize = 16 * 1024 * 1024;

type Job = Box<dyn FnOnce(Lease, Arc<Shared>) + Send>;
struct Work {
    bytes: usize,
    abandoned: Arc<AtomicBool>,
    job: Job,
}
#[derive(Default)]
struct State {
    queue: VecDeque<Work>,
    bytes: usize,
    peak: usize,
    tasks: usize,
    work_ms: f64,
    fatal: Option<String>,
}
#[derive(Default)]
struct Shared {
    stop: AtomicBool,
    state: Mutex<State>,
    changed: Condvar,
}
impl Shared {
    fn stop(&self) {
        // A predicate change must hold the same mutex as Condvar::wait. An
        // atomic flag alone can lose the notification between check and wait,
        // leaving finish() blocked forever while joining a sleeping worker.
        let _state = self.state.lock().expect("preparation state");
        self.stop.store(true, Ordering::Release);
        self.changed.notify_all();
    }

    fn check(&self) -> Result<()> {
        let state = self
            .state
            .lock()
            .map_err(|_| anyhow!("Preparation pool state poisoned"))?;
        ensure!(
            state.fatal.is_none(),
            "Preparation worker failed: {}",
            state.fatal.as_deref().unwrap_or("")
        );
        Ok(())
    }
}
struct Lease {
    shared: Arc<Shared>,
    bytes: usize,
}
impl Drop for Lease {
    fn drop(&mut self) {
        let mut state = self.shared.state.lock().expect("preparation state");
        state.bytes -= self.bytes;
        self.shared.changed.notify_all();
    }
}
struct Completed<T> {
    value: T,
    _lease: Lease,
}
pub(crate) struct Ticket<T> {
    receiver: mpsc::Receiver<Completed<T>>,
    shared: Arc<Shared>,
    abandoned: Arc<AtomicBool>,
}
impl<T> Drop for Ticket<T> {
    fn drop(&mut self) {
        let _state = self.shared.state.lock().expect("preparation state");
        self.abandoned.store(true, Ordering::Release);
        self.shared.changed.notify_all();
    }
}
impl<T> Ticket<T> {
    pub(crate) fn wait(self, cancelled: &impl Fn() -> bool) -> Result<Option<T>> {
        loop {
            self.shared.check()?;
            if cancelled() {
                self.shared.stop();
                return Ok(None);
            }
            match self.receiver.recv_timeout(Duration::from_millis(20)) {
                Ok(result) => return Ok(Some(result.value)),
                Err(mpsc::RecvTimeoutError::Timeout) => (),
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    self.shared.check()?;
                    return Err(anyhow!(
                        "Preparation worker disconnected before returning its result"
                    ));
                }
            }
        }
    }
}

pub(crate) struct PreparationPool {
    shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
}
impl PreparationPool {
    pub(crate) fn new() -> Result<Self> {
        let shared = Arc::new(Shared::default());
        let mut pool = Self {
            shared,
            threads: Vec::new(),
        };
        for index in 0..WORKERS {
            let shared = pool.shared.clone();
            pool.threads.push(
                thread::Builder::new()
                    .name(format!("index-prepare-{index}"))
                    .spawn(move || {
                        loop {
                            // Reserve the queue head before advancing. Completed
                            // outputs retain their reservation until consumed/dropped.
                            let mut state = shared.state.lock().expect("preparation state");
                            let work = loop {
                                state
                                    .queue
                                    .retain(|work| !work.abandoned.load(Ordering::Acquire));
                                if shared.stop.load(Ordering::Acquire) {
                                    return;
                                }
                                if state
                                    .queue
                                    .front()
                                    .is_some_and(|work| state.bytes + work.bytes <= BUDGET)
                                {
                                    break state.queue.pop_front().expect("queue head");
                                }
                                state = shared.changed.wait(state).expect("preparation budget");
                            };
                            state.bytes += work.bytes;
                            state.peak = state.peak.max(state.bytes);
                            drop(state);
                            let lease = Lease {
                                shared: shared.clone(),
                                bytes: work.bytes,
                            };
                            let started = Instant::now();
                            let result = catch_unwind(AssertUnwindSafe(|| {
                                (work.job)(lease, shared.clone())
                            }));
                            let mut state = shared.state.lock().expect("preparation state");
                            state.tasks += 1;
                            state.work_ms += started.elapsed().as_secs_f64() * 1000.0;
                            if result.is_err() {
                                state.fatal = Some(
                                    "CPU preparation panicked; indexing was stopped without retry"
                                        .into(),
                                );
                                shared.stop.store(true, Ordering::Release);
                                shared.changed.notify_all();
                            }
                        }
                    })
                    .context("Start shared indexing preparation worker")?,
            );
        }
        Ok(pool)
    }
    pub(crate) fn submit<T: Send + 'static>(
        &self,
        bytes: usize,
        job: impl FnOnce(&AtomicBool) -> T + Send + 'static,
    ) -> Result<Ticket<T>> {
        self.check()?;
        ensure!(
            bytes > 0 && bytes <= BUDGET,
            "Invalid indexing preparation memory reservation"
        );
        ensure!(
            !self.shared.stop.load(Ordering::Acquire),
            "Preparation pool has stopped"
        );
        let (sender, receiver) = mpsc::sync_channel(1);
        let abandoned = Arc::new(AtomicBool::new(false));
        let mut state = self
            .shared
            .state
            .lock()
            .map_err(|_| anyhow!("Preparation queue poisoned"))?;
        state
            .queue
            .retain(|work| !work.abandoned.load(Ordering::Acquire));
        ensure!(
            state.queue.len() < 4,
            "Preparation queue exceeded four pending jobs"
        );
        state.queue.push_back(Work {
            bytes,
            abandoned: abandoned.clone(),
            job: Box::new(move |lease, shared| {
                let value = job(&shared.stop);
                // An admission check can establish that prefetched work is no
                // longer needed; dropping that ticket discards its output.
                let _ = sender.send(Completed {
                    value,
                    _lease: lease,
                });
            }),
        });
        self.shared.changed.notify_all();
        drop(state);
        Ok(Ticket {
            receiver,
            shared: self.shared.clone(),
            abandoned,
        })
    }
    pub(crate) fn check(&self) -> Result<()> {
        self.shared.check()
    }
    pub(crate) fn finish(&mut self) -> Result<()> {
        self.shared.stop();
        let mut panicked = false;
        for worker in self.threads.drain(..) {
            panicked |= worker.join().is_err();
        }
        ensure!(
            !panicked,
            "Preparation worker thread panicked during shutdown"
        );
        self.check()
    }
}
impl Drop for PreparationPool {
    fn drop(&mut self) {
        if let Err(error) = self.finish() {
            eprintln!("[Preparation] Shutdown error: {error:#}");
        }
        let state = self.shared.state.lock().expect("preparation state");
        eprintln!("[Preparation] workers={} tasks={} worker_ms={:.1} peak_reserved_bytes={} budget_bytes={}", WORKERS, state.tasks, state.work_ms, state.peak, BUDGET);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stopping_synchronizes_with_the_waiter_predicate() {
        let shared = Arc::new(Shared::default());
        let waiter_shared = shared.clone();
        let (checked, checked_rx) = mpsc::channel();
        let (release, release_rx) = mpsc::channel();
        let waiter = thread::spawn(move || {
            let state = waiter_shared.state.lock().unwrap();
            assert!(!waiter_shared.stop.load(Ordering::Acquire));
            checked.send(()).unwrap();
            release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            let (_state, timeout) = waiter_shared
                .changed
                .wait_timeout_while(state, Duration::from_secs(5), |_| {
                    !waiter_shared.stop.load(Ordering::Acquire)
                })
                .unwrap();
            !timeout.timed_out()
        });
        checked_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let stop_shared = shared.clone();
        let (stopped, stopped_rx) = mpsc::channel();
        let stopper = thread::spawn(move || {
            stop_shared.stop();
            stopped.send(()).unwrap();
        });
        let premature = stopped_rx.recv_timeout(Duration::from_millis(50));
        // Always release/join before asserting so a regression cannot hang tests.
        release.send(()).unwrap();
        let woke = waiter.join().unwrap();
        stopper.join().unwrap();
        assert!(
            premature.is_err(),
            "stop bypassed the waiter's predicate lock"
        );
        assert!(woke, "worker missed the stop notification");
    }

    #[test]
    fn two_workers_overlap_and_results_keep_ticket_order() {
        let pool = PreparationPool::new().unwrap();
        let (started, arrivals) = mpsc::channel();
        let (release, gate) = mpsc::channel();
        let first_started = started.clone();
        let first = pool
            .submit(IMAGE_RESERVATION, move |_| {
                first_started.send(1).unwrap();
                gate.recv_timeout(Duration::from_secs(5)).unwrap();
                10
            })
            .unwrap();
        let second = pool
            .submit(IMAGE_RESERVATION, move |_| {
                started.send(2).unwrap();
                20
            })
            .unwrap();
        let mut seen = vec![
            arrivals.recv_timeout(Duration::from_secs(5)).unwrap(),
            arrivals.recv_timeout(Duration::from_secs(5)).unwrap(),
        ];
        seen.sort();
        assert_eq!(seen, [1, 2]);
        assert_eq!(second.wait(&|| false).unwrap(), Some(20));
        release.send(()).unwrap();
        assert_eq!(first.wait(&|| false).unwrap(), Some(10));
        assert!(pool.shared.state.lock().unwrap().peak <= BUDGET);
    }

    #[test]
    fn completed_output_retains_budget_until_consumed() {
        let pool = PreparationPool::new().unwrap();
        let (started, arrivals) = mpsc::channel();
        let first = pool.submit(BUDGET, |_| 1).unwrap();
        let second = pool
            .submit(1, move |_| {
                started.send(()).unwrap();
                2
            })
            .unwrap();
        assert!(arrivals.recv_timeout(Duration::from_millis(50)).is_err());
        assert_eq!(first.wait(&|| false).unwrap(), Some(1));
        arrivals.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(second.wait(&|| false).unwrap(), Some(2));
    }

    #[test]
    fn cancellation_signals_active_jobs_and_joins_them() {
        let pool = PreparationPool::new().unwrap();
        let (started, arrivals) = mpsc::channel();
        let (finished, completion) = mpsc::channel();
        let ticket = pool
            .submit(1, move |stop| {
                started.send(()).unwrap();
                while !stop.load(Ordering::Acquire) {
                    thread::yield_now();
                }
                finished.send(()).unwrap();
            })
            .unwrap();
        arrivals.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(ticket.wait(&|| true).unwrap().is_none());
        drop(pool);
        completion.recv_timeout(Duration::from_secs(5)).unwrap();
    }

    #[test]
    fn abandoned_queue_does_not_accumulate_or_execute() {
        let pool = PreparationPool::new().unwrap();
        let occupied = pool.submit(BUDGET, |_| 1).unwrap();
        // The occupied result holds all reservations, so these cannot run.
        for _ in 0..100 {
            drop(
                pool.submit(1, |_| panic!("abandoned job executed"))
                    .unwrap(),
            );
        }
        assert_eq!(occupied.wait(&|| false).unwrap(), Some(1));
        pool.check().unwrap();
    }

    #[test]
    fn worker_panic_is_fatal_without_retry() {
        let pool = PreparationPool::new().unwrap();
        let ticket = pool.submit(1, |_| panic!("test failure")).unwrap();
        assert!(ticket
            .wait(&|| false)
            .unwrap_err()
            .to_string()
            .contains("worker"));
    }
}
