//! Serial admission and a condition-variable reaper, independent of ONNX assets.
use anyhow::{ensure, Context, Result};
use std::{
    sync::{Arc, Condvar, Mutex},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdlePolicy {
    KeepWarm,
    ReleaseAfter(Duration),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeIdlePolicy {
    pub text: IdlePolicy,
    pub vision: IdlePolicy,
    pub audio: IdlePolicy,
}

impl Default for RuntimeIdlePolicy {
    fn default() -> Self {
        Self {
            text: IdlePolicy::ReleaseAfter(Duration::from_secs(15 * 60)),
            vision: IdlePolicy::ReleaseAfter(Duration::from_secs(5 * 60)),
            audio: IdlePolicy::ReleaseAfter(Duration::from_secs(5 * 60)),
        }
    }
}

impl RuntimeIdlePolicy {
    pub const KEEP_WARM: Self = Self {
        text: IdlePolicy::KeepWarm,
        vision: IdlePolicy::KeepWarm,
        audio: IdlePolicy::KeepWarm,
    };

    fn policies(self) -> [IdlePolicy; 3] {
        [self.text, self.vision, self.audio]
    }

    pub(super) fn validate(self) -> Result<()> {
        for policy in self.policies() {
            if let IdlePolicy::ReleaseAfter(duration) = policy {
                ensure!(
                    !duration.is_zero(),
                    "Model idle release duration must be positive"
                );
            }
        }
        Ok(())
    }
}

pub(super) const TEXT: [bool; 3] = [true, false, false];
pub(super) const VISION: [bool; 3] = [true, true, false];
pub(super) const AUDIO: [bool; 3] = [true, false, true];
pub(super) const NONE: [bool; 3] = [false; 3];

pub(super) trait Resources: Send + 'static {
    fn loaded(&self) -> [bool; 3];
    fn release(&mut self, modality: usize);
    fn release_enabled(&self) -> bool {
        true
    }
}

struct State<T> {
    resources: T,
    policy: RuntimeIdlePolicy,
    last_used: [Option<Instant>; 3],
}

impl<T: Resources> State<T> {
    fn finish(&mut self, used: [bool; 3], now: Instant) {
        let loaded = self.resources.loaded();
        for index in 0..3 {
            if loaded[index] && (used[index] || self.last_used[index].is_none()) {
                self.last_used[index] = Some(now);
            }
        }
    }

    /// Called only with admission locked and no active or queued operation.
    fn reap(&mut self, now: Instant) -> Option<Duration> {
        if !self.resources.release_enabled() {
            return None;
        }
        let mut remaining: Option<Duration> = None;
        for (index, policy) in self.policy.policies().into_iter().enumerate() {
            if !self.resources.loaded()[index] {
                self.last_used[index] = None;
                continue;
            }
            let (IdlePolicy::ReleaseAfter(timeout), Some(last_used)) =
                (policy, self.last_used[index])
            else {
                continue;
            };
            let idle = now.saturating_duration_since(last_used);
            if idle >= timeout {
                self.resources.release(index);
                self.last_used[index] = None;
            } else {
                let until_release = timeout - idle;
                remaining = Some(remaining.map_or(until_release, |next| next.min(until_release)));
            }
        }
        remaining
    }
}

#[derive(Default)]
struct Admission {
    pending: usize,
    shutdown: bool,
}

struct Shared<T> {
    admission: Mutex<Admission>,
    changed: Condvar,
    state: Mutex<State<T>>,
}

/// The lock order is admission -> state. Operations release admission before
/// acquiring state and drop state before completing admission. Reaping holds
/// admission throughout the unload, so an arriving operation cannot race the
/// decision to unload. All accepted operations run, including on inference errors.
pub(super) struct RuntimeOwner<T: Resources> {
    shared: Arc<Shared<T>>,
    reaper: Option<JoinHandle<()>>,
}

struct Admitted<'a, T>(&'a Shared<T>);

impl<T> Drop for Admitted<'_, T> {
    fn drop(&mut self) {
        // This guard also drains admission when an operation unwinds. Poisoned
        // resource state is still reported by the next public operation.
        let mut admission = self.0.admission.lock().expect("Runtime admission poisoned");
        admission.pending -= 1;
        self.0.changed.notify_all();
    }
}

impl<T: Resources> RuntimeOwner<T> {
    pub(super) fn new(resources: T, policy: RuntimeIdlePolicy) -> Result<Self> {
        policy.validate()?;
        let shared = Arc::new(Shared {
            admission: Mutex::new(Admission::default()),
            changed: Condvar::new(),
            state: Mutex::new(State {
                resources,
                policy,
                last_used: [None; 3],
            }),
        });
        let reaper_shared = Arc::clone(&shared);
        let reaper = thread::Builder::new()
            .name("fractal-model-reaper".into())
            .spawn(move || reap_loop(reaper_shared))
            .context("Start model idle-release timer")?;
        Ok(Self {
            shared,
            reaper: Some(reaper),
        })
    }

    fn admit(&self) -> Result<Admitted<'_, T>> {
        let mut admission = self
            .shared
            .admission
            .lock()
            .map_err(|_| anyhow::anyhow!("Runtime admission poisoned; restart the service"))?;
        ensure!(
            !admission.shutdown,
            "Runtime has stopped accepting operations"
        );
        admission.pending += 1;
        Ok(Admitted(&self.shared))
    }

    pub(super) fn execute<R>(
        &self,
        used: [bool; 3],
        operation: impl FnOnce(&mut T) -> Result<R>,
    ) -> Result<R> {
        let _admitted = self.admit()?;
        let mut state = self
            .shared
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("Runtime state poisoned; restart the service"))?;
        let result = operation(&mut state.resources);
        // Timers start at completion, including failed work that loaded a session.
        // A long inference is never itself counted as idle time.
        state.finish(used, Instant::now());
        result
    }

    pub(super) fn set_policy(&self, policy: RuntimeIdlePolicy) -> Result<()> {
        policy.validate()?;
        let _admitted = self.admit()?;
        let mut state = self
            .shared
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("Runtime state poisoned; restart the service"))?;
        state.policy = policy;
        Ok(())
    }

    pub(super) fn inspect<R>(
        &self,
        inspect: impl FnOnce(&T, RuntimeIdlePolicy, [Option<Instant>; 3]) -> R,
    ) -> Result<R> {
        let _admitted = self.admit()?;
        let state = self
            .shared
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("Runtime state poisoned; restart the service"))?;
        Ok(inspect(&state.resources, state.policy, state.last_used))
    }
}

fn reap_loop<T: Resources>(shared: Arc<Shared<T>>) {
    let mut admission = shared.admission.lock().expect("Runtime admission poisoned");
    loop {
        if admission.shutdown {
            return;
        }
        if admission.pending != 0 {
            admission = shared
                .changed
                .wait(admission)
                .expect("Runtime admission poisoned");
            continue;
        }
        let remaining = match shared.state.lock() {
            Ok(mut state) => state.reap(Instant::now()),
            Err(_) => {
                eprintln!(
                    "[Model] Idle reaper stopped: runtime state poisoned; restart the service"
                );
                return;
            }
        };
        admission = match remaining {
            Some(duration) => {
                shared
                    .changed
                    .wait_timeout(admission, duration)
                    .expect("Runtime admission poisoned")
                    .0
            }
            None => shared
                .changed
                .wait(admission)
                .expect("Runtime admission poisoned"),
        };
    }
}

impl<T: Resources> Drop for RuntimeOwner<T> {
    fn drop(&mut self) {
        {
            let mut admission = self
                .shared
                .admission
                .lock()
                .expect("Runtime admission poisoned");
            admission.shutdown = true;
            self.shared.changed.notify_all();
        }
        if let Some(reaper) = self.reaper.take() {
            if reaper.join().is_err() {
                eprintln!("[Model] Idle reaper failed during shutdown");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc,
    };

    #[derive(Default)]
    struct FakeResources {
        loaded: [bool; 3],
        loads: [usize; 3],
        released: Option<mpsc::Sender<usize>>,
    }

    impl Resources for FakeResources {
        fn loaded(&self) -> [bool; 3] {
            self.loaded
        }
        fn release(&mut self, modality: usize) {
            self.loaded[modality] = false;
            if let Some(sender) = &self.released {
                let _ = sender.send(modality);
            }
        }
    }

    impl FakeResources {
        fn load(&mut self, modality: usize) {
            if !self.loaded[modality] {
                self.loaded[modality] = true;
                self.loads[modality] += 1;
            }
        }
    }

    #[test]
    fn independent_timers_keep_text_warm_and_release_unused_encoders() {
        let start = Instant::now();
        let mut state = State {
            resources: FakeResources {
                loaded: [true; 3],
                ..Default::default()
            },
            policy: RuntimeIdlePolicy {
                text: IdlePolicy::KeepWarm,
                vision: IdlePolicy::ReleaseAfter(Duration::from_secs(5)),
                audio: IdlePolicy::ReleaseAfter(Duration::from_secs(9)),
            },
            last_used: [Some(start); 3],
        };
        assert_eq!(
            state.reap(start + Duration::from_secs(5)),
            Some(Duration::from_secs(4))
        );
        assert_eq!(state.resources.loaded, [true, false, true]);
        state.finish(TEXT, start + Duration::from_secs(8));
        assert_eq!(state.reap(start + Duration::from_secs(9)), None);
        assert_eq!(state.resources.loaded, [true, false, false]);
        assert_eq!(state.last_used[0], Some(start + Duration::from_secs(8)));
    }

    #[test]
    fn actual_timer_releases_without_another_request_and_reloads_once() {
        let (sender, receiver) = mpsc::channel();
        let owner = RuntimeOwner::new(
            FakeResources {
                released: Some(sender),
                ..Default::default()
            },
            RuntimeIdlePolicy {
                text: IdlePolicy::ReleaseAfter(Duration::from_millis(25)),
                ..RuntimeIdlePolicy::KEEP_WARM
            },
        )
        .unwrap();
        owner
            .execute(TEXT, |resources| {
                resources.load(0);
                Ok(())
            })
            .unwrap();
        assert_eq!(receiver.recv_timeout(Duration::from_secs(5)).unwrap(), 0);
        owner.set_policy(RuntimeIdlePolicy::KEEP_WARM).unwrap();
        owner
            .execute(TEXT, |resources| {
                resources.load(0);
                Ok(())
            })
            .unwrap();
        owner
            .execute(TEXT, |resources| {
                resources.load(0);
                Ok(())
            })
            .unwrap();
        assert_eq!(owner.inspect(|r, _, _| r.loads[0]).unwrap(), 2);
    }

    #[test]
    fn active_and_queued_work_is_serial_and_cannot_be_reaped() {
        let (released_tx, released_rx) = mpsc::channel();
        let owner = Arc::new(
            RuntimeOwner::new(
                FakeResources {
                    released: Some(released_tx),
                    ..Default::default()
                },
                RuntimeIdlePolicy {
                    text: IdlePolicy::ReleaseAfter(Duration::from_millis(10)),
                    ..RuntimeIdlePolicy::KEEP_WARM
                },
            )
            .unwrap(),
        );
        let (started_tx, started_rx) = mpsc::channel();
        let (finish_tx, finish_rx) = mpsc::channel();
        let active = Arc::clone(&owner);
        let first = thread::spawn(move || {
            active.execute(TEXT, |r| {
                r.load(0);
                started_tx.send(()).unwrap();
                finish_rx.recv().unwrap();
                Ok(())
            })
        });
        started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let completed = Arc::new(AtomicUsize::new(0));
        let mut queued = Vec::new();
        for _ in 0..4 {
            let owner = Arc::clone(&owner);
            let completed = Arc::clone(&completed);
            queued.push(thread::spawn(move || {
                owner.execute(TEXT, |r| {
                    r.load(0);
                    completed.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                })
            }));
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        while owner.shared.admission.lock().unwrap().pending != 5 {
            assert!(
                Instant::now() < deadline,
                "Queued operations were not admitted"
            );
            thread::yield_now();
        }
        assert!(released_rx.recv_timeout(Duration::from_millis(30)).is_err());
        assert_eq!(completed.load(Ordering::SeqCst), 0);
        finish_tx.send(()).unwrap();
        first.join().unwrap().unwrap();
        for job in queued {
            job.join().unwrap().unwrap();
        }
        assert_eq!(completed.load(Ordering::SeqCst), 4);
        assert_eq!(owner.inspect(|r, _, _| r.loads[0]).unwrap(), 1);
        assert_eq!(released_rx.recv_timeout(Duration::from_secs(5)).unwrap(), 0);
    }

    #[test]
    fn errors_drain_admission_and_new_policy_wakes_the_timer() {
        let (sender, receiver) = mpsc::channel();
        let owner = RuntimeOwner::new(
            FakeResources {
                released: Some(sender),
                ..Default::default()
            },
            RuntimeIdlePolicy::KEEP_WARM,
        )
        .unwrap();
        assert!(owner
            .execute(TEXT, |r| -> Result<()> {
                r.load(0);
                anyhow::bail!("test inference failure")
            })
            .is_err());
        assert_eq!(owner.shared.admission.lock().unwrap().pending, 0);
        owner
            .set_policy(RuntimeIdlePolicy {
                text: IdlePolicy::ReleaseAfter(Duration::from_millis(1)),
                ..RuntimeIdlePolicy::KEEP_WARM
            })
            .unwrap();
        assert_eq!(receiver.recv_timeout(Duration::from_secs(5)).unwrap(), 0);
        assert!(owner
            .set_policy(RuntimeIdlePolicy {
                text: IdlePolicy::ReleaseAfter(Duration::ZERO),
                ..RuntimeIdlePolicy::KEEP_WARM
            })
            .is_err());
    }
}
