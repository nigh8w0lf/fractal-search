//! Filesystem notifications enqueue rescans; only the main worker touches models/SQLite.
use anyhow::{Context, Result};
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

const DEBOUNCE: Duration = Duration::from_millis(700);
const MAX_ERRORS: usize = 32;

#[derive(Default)]
struct Pending {
    // At most one entry per registered source, regardless of event volume.
    dirty: BTreeMap<i64, Instant>,
    errors: VecDeque<String>,
    errors_omitted: usize,
}
impl Pending {
    fn event(&mut self, roots: &[(i64, PathBuf)], event: &Event, now: Instant) {
        // Reads from inference and previews must not trigger another scan.
        if matches!(event.kind, EventKind::Access(_)) {
            return;
        }
        for (id, root) in roots {
            if event.need_rescan()
                || event
                    .paths
                    .iter()
                    .any(|path| path.starts_with(root) || root.starts_with(path))
            {
                self.dirty.insert(*id, now);
            }
        }
    }
}

pub struct DirectoryWatcher {
    _watcher: RecommendedWatcher,
    roots: Vec<(i64, PathBuf)>,
    pending: Arc<Mutex<Pending>>,
}
impl DirectoryWatcher {
    pub fn new(roots: Vec<(i64, PathBuf)>) -> Result<Self> {
        let pending = Arc::new(Mutex::new(Pending::default()));
        let callback_pending = pending.clone();
        let callback_roots = roots.clone();
        let mut watcher = notify::recommended_watcher(move |event: notify::Result<Event>| {
            let mut pending = match callback_pending.lock() {
                Ok(pending) => pending,
                Err(error) => {
                    eprintln!("[Watch] Notification state poisoned: {error}");
                    return;
                }
            };
            match event {
                Ok(event) => pending.event(&callback_roots, &event, Instant::now()),
                Err(error) => {
                    let message = format!("Filesystem watcher failed: {error}");
                    eprintln!("[Watch] {message}");
                    if pending.errors.len() < MAX_ERRORS {
                        pending.errors.push_back(message);
                    } else {
                        pending.errors_omitted += 1;
                    }
                }
            }
        })
        .context("Initialize native filesystem watcher")?;
        // A watch on the parent reports deletion/renaming of the selected root
        // itself; watching only the root does not cover that on every platform.
        let parents: BTreeSet<_> = roots
            .iter()
            .filter_map(|(_, root)| root.parent().map(PathBuf::from))
            .collect();
        for parent in parents {
            watcher
                .watch(&parent, RecursiveMode::NonRecursive)
                .with_context(|| format!("Watch source parent {parent:?}"))?;
        }
        for (_, root) in &roots {
            watcher
                .watch(root, RecursiveMode::Recursive)
                .with_context(|| format!("Watch indexed folder {root:?}"))?;
        }
        Ok(Self {
            _watcher: watcher,
            roots,
            pending,
        })
    }
    pub fn roots(&self) -> &[(i64, PathBuf)] {
        &self.roots
    }

    fn pending(&self) -> Result<std::sync::MutexGuard<'_, Pending>> {
        self.pending
            .lock()
            .map_err(|_| anyhow::anyhow!("Filesystem watcher state poisoned"))
    }
    pub fn inherit_pending(&self, old: &Self) -> Result<()> {
        let mut old_pending = old.pending()?;
        let mut new_pending = self.pending()?;
        for (id, time) in &old_pending.dirty {
            if self.roots.iter().any(|root| root.0 == *id) {
                let entry = new_pending.dirty.entry(*id).or_insert(*time);
                *entry = (*entry).max(*time);
            }
        }
        while let Some(error) = old_pending.errors.pop_front() {
            if new_pending.errors.len() < MAX_ERRORS {
                new_pending.errors.push_back(error);
            } else {
                new_pending.errors_omitted += 1;
            }
        }
        new_pending.errors_omitted += old_pending.errors_omitted;
        Ok(())
    }
    pub fn take_due(&self) -> Result<Option<(i64, PathBuf)>> {
        let mut pending = self.pending()?;
        let now = Instant::now();
        let due = pending
            .dirty
            .iter()
            .find_map(|(id, time)| (now.duration_since(*time) >= DEBOUNCE).then_some(*id));
        if let Some(id) = due {
            pending.dirty.remove(&id);
            let root = self
                .roots
                .iter()
                .find(|root| root.0 == id)
                .context("Dirty source missing from watcher")?;
            return Ok(Some(root.clone()));
        }
        Ok(None)
    }
    pub fn reschedule(&self, source_id: i64) -> Result<()> {
        // A manual request can win admission after take_due. Retain the work,
        // preserving any newer event's debounce, rather than losing a rescan.
        self.pending()?
            .dirty
            .entry(source_id)
            .or_insert(Instant::now() - DEBOUNCE);
        Ok(())
    }
    pub fn take_errors(&self) -> Result<Vec<String>> {
        let mut pending = self.pending()?;
        let mut errors: Vec<_> = pending.errors.drain(..).collect();
        if pending.errors_omitted > 0 {
            errors.push(format!(
                "{} additional filesystem watcher errors; full errors in backend log",
                pending.errors_omitted
            ));
            pending.errors_omitted = 0;
        }
        Ok(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use notify::event::{AccessKind, ModifyKind};
    #[test]
    fn reads_do_not_rescan_and_changes_are_coalesced_per_source() {
        let roots = vec![
            (1, PathBuf::from("/photos/a")),
            (2, PathBuf::from("/photos/b")),
        ];
        let mut pending = Pending::default();
        let now = Instant::now();
        pending.event(
            &roots,
            &Event::new(EventKind::Access(AccessKind::Read)).add_path("/photos/a/image.png".into()),
            now,
        );
        pending.event(
            &roots,
            &Event::new(EventKind::Modify(ModifyKind::Any)).add_path("/unrelated/image.png".into()),
            now,
        );
        assert!(pending.dirty.is_empty());
        let event =
            Event::new(EventKind::Modify(ModifyKind::Any)).add_path("/photos/a/image.png".into());
        pending.event(&roots, &event, now);
        pending.event(&roots, &event, now + Duration::from_millis(10));
        assert_eq!(pending.dirty.len(), 1);
        assert_eq!(pending.dirty[&1], now + Duration::from_millis(10));
        // A selected root disappearing/being renamed is relevant too.
        pending.event(
            &roots,
            &Event::new(EventKind::Remove(notify::event::RemoveKind::Folder))
                .add_path("/photos/b".into()),
            now,
        );
        assert_eq!(pending.dirty.len(), 2);
    }
}
