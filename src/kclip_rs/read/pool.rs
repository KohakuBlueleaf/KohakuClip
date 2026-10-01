//! A persistent thread pool with one FIFO queue: clips of all submitted batches share it, so a
//! thread that finishes early starts on the next batch instead of waiting at a per-batch barrier.
//! Each thread keeps its decoders between tasks.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use crossbeam_channel::Sender;

type Task = Box<dyn FnOnce() + Send>;

pub struct Pool {
    queue: Option<Sender<Task>>,
    threads: Vec<JoinHandle<()>>,
}

impl Pool {
    pub fn new(threads: usize) -> Self {
        let (queue, tasks) = crossbeam_channel::unbounded::<Task>();
        let threads = (0..threads.max(1))
            .map(|i| {
                let tasks = tasks.clone();
                std::thread::Builder::new()
                    .name(format!("kohakuclip-{i}"))
                    .spawn(move || {
                        for task in tasks {
                            task();
                        }
                    })
                    .expect("spawn a decode thread")
            })
            .collect();
        Self {
            queue: Some(queue),
            threads,
        }
    }

    pub fn spawn(&self, task: impl FnOnce() + Send + 'static) {
        let queue = self.queue.as_ref().expect("pool is running");
        queue.send(Box::new(task)).expect("pool threads are alive");
    }
}

impl Drop for Pool {
    /// Finishes the queued tasks, then joins the threads.
    fn drop(&mut self) {
        self.queue = None;
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

/// Completion of a batch of tasks: per-task results, and a wait for all of them.
pub struct Batch<T> {
    results: Mutex<Vec<Option<T>>>,
    left: AtomicUsize,
    done: Condvar,
}

impl<T> Batch<T> {
    pub fn new(n: usize) -> Arc<Self> {
        Arc::new(Self {
            results: Mutex::new((0..n).map(|_| None).collect()),
            left: AtomicUsize::new(n),
            done: Condvar::new(),
        })
    }

    pub fn finish(&self, i: usize, result: T) {
        let mut results = self.results.lock().unwrap();
        results[i] = Some(result);
        if self.left.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.done.notify_all();
        }
    }

    /// Block until every task finished; returns their results in order.
    pub fn wait(&self) -> Vec<T> {
        let mut results = self.results.lock().unwrap();
        while self.left.load(Ordering::Acquire) > 0 {
            results = self.done.wait(results).unwrap();
        }
        results
            .iter_mut()
            .map(|r| r.take().expect("task finished"))
            .collect()
    }
}
