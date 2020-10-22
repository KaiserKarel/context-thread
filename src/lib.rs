#![doc(test(
    no_crate_inject,
    attr(
        deny(warnings, rust_2018_idioms),
        allow(dead_code, unused_assignments, unused_variables)
    )
))]
#![deny(missing_docs, missing_debug_implementations, rust_2018_idioms)]

//! Threads that run within a context.
//!
//! Most of the time, threads that outlive the parent thread are considered a code smell.
//! Ctx-thread ensures that all threads are joined before returning from the scope. Child threads
//! have access to the Context object, which they can use to poll the status of the thread group.
//! If one of the threads panics, the context is cancelled.
//!
//! # Scope
//!
//! This library is based on the [crossbeam](https://docs.rs/crossbeam/0.8.0/crossbeam/)'s scoped threads:
//!
//! ```
//! use ctx_thread::scope;
//!
//! let people = vec![
//!     "Alice".to_string(),
//!     "Bob".to_string(),
//!     "Carol".to_string(),
//! ];
//!
//! scope(|ctx| {
//!     for person in &people {
//!         ctx.spawn(move |_| {
//!             println!("Hello, {}", person);
//!         });
//!     }
//! }).unwrap();
//! ```
//!
//! # Context
//!
//! Aside from referring to the outer scope, threads may check the extra methods and return if
//! necessary:
//!
//! ```
//! use ctx_thread::scope;
//!
//!
//! scope(|ctx| {
//!     ctx.spawn(|ctx| {
//!         while ctx.active() {
//!             // do work
//!         }
//!     });
//!
//!     ctx.spawn(|ctx| {
//!         ctx.cancel();
//!     });
//! }).unwrap();
//! ```

use std::fmt;
use std::io;
use std::marker::PhantomData;
use std::mem;
use std::panic;
use std::sync::{Arc, Mutex};
use std::thread;

use cfg_if::cfg_if;
use crossbeam_utils::sync::WaitGroup;
use std::sync::atomic::{AtomicBool, Ordering};

type SharedVec<T> = Arc<Mutex<Vec<T>>>;
type SharedOption<T> = Arc<Mutex<Option<T>>>;

/// Creates a new scope for spawning threads.
///
/// All child threads that haven't been manually joined will be automatically joined just before
/// this function invocation ends. If all joined threads have successfully completed, `Ok` is
/// returned with the return value of `f`. If any of the joined threads has panicked, an `Err` is
/// returned containing errors from panicked threads.
///
/// # Examples
///
/// ```
/// use ctx_thread::scope;
///
/// let var = vec![1, 2, 3];
///
/// scope(|ctx| {
///     ctx.spawn(|_| {
///         println!("A child thread borrowing `var`: {:?}", var);
///     });
/// }).unwrap();
/// ```
pub fn scope<'env, F, R>(f: F) -> thread::Result<R>
where
    F: FnOnce(&Context<'env>) -> R,
{
    let wg = WaitGroup::new();

    let ctx = Context::<'env> {
        done: Arc::new(AtomicBool::new(false)),
        handles: SharedVec::default(),
        wait_group: wg.clone(),
        _marker: PhantomData,
    };

    // Execute the scoped function, but catch any panics.
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| f(&ctx)));

    // Signal to any remaining threads that the context is done if f panicked.
    if result.is_err() {
        ctx.cancel();
    }

    // Wait until all nested scopes are dropped.
    drop(ctx.wait_group);
    wg.wait();

    // Join all remaining spawned threads.
    let panics: Vec<_> = ctx
        .handles
        .lock()
        .unwrap()
        // Filter handles that haven't been joined, join them, and collect errors.
        .drain(..)
        .filter_map(|handle| handle.lock().unwrap().take())
        .filter_map(|handle| handle.join().err())
        .collect();

    // If `f` has panicked, resume unwinding.
    // If any of the child threads have panicked, return the panic errors.
    // Otherwise, everything is OK and return the result of `f`.
    match result {
        Err(err) => panic::resume_unwind(err),
        Ok(res) => {
            if panics.is_empty() {
                Ok(res)
            } else {
                Err(Box::new(panics))
            }
        }
    }
}

/// The context in which threads run, including their scope and thread group status.
pub struct Context<'env> {
    done: Arc<AtomicBool>,

    /// The list of the thread join handles.
    handles: SharedVec<SharedOption<thread::JoinHandle<()>>>,

    /// Used to wait until all subscopes all dropped.
    wait_group: WaitGroup,

    /// Borrows data with invariant lifetime `'env`.
    _marker: PhantomData<&'env mut &'env ()>,
}

unsafe impl Sync for Context<'_> {}

impl<'env> Context<'env> {
    /// Check if the current context has finished. Threads performing work should regularly check
    /// and return early if cancellation has been signalled. Usually this indicates some critical
    /// failure in a sibling thread, thus making the result of the current thread inconsequential.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ctx_thread::scope;
    ///
    /// scope(|ctx| {
    ///     ctx.spawn(|ctx| {
    ///         assert_eq!(ctx.active(), !ctx.done());
    ///         ctx.spawn(|ctx| {
    ///            ctx.cancel()
    ///         });
    ///
    ///         while ctx.active() {}
    ///     });
    /// }).unwrap();
    /// ```
    pub fn done(&self) -> bool {
        self.done.load(Ordering::Relaxed)
    }

    /// Signals cancellation of the current context, causing [done] to return true. A cancelled
    /// context cannot be re-enabled.
    /// [done]: Context::done
    pub fn cancel(&self) {
        self.done.store(true, Ordering::Relaxed)
    }

    /// Alias for !ctx.done(); which is easier on the eyes in for loops.
    pub fn active(&self) -> bool {
        !self.done()
    }

    /// Spawns a scoped thread, providing a derived context.
    ///
    /// This method is similar to the [`spawn`] function in Rust's standard library. The difference
    /// is that this thread is scoped, meaning it's guaranteed to terminate before the scope exits,
    /// allowing it to reference variables outside the scope.
    ///
    /// The scoped thread is passed a reference to this scope as an argument, which can be used for
    /// spawning nested threads.
    ///
    /// The returned [handle](ContextJoinHandle) can be used to manually
    /// [join](ContextJoinHandle::join) the thread before the scope exits.
    ///
    /// This will create a thread using default parameters of [`ScopedThreadBuilder`], if you want to specify the
    /// stack size or the name of the thread, use this API instead.
    ///
    /// [`spawn`]: std::thread::spawn
    ///
    /// # Panics
    ///
    /// Panics if the OS fails to create a thread; use [`ScopedThreadBuilder::spawn`]
    /// to recover from such errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use ctx_thread::scope;
    ///
    /// scope(|ctx| {
    ///     let handle = ctx.spawn(|_| {
    ///         println!("A child thread is running");
    ///         42
    ///     });
    ///
    ///     // Join the thread and retrieve its result.
    ///     let res = handle.join().unwrap();
    ///     assert_eq!(res, 42);
    /// }).unwrap();
    /// ```
    pub fn spawn<'scope, F, T>(&'scope self, f: F) -> ContextJoinHandle<'scope, T>
    where
        F: FnOnce(&Context<'env>) -> T,
        F: Send + 'env,
        T: Send + 'env,
    {
        self.builder()
            .spawn(|ctx| {
                let result = panic::catch_unwind(panic::AssertUnwindSafe(|| f(ctx)));
                if let Err(e) = result {
                    ctx.cancel();
                    panic::resume_unwind(e)
                }

                result.unwrap()
            })
            .expect("failed to spawn scoped thread")
    }

    /// Creates a builder that can configure a thread before spawning.
    ///
    /// # Examples
    ///
    /// ```
    /// use ctx_thread::scope;
    ///
    /// scope(|ctx| {
    ///     ctx.builder()
    ///         .name(String::from("child"))
    ///         .stack_size(1024)
    ///         .spawn(|_| println!("A child thread is running"))
    ///         .unwrap();
    /// }).unwrap();
    /// ```
    pub fn builder<'scope>(&'scope self) -> ContextThreadBuilder<'scope, 'env> {
        ContextThreadBuilder {
            scope: self,
            builder: thread::Builder::new(),
        }
    }
}

impl fmt::Debug for Context<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad("Scope { .. }")
    }
}

/// Configures the properties of a new thread.
///
/// The two configurable properties are:
///
/// - [`name`]: Specifies an [associated name for the thread][naming-threads].
/// - [`stack_size`]: Specifies the [desired stack size for the thread][stack-size].
///
/// The [`spawn`] method will take ownership of the builder and return an [`io::Result`] of the
/// thread handle with the given configuration.
///
/// The [`Context::spawn`] method uses a builder with default configuration and unwraps its return
/// value. You may want to use this builder when you want to recover from a failure to launch a
/// thread.
///
/// # Examples
///
/// ```
/// use ctx_thread::scope;
///
/// scope(|ctx| {
///     ctx.builder()
///         .spawn(|_| println!("Running a child thread"))
///         .unwrap();
/// }).unwrap();
/// ```
///
/// [`name`]: ContextThreadBuilder::name
/// [`stack_size`]: ContextThreadBuilder::stack_size
/// [`spawn`]: ContextThreadBuilder::spawn
/// [`io::Result`]: std::io::Result
/// [naming-threads]: std::thread#naming-threads
/// [stack-size]: std::thread#stack-size
#[derive(Debug)]
pub struct ContextThreadBuilder<'scope, 'env> {
    scope: &'scope Context<'env>,
    builder: thread::Builder,
}

impl<'scope, 'env> ContextThreadBuilder<'scope, 'env> {
    /// Sets the name for the new thread.
    ///
    /// The name must not contain null bytes (`\0`).
    ///
    /// For more information about named threads, see [here][naming-threads].
    pub fn name(mut self, name: String) -> ContextThreadBuilder<'scope, 'env> {
        self.builder = self.builder.name(name);
        self
    }

    /// Sets the size of the stack for the new thread.
    ///
    /// The stack size is measured in bytes.
    ///
    /// For more information about the stack size for threads, see [here][stack-size].
    pub fn stack_size(mut self, size: usize) -> ContextThreadBuilder<'scope, 'env> {
        self.builder = self.builder.stack_size(size);
        self
    }

    /// Spawns a scoped thread with this configuration, providing a derived context.
    ///
    /// The scoped thread is passed a reference to this scope as an argument, which can be used for
    /// spawning nested threads.
    ///
    /// The returned handle can be used to manually join the thread before the scope exits.
    ///
    /// # Errors
    ///
    /// Unlike the [`Scope::spawn`] method, this method yields an
    /// [`io::Result`] to capture any failure to create the thread at
    /// the OS level.
    ///
    /// [`io::Result`]: std::io::Result
    ///
    /// # Panics
    ///
    /// Panics if a thread name was set and it contained null bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use ctx_thread::scope;
    ///
    /// scope(|ctx| {
    ///     let handle = ctx.builder()
    ///         .spawn(|_| {
    ///             println!("A child thread is running");
    ///             42
    ///         })
    ///         .unwrap();
    ///
    ///     // Join the thread and retrieve its result.
    ///     let res = handle.join().unwrap();
    ///     assert_eq!(res, 42);
    /// }).unwrap();
    /// ```
    pub fn spawn<F, T>(self, f: F) -> io::Result<ContextJoinHandle<'scope, T>>
    where
        F: FnOnce(&Context<'env>) -> T,
        F: Send + 'env,
        T: Send + 'env,
    {
        // The result of `f` will be stored here.
        let result = SharedOption::default();

        // Spawn the thread and grab its join handle and thread handle.
        let (handle, thread) = {
            let result = Arc::clone(&result);

            // A clone of the context that will be moved into the new thread.
            let ctx = Context::<'env> {
                done: self.scope.done.clone(),
                handles: Arc::clone(&self.scope.handles),
                wait_group: self.scope.wait_group.clone(),
                _marker: PhantomData,
            };

            // Spawn the thread.
            let handle = {
                let closure = move || {
                    // Make sure the scope is inside the closure with the proper `'env` lifetime.
                    let scope: Context<'env> = ctx;

                    // Run the closure.
                    let res = f(&scope);

                    // Store the result if the closure didn't panic.
                    *result.lock().unwrap() = Some(res);
                };

                // Allocate `closure` on the heap and erase the `'env` bound.
                let closure: Box<dyn FnOnce() + Send + 'env> = Box::new(closure);
                let closure: Box<dyn FnOnce() + Send + 'static> =
                    unsafe { mem::transmute(closure) };

                // Finally, spawn the closure.
                self.builder.spawn(move || closure())?
            };

            let thread = handle.thread().clone();
            let handle = Arc::new(Mutex::new(Some(handle)));
            (handle, thread)
        };

        // Add the handle to the shared list of join handles.
        self.scope.handles.lock().unwrap().push(Arc::clone(&handle));

        Ok(ContextJoinHandle {
            handle,
            result,
            thread,
            _marker: PhantomData,
        })
    }
}

unsafe impl<T> Send for ContextJoinHandle<'_, T> {}
unsafe impl<T> Sync for ContextJoinHandle<'_, T> {}

/// A handle that can be used to join its context thread.
///
/// This struct is created by the [`Context::spawn`] method and the
/// [`ContextJoinHandle::spawn`] method.
pub struct ContextJoinHandle<'scope, T> {
    /// A join handle to the spawned thread.
    handle: SharedOption<thread::JoinHandle<()>>,

    /// Holds the result of the inner closure.
    result: SharedOption<T>,

    /// A handle to the the spawned thread.
    thread: thread::Thread,

    /// Borrows the parent scope with lifetime `'scope`.
    _marker: PhantomData<&'scope ()>,
}

impl<T> ContextJoinHandle<'_, T> {
    /// Waits for the thread to finish and returns its result.
    ///
    /// If the child thread panics, an error is returned.
    ///
    /// # Panics
    ///
    /// This function may panic on some platforms if a thread attempts to join itself or otherwise
    /// may create a deadlock with joining threads.
    ///
    /// # Examples
    ///
    /// ```
    /// use ctx_thread::scope;
    ///
    /// scope(|ctx| {
    ///     let handle1 = ctx.spawn(|_| println!("I'm a happy thread :)"));
    ///     let handle2 = ctx.spawn(|_| panic!("I'm a sad thread :("));
    ///
    ///     // Join the first thread and verify that it succeeded.
    ///     let res = handle1.join();
    ///     assert!(res.is_ok());
    ///
    ///     // Join the second thread and verify that it panicked.
    ///     let res = handle2.join();
    ///     assert!(res.is_err());
    /// }).unwrap();
    /// ```
    pub fn join(self) -> thread::Result<T> {
        // Take out the handle. The handle will surely be available because the root scope waits
        // for nested scopes before joining remaining threads.
        let handle = self.handle.lock().unwrap().take().unwrap();

        // Join the thread and then take the result out of its inner closure.
        handle
            .join()
            .map(|()| self.result.lock().unwrap().take().unwrap())
    }

    /// Returns a handle to the underlying thread.
    pub fn thread(&self) -> &thread::Thread {
        &self.thread
    }
}

cfg_if! {
    if #[cfg(unix)] {
        use std::os::unix::thread::{JoinHandleExt, RawPthread};

        impl<T> JoinHandleExt for ContextJoinHandle<'_, T> {
            fn as_pthread_t(&self) -> RawPthread {
                // Borrow the handle. The handle will surely be available because the root scope waits
                // for nested scopes before joining remaining threads.
                let handle = self.handle.lock().unwrap();
                handle.as_ref().unwrap().as_pthread_t()
            }
            fn into_pthread_t(self) -> RawPthread {
                self.as_pthread_t()
            }
        }
    } else if #[cfg(windows)] {
        use std::os::windows::io::{AsRawHandle, IntoRawHandle, RawHandle};

        impl<T> AsRawHandle for ContextJoinHandle<'_, T> {
            fn as_raw_handle(&self) -> RawHandle {
                // Borrow the handle. The handle will surely be available because the root scope waits
                // for nested scopes before joining remaining threads.
                let handle = self.handle.lock().unwrap();
                handle.as_ref().unwrap().as_raw_handle()
            }
        }

        #[cfg(windows)]
        impl<T> IntoRawHandle for ContextJoinHandle<'_, T> {
            fn into_raw_handle(self) -> RawHandle {
                self.as_raw_handle()
            }
        }
    }
}

impl<T> fmt::Debug for ContextJoinHandle<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(&format!(
            "ScopedJoinHandle {{ name: {:?} }}",
            self.thread.name()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cancellation_nested() {
        scope(|ctx| {
            ctx.spawn(|ctx| while !ctx.done() {});

            ctx.spawn(|ctx| {
                while ctx.active() {
                    ctx.spawn(|ctx| ctx.cancel());
                }
            });
        })
        .unwrap()
    }

    #[test]
    #[should_panic]
    fn test_panic_cancellation() {
        scope(|ctx| {
            ctx.spawn(|_| panic!());
            assert!(ctx.active())
        })
        .unwrap()
    }
}
