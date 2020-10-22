
[![tests Actions Status](https://github.com/kaiserkarel/context-thread/workflows/test/badge.svg)](https://github.com/kaiserkarel/context-thread/actions)

Threads that run within a context.

Most of the time, threads that outlive the parent thread are considered a code smell.
`ctx-thread` ensures that all threads are joined before returning from the scope. Child threads
have access to the Context object, which they can use to poll the status of the thread group.
If one of the threads panics, the context is cancelled.

# Scope

This library is based on the [crossbeam](https://docs.rs/crossbeam/0.8.0/crossbeam/)'s scoped threads:

```
use ctx_thread::scope;

let people = vec![
    "Alice".to_string(),
    "Bob".to_string(),
    "Carol".to_string(),
];

scope(|ctx| {
    for person in &people {
        ctx.spawn(move |_| {
            println!("Hello, {}", person);
        });
    }
}).unwrap();
```

# Context

Aside from referring to the outer scope, threads may check the extra methods and return if
necessary:

```
use ctx_thread::scope;


scope(|ctx| {
    ctx.spawn(|ctx| {
        while ctx.active() {
            // do work
        }
    });

    ctx.spawn(|ctx| {
        ctx.cancel();
    });
}).unwrap();
```

Note that these context based cancellations are a form of cooperative scheduling. Threads
can still block even if a context expires.