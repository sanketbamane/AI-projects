const QUESTIONS_BANK = {
  behavioral: [
    { q: "Tell me about yourself.", a: "Structure your answer using Present-Past-Future: Start with your current role, mention key past experiences, then explain why you're excited about this opportunity." },
    { q: "What is your greatest strength?", a: "Choose a strength relevant to the role. Give a specific example (STAR method) showing it in action, and tie it to impact." },
    { q: "What is your greatest weakness?", a: "Choose a real (but not critical) weakness. Show self-awareness and describe concrete steps you've taken to improve." },
    { q: "Tell me about a time you failed.", a: "Use STAR method. Focus on what you learned and how you applied that lesson — employers value growth mindset." },
    { q: "Why do you want to leave your current job?", a: "Stay positive. Focus on growth opportunities, not negatives about your current employer. 'I'm looking for a new challenge' works well." },
    { q: "Where do you see yourself in 5 years?", a: "Show ambition aligned with the company's path. Mention skill growth, leadership, or domain expertise — not 'your job'." },
    { q: "Describe a time you worked under pressure.", a: "Use a real example with STAR. Highlight your process: how you prioritized, communicated, and delivered despite the pressure." },
    { q: "How do you handle conflict with a teammate?", a: "Show maturity: listen first, seek to understand, address privately, focus on the issue not the person, involve management only if needed." },
    { q: "What motivates you?", a: "Be authentic. Common strong answers: solving complex problems, seeing user impact, learning, mentoring others, or building something new." },
    { q: "Why should we hire you?", a: "Summarize your top 3 strengths that directly match the job requirements. End with your enthusiasm for the specific role/company." }
  ],
  javascript: [
    { q: "What is the difference between `let`, `var`, and `const`?", a: "`var` is function-scoped and hoisted. `let` and `const` are block-scoped. `const` is immutable (the binding, not the value for objects). Prefer `const` by default, `let` when reassignment is needed." },
    { q: "Explain closures in JavaScript.", a: "A closure is a function that retains access to its lexical scope even when executed outside that scope. Created every time a function is created. Used in data encapsulation, memoization, and callbacks." },
    { q: "What is the event loop?", a: "JS is single-threaded. The event loop checks the call stack; when empty, it pushes tasks from the callback queue. Microtasks (Promises) run before macrotasks (setTimeout)." },
    { q: "What is `this` in JavaScript?", a: "`this` refers to the execution context. In a method, it's the object. In a function, it's the global object (or `undefined` in strict mode). Arrow functions inherit `this` from their enclosing scope." },
    { q: "Explain Promise vs async/await.", a: "Promises represent future values with `.then()/.catch()`. `async/await` is syntactic sugar over Promises making async code look synchronous. Both handle the same underlying mechanics." },
    { q: "What is prototypal inheritance?", a: "Objects in JS inherit properties via a prototype chain. Each object has a `[[Prototype]]` link to another object. You can set it via `Object.create()`, classes (internally the same), or `Object.setPrototypeOf()`." },
    { q: "What are higher-order functions?", a: "Functions that take other functions as arguments or return them. Examples: `map`, `filter`, `reduce`. Core to functional programming in JS." },
    { q: "What is the difference between `==` and `===`?", a: "`==` checks equality with type coercion (e.g. `1 == '1'` is true). `===` checks value AND type without coercion. Always prefer `===` to avoid bugs." },
    { q: "What is debouncing and throttling?", a: "Debouncing delays a function call until after a pause (e.g. search input). Throttling limits calls to once per time interval (e.g. scroll handler). Both optimize performance." },
    { q: "How does `Array.prototype.reduce` work?", a: "`reduce(callback, initialValue)` iterates the array, accumulating a single result. The callback gets `(accumulator, currentValue, index, array)`. Great for summing, grouping, or transforming arrays." }
  ],
  dsa: [
    { q: "What is the time complexity of binary search?", a: "O(log n) — each step halves the search space. Requires a sorted array. Space complexity is O(1) for iterative, O(log n) for recursive (call stack)." },
    { q: "Explain the difference between a stack and a queue.", a: "Stack: LIFO (Last In First Out) — push/pop from the same end. Queue: FIFO (First In First Out) — enqueue at back, dequeue from front. Both O(1) operations with proper implementation." },
    { q: "What is a hash table and how does it work?", a: "Maps keys to values using a hash function. O(1) average for insert/lookup/delete. Handles collisions via chaining or open addressing. Worst case O(n) with many collisions." },
    { q: "What is dynamic programming?", a: "Optimization technique: break problem into overlapping sub-problems, solve each once, store results (memoization or tabulation). Key: optimal substructure + overlapping subproblems." },
    { q: "Explain BFS vs DFS.", a: "BFS uses a queue, explores level by level — great for shortest path. DFS uses a stack/recursion, goes deep first — great for exploring all paths, cycle detection, topological sort." },
    { q: "What is a balanced binary search tree?", a: "A BST where height is O(log n). Examples: AVL tree, Red-Black tree. Guarantees O(log n) for search, insert, delete. Prevents degradation to O(n) in worst case." },
    { q: "What is the sliding window technique?", a: "Maintains a window of elements that moves through an array. Reduces O(n²) brute-force to O(n). Used for: max subarray sum, longest substring without repeat, etc." },
    { q: "What is the two-pointer technique?", a: "Use two indices moving through an array, often from both ends toward center. Used for pair sum problems, removing duplicates, palindrome check. Usually O(n) time, O(1) space." },
    { q: "Explain quicksort.", a: "Divide & conquer: pick a pivot, partition array into elements < pivot and > pivot, recursively sort both halves. Average O(n log n), worst O(n²) with bad pivot. In-place." },
    { q: "What is memoization?", a: "Caching function results so repeated calls with same inputs return instantly. Trades memory for speed. Key optimization for recursive DP solutions. Can convert O(2^n) to O(n)." }
  ],
  systemDesign: [
    { q: "How would you design a URL shortener (like bit.ly)?", a: "Key components: Hash function (MD5/custom base62), DB to store long↔short mapping, redirect service (301/302), analytics, cache (Redis) for hot URLs, CDN for global speed." },
    { q: "Explain horizontal vs vertical scaling.", a: "Vertical: bigger machine (more CPU/RAM) — simpler but has limits. Horizontal: more machines — complex but unlimited scale. Modern systems prefer horizontal with load balancers." },
    { q: "What is a CDN and why use it?", a: "Content Delivery Network: distributed servers that cache static content close to users. Reduces latency, offloads origin server, improves availability. Essential for global apps." },
    { q: "What is the CAP theorem?", a: "A distributed system can guarantee only 2 of 3: Consistency (all reads see latest write), Availability (every request gets a response), Partition Tolerance (system works despite network splits). CP or AP in practice." },
    { q: "How does a load balancer work?", a: "Distributes incoming traffic across multiple servers. Algorithms: Round Robin, Least Connections, IP Hash. Can be L4 (transport) or L7 (HTTP). Improves availability and scalability." },
    { q: "What is eventual consistency?", a: "In distributed systems, updates propagate to all nodes eventually — but reads may return stale data temporarily. Trade-off for high availability. Used in DynamoDB, Cassandra, DNS." },
    { q: "What is a microservices architecture?", a: "Break an app into small, independent services each running its own process. Benefits: independent deployment, fault isolation, tech diversity. Challenges: distributed system complexity, network overhead." },
    { q: "How would you design a notification system?", a: "Components: Event producers → Message queue (Kafka/RabbitMQ) → Worker consumers → Notification providers (APNs, FCM, SMTP). Key: idempotency, retry logic, user preferences, rate limiting." },
    { q: "What is database sharding?", a: "Horizontal partitioning: split large DB into smaller shards each handling a subset of data (by user ID, region, etc.). Improves write scalability. Challenges: cross-shard queries, rebalancing." },
    { q: "What is caching and what strategies exist?", a: "Store frequently accessed data in fast storage (memory). Strategies: Cache-aside (app manages), Write-through (write to cache+DB), Write-back (async DB write). Eviction: LRU, LFU, TTL." }
  ],
  python: [
    { q: "What is a Python decorator?", a: "A function that wraps another function, adding behavior before/after it runs. Used with @syntax. Common uses: logging, authentication, timing. Implemented using closures." },
    { q: "Explain list comprehension.", a: "[expression for item in iterable if condition]. Creates a new list. More readable and faster than equivalent for-loops. Example: [x**2 for x in range(10) if x % 2 == 0]" },
    { q: "What is the GIL in Python?", a: "Global Interpreter Lock: allows only one thread to execute Python bytecode at a time. Prevents true multi-threading for CPU-bound tasks. Use multiprocessing or async I/O instead." },
    { q: "What is the difference between a list and a tuple?", a: "Lists are mutable (can be changed), tuples are immutable. Tuples are faster and can be used as dict keys or set elements (hashable). Use tuple for fixed data." },
    { q: "What are Python generators?", a: "Functions that yield values one at a time using `yield`. Memory efficient — don't store all values at once. Used for large data streams. `next()` advances the generator." },
    { q: "Explain `*args` and `**kwargs`.", a: "`*args` collects extra positional arguments as a tuple. `**kwargs` collects extra keyword arguments as a dict. Allow functions to accept variable numbers of arguments." },
    { q: "What is the difference between `deepcopy` and `copy`?", a: "`copy` creates a shallow copy — nested objects are still shared. `deepcopy` recursively copies all nested objects — fully independent. Use `copy` module for both." },
    { q: "What are Python's built-in data structures?", a: "list (ordered, mutable), tuple (ordered, immutable), dict (key-value, ordered since 3.7), set (unordered, unique), frozenset (immutable set). Each has specific use cases." },
    { q: "What is `__init__` vs `__new__` in Python?", a: "`__new__` creates the instance (called first, rarely overridden). `__init__` initializes it (commonly overridden for setup). Both are called during object creation." },
    { q: "What is duck typing?", a: "If it walks like a duck and quacks like a duck, it's a duck. Python checks behavior, not type. If an object has the required method/attribute, it works — no explicit interface needed." }
  ],
  hr: [
    { q: "What are your salary expectations?", a: "Research market rates first. Give a range based on your research, your experience, and the role requirements. 'Based on my research and experience, I'm looking for $X–$Y, but I'm open to discussion.'" },
    { q: "Are you interviewing at other companies?", a: "Be honest — say yes if you are. It shows you're in demand. Don't disclose specifics. 'Yes, I'm exploring a few opportunities, but this role is particularly exciting because…'" },
    { q: "When can you start?", a: "If employed: give your notice period (typically 2–4 weeks) plus a few days to transition. If available: 'I can start as soon as needed, though I'd appreciate a week to wrap up personal matters.'" },
    { q: "Do you prefer working alone or in a team?", a: "Show flexibility. 'I enjoy both — I'm effective independently and thrive collaborating in teams. I think the best results come from combining individual deep work with team collaboration.'" },
    { q: "What do you know about our company?", a: "Research before the interview: product, mission, recent news, culture. Show genuine interest. 'I know that [company] built [product] and recently [news]. What excites me is [specific aspect].'" }
  ]
};
