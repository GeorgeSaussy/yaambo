use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt::{Debug, Display, Error, Formatter};
use std::ops::Deref;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, RwLock};
use std::thread::yield_now;

#[derive(Debug, Clone)]
struct XorShift {
    state: u64,
}

impl XorShift {
    fn new() -> XorShift {
        XorShift { state: 1337 }
    }
    fn gen(&mut self) -> u64 {
        let mut x = self.state;
        if x == 0 {
            x = 1;
        }
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

#[derive(Debug)]
enum SuperKey<K> {
    Start,
    Real(K),
    End,
}

impl<K> SuperKey<K> {
    fn cast<T>(&self) -> SuperKey<&T>
    where
        T: Ord + ?Sized,
        K: Borrow<T> + Ord,
    {
        match self {
            SuperKey::Start => SuperKey::Start,
            SuperKey::Real(key) => {
                let key = key.borrow();
                SuperKey::Real(key)
            }
            SuperKey::End => SuperKey::End,
        }
    }
}

impl<K> Ord for SuperKey<K>
where
    K: Ord,
{
    fn cmp(&self, other: &SuperKey<K>) -> Ordering {
        match self {
            SuperKey::Start => match other {
                SuperKey::Start => Ordering::Equal,
                _ => Ordering::Less,
            },
            SuperKey::Real(key1) => match other {
                SuperKey::Start => Ordering::Greater,
                SuperKey::Real(key2) => key1.cmp(key2),
                SuperKey::End => Ordering::Less,
            },
            SuperKey::End => match other {
                SuperKey::End => Ordering::Equal,
                _ => Ordering::Greater,
            },
        }
    }
}

impl<K> PartialOrd for SuperKey<K>
where
    K: Ord,
{
    fn partial_cmp(&self, other: &SuperKey<K>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K> PartialEq for SuperKey<K>
where
    K: Ord,
{
    fn eq(&self, other: &SuperKey<K>) -> bool {
        match self {
            SuperKey::Start => matches!(other, SuperKey::Start),
            SuperKey::Real(key1) => match other {
                SuperKey::Real(key2) => key1.cmp(key2) == Ordering::Equal,
                _ => false,
            },
            SuperKey::End => matches!(other, SuperKey::End),
        }
    }
}

impl<K> Eq for SuperKey<K>
where
    K: Ord,
{
    // Nothing needs to be implemented here.
}

#[derive(Debug)]
struct Node<K, V: ?Sized> {
    key: SuperKey<K>,
    value: RwLock<Option<Arc<V>>>,
    forward: Vec<RwLock<NodeWrapper<K, V>>>,

    // TODO(gs): Consider other methods for locking.
    // Extra locking in needed because the RwLock guards will go out of scope
    // while the node is still needs to be exclusively modified by one thread.
    // There may be a way to handle this by keep an in-memory list of the
    // RwLock guards, but the benefits are not obvious.
    locks: Vec<AtomicBool>,
}

#[derive(Debug)]
struct NodeWrapper<K, V: ?Sized> {
    link: Arc<Node<K, V>>,
}

impl<K, V: ?Sized> Clone for NodeWrapper<K, V> {
    fn clone(&self) -> NodeWrapper<K, V> {
        NodeWrapper {
            link: self.link.clone(),
        }
    }
}

impl<K, V: ?Sized> NodeWrapper<K, V> {
    fn forward(&self, level: usize) -> NodeWrapper<K, V> {
        let link = &self.link;
        let node = link.as_ref();
        node.forward[level].read().unwrap().clone()
    }
    fn set_forward(&mut self, level: usize, wrapper: &NodeWrapper<K, V>) {
        let mut yield_yet = false;
        loop {
            if !yield_yet {
                yield_yet = true;
            } else {
                yield_now();
            }
            let node = self.link.as_ref();
            if node.locks.len() > level && node.locks[level].load(AtomicOrdering::SeqCst) {
                continue;
            }
            debug_assert_ne!(level, node.forward.len());
            *node.forward[level].write().unwrap() = wrapper.clone();
            break;
        }
    }
    fn locked_set_forward(&mut self, level: usize, wrapper: &NodeWrapper<K, V>) {
        let node = self.link.as_ref();
        debug_assert_ne!(level, node.forward.len());
        debug_assert!(node.locks[level].load(AtomicOrdering::SeqCst));
        *node.forward[level].write().unwrap() = wrapper.clone();
    }
    fn key(&self) -> &SuperKey<K> {
        let node = self.link.as_ref();
        &node.key
    }
    fn value(&self) -> Option<Arc<V>> {
        let node = self.link.as_ref();
        let value = node.value.read().unwrap();
        value.clone()
    }
    fn set_value<VAL>(&mut self, value: VAL)
    where
        V: From<VAL>,
    {
        let node = self.link.as_ref();
        *node.value.write().unwrap() = Some(Arc::new(value.into()));
    }
    fn lock(&mut self, level: usize) {
        let mut yield_yet = false;
        loop {
            if !yield_yet {
                yield_yet = true;
            } else {
                yield_now();
            }
            let node = self.link.as_ref();
            let result = node.locks[level].compare_exchange(
                false,
                true,
                AtomicOrdering::SeqCst,
                AtomicOrdering::SeqCst,
            );
            if result.is_ok() {
                break;
            }
        }
    }
    fn unlock(&mut self, level: usize) {
        let node = self.link.as_ref();
        let result = node.locks[level].compare_exchange(
            true,
            false,
            AtomicOrdering::SeqCst,
            AtomicOrdering::SeqCst,
        );
        debug_assert!(result.is_ok());
    }
    fn get_lock<KEY>(&self, search_key: &SuperKey<&KEY>, i: usize) -> NodeWrapper<K, V>
    where
        KEY: Ord + ?Sized,
        K: Borrow<KEY> + Ord,
    {
        debug_assert!(self.level() >= i);
        debug_assert!(&self.key().cast() < search_key);
        let mut x = self.clone();
        let mut y = self.forward(i);
        while &y.key().cast() < search_key {
            x = y;
            y = x.forward(i);
        }

        x.lock(i);
        y = x.forward(i);
        while &y.key().cast() < search_key {
            x.unlock(i);
            x = y;
            x.lock(i);
            y = x.forward(i);
        }
        x // still locked
    }
    fn level(&self) -> usize {
        let node = self.link.as_ref();
        node.forward.len() - 1
    }
}

/// `ConcurrentSkipList` is a implements the skip list data structure as
/// described in William Pugh's 1989, paper
/// ["Concurrent Maintenance of Skip Lists"].
///
/// ["Concurrent Maintenance of Skip Lists"]: https://15721.courses.cs.cmu.edu/spring2018/papers/08-oltpindexes1/pugh-skiplists-cacm1990.pdf
#[derive(Clone, Debug)]
pub struct ConcurrentSkipList<K, V: ?Sized> {
    head: NodeWrapper<K, V>,
    max_level: usize,
    rng: XorShift,

    // TODO(gs): Consider other methods for locking.
    // Perhaps this could be implemented without a performance loss
    // with an an RwLock (and maybe performance gains?), but this is
    // how the paper described the algorithm.
    // It is also worth revisiting the memory order used in the atomic
    // accesses.
    level_hint: Arc<AtomicUsize>,
    level_hint_lock: Arc<AtomicBool>,
}

impl<K, V: ?Sized> ConcurrentSkipList<K, V>
where
    K: Ord,
{
    /// Create a new ConcurrentSkipList instance.
    /// # Arguments
    /// - max_level: the maximum level for the skip list
    /// # Returns
    /// A new ConcurrentSkipList instance.
    pub fn new() -> ConcurrentSkipList<K, V> {
        const DEFAULT_SIZE: usize = 16;
        let nil_link = NodeWrapper {
            link: Arc::new(Node {
                key: SuperKey::End,
                value: RwLock::new(None),
                forward: Vec::new(),
                locks: Vec::new(),
            }),
        };
        let start_link = NodeWrapper {
            link: Arc::new(Node {
                key: SuperKey::Start,
                value: RwLock::new(None),
                forward: (0..DEFAULT_SIZE + 1)
                    .map(|_i| RwLock::new(nil_link.clone()))
                    .collect(),
                locks: (0..DEFAULT_SIZE + 1)
                    .map(|_i| AtomicBool::new(false))
                    .collect(),
            }),
        };
        ConcurrentSkipList {
            head: start_link,
            level_hint: Arc::new(AtomicUsize::new(0)),
            level_hint_lock: Arc::new(AtomicBool::new(false)),
            max_level: DEFAULT_SIZE,
            rng: XorShift::new(),
        }
    }

    /// Get the value stored for a given key.
    /// This is based on the "Search" method in the Pugh paper.
    ///
    /// # Arguments
    ///
    /// - key: the given key
    ///
    /// Note that the `Ord` implementation of `KEY` must match that of the
    /// concurrent skip list's key type `K`.
    ///
    /// # Returns
    ///
    /// A shared pointer to the value if present.
    pub fn get<KEY>(&self, key: &KEY) -> Option<Arc<V>>
    where
        KEY: Ord + ?Sized,
        K: Borrow<KEY>,
    {
        let search_key = SuperKey::Real(key);
        let mut x = self.head.clone();
        let mut y = x.clone();
        debug_assert!(x.key().cast() < search_key);
        for i in (0..self.get_level_hint() + 1).rev() {
            y = x.forward(i);
            while y.key().cast() < search_key {
                x = y;
                y = x.forward(i);
            }
        }
        debug_assert!(x.key().cast() < search_key);
        if y.key().cast() == search_key {
            return y.value();
        }
        None
    }
    /// Set the value for a given key.
    /// This method will not fail and will insert a key if it is not already present.
    ///
    /// # Arguments
    ///
    /// - key: the given key
    /// - value: the given value
    pub fn set<KEY: Into<K>, VAL>(&mut self, key: KEY, value: VAL)
    where
        V: From<VAL>,
    {
        let search_key = SuperKey::Real(key.into());
        let mut updates = Vec::new();
        let mut x = self.head.clone();
        let big_l = self.get_level_hint();
        for i in (0..big_l + 1).rev() {
            let mut y = x.forward(i);
            while y.key() < &search_key {
                x = y;
                y = x.forward(i);
            }
            updates.push(x.clone());
        }
        let mut updates: Vec<NodeWrapper<K, V>> = updates.into_iter().rev().collect();
        let mut x = x.get_lock(&search_key.cast(), 0);
        if x.forward(0).key() == &search_key {
            x.forward(0).set_value(value);
            x.unlock(0);
            return;
        }

        let new_level = self.random_level();
        let forward_0 = x.forward(0);
        let mut y = NodeWrapper {
            link: Arc::new(Node {
                key: search_key,
                value: RwLock::new(Some(Arc::new(value.into()))),
                forward: (0..new_level + 1)
                    .map(|_i| RwLock::new(forward_0.clone()))
                    .collect(),
                locks: (0..new_level + 1)
                    .map(|_i| AtomicBool::new(false))
                    .collect(),
            }),
        };
        for _ in big_l..new_level {
            updates.push(self.head.clone());
        }
        for (i, update) in updates.iter().take(new_level + 1).enumerate() {
            if i != 0 {
                x = update.get_lock(&y.key().cast(), i);
            }
            y.set_forward(i, &x.forward(i));
            x.locked_set_forward(i, &y);
            x.unlock(i);
        }

        let big_l = self.get_level_hint();
        if big_l < self.max_level
            && self.head.forward(big_l + 1).key() != &SuperKey::End
            && !self.is_level_hint_locked()
        {
            self.lock_level_hint();
            let mut level_hint = self.get_level_hint();
            while level_hint < self.max_level
                && self.head.forward(level_hint + 1).key() != &SuperKey::End
            {
                level_hint += 1;
            }
            self.set_level_hint(level_hint);
            self.unlock_level_hint();
        }
    }
    /// Remove a given key.
    ///
    /// # Arguments
    ///
    /// - key: the given key
    ///
    /// Note that the `Ord` implementation of `KEY` must match that of the
    /// concurrent skip list's key type `K`.
    ///
    /// # Returns
    ///
    /// A pointer to the value that was removed if present.
    pub fn remove<KEY>(&mut self, key: &KEY) -> Option<Arc<V>>
    where
        KEY: Debug + Ord + ?Sized,
        K: Borrow<KEY> + Debug,
    {
        let search_key = SuperKey::Real(key);
        let mut updates = Vec::new();
        let mut x = self.head.clone();
        let big_l = self.get_level_hint();
        for i in (0..big_l + 1).rev() {
            let mut y = x.forward(i);
            while y.key().cast() < search_key {
                x = y;
                y = x.forward(i);
            }
            updates.push(x.clone());
        }
        let mut updates: Vec<NodeWrapper<K, V>> = updates.into_iter().rev().collect();
        let mut y = x;
        loop {
            let i = 0;
            y = y.forward(i);
            if y.key().cast() > search_key {
                return None;
            }
            // lock(y, level)
            let is_garbage = y.key() > y.forward(i).key();
            // if is_garbage then unlock(y, level)
            if y.key().cast() == search_key && !is_garbage {
                break;
            }
        }
        for _ in big_l + 1..y.level() {
            updates.push(self.head.clone());
        }
        for i in (0..y.level() + 1).rev() {
            x = updates[i].get_lock(&search_key, i);
            debug_assert_eq!(x.forward(i).key(), y.key());
            debug_assert!(Arc::ptr_eq(&x.forward(i).link, &y.link));
            y.lock(i);
            x.locked_set_forward(i, &y.forward(i));
            y.locked_set_forward(i, &x);
            x.unlock(i);
            y.unlock(i);
        }

        let big_l = self.get_level_hint();
        if big_l > 0
            && self.head.forward(big_l).key() == &SuperKey::End
            && !self.is_level_hint_locked()
        {
            self.lock_level_hint();
            let mut new_level_hint = self.get_level_hint();
            while new_level_hint > 0 && self.head.forward(new_level_hint).key() == &SuperKey::End {
                new_level_hint -= 1;
            }
            self.set_level_hint(new_level_hint);
            self.unlock_level_hint();
        }
        y.value()
    }
    /// Check if the skip list contains an instance of the key.
    ///
    /// # Arguments
    ///
    /// - key: the key to search for
    ///
    /// Note that the `Ord` implementation of `KEY` must match that of the
    /// concurrent skip list's key type `K`.
    ///
    /// # Returns
    ///
    /// True if the key is present, false otherwise.
    pub fn contains<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.get(key).is_some()
    }
    /// Get an iterator over the the key-value pairs in the skip list.
    ///
    /// # Returns
    ///
    /// An outright owned iterator. This works because the skip list is
    /// designed so that an iterator that is accessing a removed node will
    /// be 'guided' back to the main list. This means that an iterator
    /// WILL NOT have a strongly consistent view of the skip list,
    /// but this allows for safe multi-threading.
    pub fn iter(&self) -> ConcurrentSkipListScanner<K, V> {
        ConcurrentSkipListScanner {
            node_wrapper: self.head.clone(),
            end: None,
            done: false,
            started: false,
        }
    }
    fn get_level_hint(&self) -> usize {
        self.level_hint.load(AtomicOrdering::SeqCst)
    }
    fn set_level_hint(&mut self, level_hint: usize) {
        debug_assert!(self.level_hint_lock.load(AtomicOrdering::SeqCst));
        self.level_hint.swap(level_hint, AtomicOrdering::SeqCst);
    }
    fn is_level_hint_locked(&self) -> bool {
        self.level_hint_lock.load(AtomicOrdering::SeqCst)
    }
    fn lock_level_hint(&mut self) {
        let mut yield_yet = false;
        loop {
            if !yield_yet {
                yield_yet = true;
            } else {
                yield_now();
            }
            let result = self.level_hint_lock.compare_exchange(
                false,
                true,
                AtomicOrdering::SeqCst,
                AtomicOrdering::SeqCst,
            );
            if result.is_ok() {
                break;
            }
        }
    }
    fn unlock_level_hint(&mut self) {
        let result = self.level_hint_lock.compare_exchange(
            true,
            false,
            AtomicOrdering::SeqCst,
            AtomicOrdering::SeqCst,
        );
        debug_assert!(result.is_ok());
    }
    fn random_level(&mut self) -> usize {
        let p = 0.5; // TODO(OMEGA-937): This should be tunable.
        let mut lvl = 0;
        // TODO(gs): Write a better random library.
        while ((self.rng.gen() % 10_000) as f64) < p * 10_000.0 && lvl < self.max_level - 1 {
            lvl += 1;
        }
        lvl
    }
}

impl<K: Ord, V> Default for ConcurrentSkipList<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Ord, V> IntoIterator for ConcurrentSkipList<K, V> {
    type Item = (ConcurrentSkipListKeyView<K, V>, Arc<V>);
    type IntoIter = ConcurrentSkipListScanner<K, V>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K: Ord, V> IntoIterator for &'a ConcurrentSkipList<K, V> {
    type Item = (ConcurrentSkipListKeyView<K, V>, Arc<V>);
    type IntoIter = ConcurrentSkipListScanner<K, V>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<KEY, K, VAL, V: ?Sized> FromIterator<(KEY, VAL)> for ConcurrentSkipList<K, V>
where
    KEY: Into<K>,
    K: Ord,
    V: From<VAL>,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = (KEY, VAL)>,
    {
        let mut skip_list = Self::new();
        for (key, value) in iter {
            skip_list.set(key, value);
        }
        skip_list
    }
}

impl<KEY, K, VAL, V, const N: usize> From<[(KEY, VAL); N]> for ConcurrentSkipList<K, V>
where
    KEY: Into<K>,
    K: Ord,
    V: From<VAL>,
{
    fn from(pairs: [(KEY, VAL); N]) -> Self {
        let mut skip_list = Self::new();
        for (key, value) in pairs {
            skip_list.set(key, value);
        }
        skip_list
    }
}

pub struct ConcurrentSkipListKeyView<K, V> {
    node_wrapper: NodeWrapper<K, V>,
}

impl<K, V> ConcurrentSkipListKeyView<K, V> {
    fn new(node_wrapper: NodeWrapper<K, V>) -> Self {
        Self { node_wrapper }
    }
}

impl<K, V> AsRef<K> for ConcurrentSkipListKeyView<K, V> {
    fn as_ref(&self) -> &K {
        match self.node_wrapper.link.key {
            SuperKey::Real(ref key) => key,
            SuperKey::End => panic!("this should never happen"),
            SuperKey::Start => panic!("this should never happen"),
        }
    }
}

impl<K, V> Borrow<K> for ConcurrentSkipListKeyView<K, V> {
    fn borrow(&self) -> &K {
        self.as_ref()
    }
}

impl<K, V> Deref for ConcurrentSkipListKeyView<K, V> {
    type Target = K;

    fn deref(&self) -> &K {
        self.as_ref()
    }
}

impl<K: Display, V> Display for ConcurrentSkipListKeyView<K, V> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        write!(f, "{}", self.as_ref())
    }
}

impl<K: Debug, V> Debug for ConcurrentSkipListKeyView<K, V> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        write!(f, "{:?}", self.as_ref())
    }
}

pub struct ConcurrentSkipListScanner<K, V: ?Sized> {
    node_wrapper: NodeWrapper<K, V>,
    end: Option<K>,
    done: bool,
    started: bool,
}

impl<K, V> ConcurrentSkipListScanner<K, V> {
    fn new(node_wrapper: NodeWrapper<K, V>, end: Option<K>) -> Self {
        Self {
            node_wrapper,
            end,
            done: false,
            started: false,
        }
    }
    pub fn from<Q>(self, key: &Q) -> Self
    where
        K: Borrow<Q> + Clone,
        V: Clone,
        Q: Ord + ?Sized,
    {
        if self.started {
            panic!("from cannot be called on concurrent skip list scanner once it has started");
        }
        let mut node_wrapper = self.node_wrapper;
        loop {
            let node_wrapper_key = node_wrapper.key();
            if let SuperKey::Real(base_key) = node_wrapper_key {
                if base_key.borrow() >= key {
                    break;
                }
            }
            node_wrapper = node_wrapper.forward(0);
        }
        ConcurrentSkipListScanner::new(node_wrapper, self.end)
    }
    pub fn to(self, key: K) -> Self {
        if self.started {
            panic!("to cannot be called on concurrent skip list scanner once it has started");
        }
        ConcurrentSkipListScanner::new(self.node_wrapper, Some(key))
    }
}

impl<K: Ord, V> Iterator for ConcurrentSkipListScanner<K, V> {
    type Item = (ConcurrentSkipListKeyView<K, V>, Arc<V>);
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        match self.node_wrapper.key() {
            SuperKey::Start => {
                self.node_wrapper = self.node_wrapper.forward(0);
                self.next()
            }
            SuperKey::Real(key) => {
                if let Some(end) = &self.end {
                    if key >= end {
                        self.done = true;
                        return None;
                    }
                }
                let value = self.node_wrapper.value().unwrap();
                let key_view = ConcurrentSkipListKeyView::new(self.node_wrapper.clone());
                self.node_wrapper = self.node_wrapper.forward(0);
                Some((key_view, value))
            }
            SuperKey::End => None,
        }
    }
}

#[cfg(test)]
mod test {
    use std::thread::spawn;
    use std::time::Instant;

    use super::*;

    #[test]
    fn test_xor_shift_randomness() {
        // The XorShift RNG is not cryptographically secure, but it should be
        // good enough for our purposes.

        let seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        for seed in seeds {
            let mut rng = XorShift::new();
            rng.state = seed;
            let mut even_count = 0;
            let mut odd_count = 0;
            for _ in 0..1000 {
                let num = rng.gen();
                if num % 2 == 0 {
                    even_count += 1;
                } else {
                    odd_count += 1;
                }
            }
            let diff = i32::abs(even_count - odd_count);
            assert!(
                diff < 100,
                "Seed: {}, even: {}, odd: {}",
                seed,
                even_count,
                odd_count
            );
        }
    }

    #[test]
    pub fn test_skip_list_end_to_end() {
        let mut skip_list: ConcurrentSkipList<i32, i32> = ConcurrentSkipList::new();
        let top = 1_000;
        for k in 0..top {
            if k % 2 == 0 {
                assert!(skip_list.get(&k).is_none());
                skip_list.set(k, k * k);
                assert!(skip_list.contains(&k));
                assert_eq!(*skip_list.get(&k).unwrap(), k * k);
            }
        }
        for k in 1..top {
            if k % 2 == 1 {
                assert!(skip_list.get(&k).is_none());
                skip_list.set(k, k + 1);
                let option = skip_list.get(&k);
                assert!(option.is_some());
                if let Some(unwrap) = option {
                    assert_eq!(*unwrap, k + 1);
                }
            }
        }

        for k in top / 2..top {
            let mut expect = k * k;
            if k % 2 == 1 {
                expect = k + 1;
            }
            let val = skip_list.get(&k);
            let is_some = val.is_some();
            assert!(is_some);
            if is_some {
                assert_eq!(*val.unwrap(), expect);
            }
            expect *= 2;
            skip_list.set(k, expect);
            let val = skip_list.get(&k);
            let is_some = val.is_some();
            assert!(is_some);
            if is_some {
                assert_eq!(*val.unwrap(), expect);
            }
        }
        for k in 0..top {
            let is_some = skip_list.get(&k).is_some();
            assert!(is_some);
            let option = skip_list.remove(&k);
            assert!(option.is_some());
            let is_none = skip_list.get(&k).is_none();
            assert!(is_none);
        }
    }

    fn run_pre_planned_benchmark(num_workers: usize, num_entries_per_worker: usize) -> usize {
        let skip_list: ConcurrentSkipList<usize, usize> = ConcurrentSkipList::new();
        let mut handles = Vec::new();
        let start = Instant::now();
        for worker_idx in 0..num_workers {
            let mut skip_list_copy = skip_list.clone();
            handles.push(spawn(move || {
                let start_idx = worker_idx * num_entries_per_worker;
                let mid_idx = start_idx + num_entries_per_worker / 2;
                let end_idx = start_idx + num_entries_per_worker;
                for i in mid_idx..end_idx {
                    skip_list_copy.set(i, i * i);
                }
                for i in start_idx..mid_idx {
                    skip_list_copy.set(i, i * i);
                }
                for i in start_idx..end_idx {
                    assert!(skip_list_copy.get(&i).is_some());
                    assert!(skip_list_copy.remove(&i).is_some());
                }
            }));
        }
        for handle in handles {
            assert!(handle.join().is_ok());
        }
        let duration = start.elapsed();
        duration.as_millis() as usize
    }

    #[test]
    fn test_skip_list_benchmark() {
        let max_num_threads = 16;
        let mut num_threads = 1;
        let total_entries = 1_000;
        while num_threads <= max_num_threads {
            let duration = run_pre_planned_benchmark(num_threads, total_entries / num_threads);
            println!("{} threads took {} ms", num_threads, duration);
            num_threads *= 2;
        }
    }

    #[test]
    fn test_skip_list_scanner() {
        let top = 100;
        let mut keys: Vec<String> = (0..top).map(|i| i.to_string()).collect();
        keys.sort();
        let sorted_keys = keys;

        let mut skip_list: ConcurrentSkipList<String, String> = ConcurrentSkipList::new();
        for i in 0..top {
            let s = i.to_string();
            skip_list.set(s.clone(), s.clone());
            assert_eq!(*skip_list.get(&s).unwrap(), s);
        }

        for (i, (key, value)) in skip_list.iter().enumerate() {
            let s = sorted_keys[i].clone();
            assert_eq!(s, *key);
            assert_eq!(s, *value);
        }

        let mut count = 0;
        for (key, value) in skip_list.iter().from("25").to(String::from("75")) {
            count += 1;
            assert!("25" <= &key);
            assert!("75" > &value);
        }
        let index_25 = sorted_keys.iter().position(|r| r == "25").unwrap();
        let index_75 = sorted_keys.iter().position(|r| r == "75").unwrap();
        assert_eq!(count, index_75 - index_25);
    }
}
