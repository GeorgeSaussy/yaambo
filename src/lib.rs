#![deny(missing_docs)]
//! A concurrent skip list is a multithreaded implementation of the skip list data structure.
//! It is a probabilistic data structure that allows for fast insertion, deletion, and lookup of
//! elements. It is also efficient in terms of space usage.
//!
//! A skip list is a linked list with a number of levels. Each level is a linked list of nodes,
//! where each node contains a key and a value. The keys are sorted in ascending order. The higher
//! levels of the skip list skip over a large number of nodes, which makes lookup operations very
//! efficient.
//!
//! A concurrent skip list uses hand-over-hand locking to protect the nodes in the skip list.
//! This means that only one thread can be modifying a node at a time. However, multiple threads
//! can be reading the nodes at the same time.
//!
//! The concurrent skip list is a good choice for applications that need to be able to handle a
//! large number of concurrent operations. It is also a good choice for applications that need to
//! be able to perform fast lookup operations.
//!
//! Here are some of the benefits of using a concurrent skip list:
//!
//! - Fast insertion, deletion, and lookup of elements
//! - Efficient in terms of space usage
//! - Scalable to handle a large number of concurrent operations
//! - Good for applications that need to perform fast lookup operations
//!
//! Here are some of the limitations of using a concurrent skip list:
//!
//! Can be more complex to implement than other data structures
//!
//! - May not be as efficient as other data structures for some operations, such as range queries
//! - May not be as scalable as other data structures for some applications
//!
//! This implementation is based on the on documented in
//! William Pugh's 1989, paper ["Concurrent Maintenance of Skip Lists"].
//!
//! ["Concurrent Maintenance of Skip Lists"]: https://15721.courses.cs.cmu.edu/spring2018/papers/08-oltpindexes1/pugh-skiplists-cacm1990.pdf
//!
//! # Examples
//!
//! The following examples are adapted from the examples in the Rust standard library
//! documentation for [`HashMap`].
//!
//! ```
//! use yaambo::ConcurrentSkipList;
//!
//! let mut movie_reviews: ConcurrentSkipList<String, String> = ConcurrentSkipList::new();
//!
//! // Review some movies.
//! movie_reviews.set(
//!     "The Matrix".to_string(),
//!     "My favorite movie.".to_string(),
//! );
//! movie_reviews.set(
//!     "Liquorice Pizza".to_string(),
//!     "Not for me.".to_string(),
//! );
//! movie_reviews.set(
//!     "Moulin Rouge".to_string(),
//!     "A guilty pleasure.".to_string(),
//! );
//! movie_reviews.set(
//!     "The Power of the Dog".to_string(),
//!     "Underroted.".to_string(),
//! );
//!
//! // Check for a specific one.
//! // When collections store owned values (String), they can still be
//! // queried using references (&str).
//! if !movie_reviews.contains("Les Misérables") {
//!     println!("We've got {} reviews, but Les Misérables ain't one.",
//!              movie_reviews.iter().count());
//! }
//!
//! // oops, this review has a lot of spelling mistakes, let's delete it.
//! movie_reviews.remove("The Power of the Dog");
//!
//! // Look up the values associated with some keys.
//! let to_find = ["Moulin Rouge", "Alice's Adventure in Wonderland"];
//! for &movie in &to_find {
//!     match movie_reviews.get(movie) {
//!         Some(review) => println!("{movie}: {review}"),
//!         None => println!("{movie} is un-reviewed.")
//!     }
//! }
//!
//! // Iterate over everything.
//! for (movie, review) in &movie_reviews {
//!     println!("{movie}: \"{review}\"");
//! }
//! ```
//!
//! A `ConcurrentSkipList` with a known list of items can be initialized
//! from an array:
//!
//! ```
//! use yaambo::ConcurrentSkipList;
//!
//! let solar_distance: ConcurrentSkipList<String, f32> = ConcurrentSkipList::from([
//!     ("Mercury", 0.4),
//!     ("Venus", 0.7),
//!     ("Earth", 1.0),
//!     ("Mars", 1.5),
//! ]);
//! ```
//!
//! The easiest way to use `ConcurrentSkipList` with a custom key type is
//! to derive [`Ord`]. This requires that the key derive [`PartialEq`],
//! [`Eq`], and [`PartialOrd`] as well.
//!
//! ```
//! use yaambo::ConcurrentSkipList;
//!
//! #[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
//! struct StreetFighter {
//!     name: String,
//!     country: String,
//! }
//!
//! impl StreetFighter {
//!     //! Creates a new Street Fighter.
//!     fn new(name: &str, country: &str) -> StreetFighter {
//!         StreetFighter { name: name.to_string(), country: country.to_string() }
//!     }
//! }
//!
//! // Use a ConcurrentSkipList to store the fighters' health points.
//! let fighters: ConcurrentSkipList<StreetFighter, usize> = ConcurrentSkipList::from([
//!     (StreetFighter::new("Akuma", "Japan"), 900_usize),
//!     (StreetFighter::new("Zangief", "Russia"), 1075_usize),
//!     (StreetFighter::new("Chun Li", "China"), 975_usize),
//! ]);
//!
//! // Use derived implementation to print the status of the fighters.
//! for (fighter, health) in &fighters {
//!     println!("{fighter:?} has {health} hp");
//! }
//! ```
//!
//! [`HashMap`]: https://doc.rust-lang.org/std/collections/struct.HashMap.html
mod concurrent_skip_list;
pub use concurrent_skip_list::ConcurrentSkipList;
