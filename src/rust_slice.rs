// From https://raw.githubusercontent.com/rust-lang/rust/1.56.1/library/core/tests/slice.rs
// (which has not changed from 1.56.1 to 1.67.1).

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn sort_unstable() {
    use core::cmp::Ordering::{Equal, Greater, Less};
    // MODIFIED: test heapsort from this module, not the stdlib one (which is internal anyway).
    use crate::sort::heapsort;
    use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};

    // Miri is too slow (but still need to `chain` to make the types match)
    let lens = if cfg!(miri) { (2..20).chain(0..0) } else { (2..25).chain(500..510) };
    let rounds = if cfg!(miri) { 1 } else { 100 };

    let mut v = [0; 600];
    let mut tmp = [0; 600];
    let mut rng = StdRng::from_entropy();

    for len in lens {
        let v = &mut v[0..len];
        let tmp = &mut tmp[0..len];

        for &modulus in &[5, 10, 100, 1000] {
            for _ in 0..rounds {
                for i in 0..len {
                    v[i] = rng.gen::<i32>() % modulus;
                }

                // Sort in default order.
                tmp.copy_from_slice(v);
                tmp.sort_unstable();
                assert!(tmp.windows(2).all(|w| w[0] <= w[1]));

                // Sort in ascending order.
                tmp.copy_from_slice(v);
                tmp.sort_unstable_by(|a, b| a.cmp(b));
                assert!(tmp.windows(2).all(|w| w[0] <= w[1]));

                // Sort in descending order.
                tmp.copy_from_slice(v);
                tmp.sort_unstable_by(|a, b| b.cmp(a));
                assert!(tmp.windows(2).all(|w| w[0] >= w[1]));

                // Test heapsort using `<` operator.
                tmp.copy_from_slice(v);
                heapsort(tmp, |a, b| a < b);
                assert!(tmp.windows(2).all(|w| w[0] <= w[1]));

                // Test heapsort using `>` operator.
                tmp.copy_from_slice(v);
                heapsort(tmp, |a, b| a > b);
                assert!(tmp.windows(2).all(|w| w[0] >= w[1]));
            }
        }
    }

    // Sort using a completely random comparison function.
    // This will reorder the elements *somehow*, but won't panic.
    for i in 0..v.len() {
        v[i] = i as i32;
    }
    v.sort_unstable_by(|_, _| *[Less, Equal, Greater].choose(&mut rng).unwrap());
    v.sort_unstable();
    for i in 0..v.len() {
        assert_eq!(v[i], i as i32);
    }

    // Should not panic.
    [0i32; 0].sort_unstable();
    [(); 10].sort_unstable();
    [(); 100].sort_unstable();

    let mut v = [0xDEADBEEFu64];
    v.sort_unstable();
    assert!(v == [0xDEADBEEF]);
}
