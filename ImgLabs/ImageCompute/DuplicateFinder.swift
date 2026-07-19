//
//  DuplicateFinder.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-05.
//  Turns a ZNCC similarity matrix into clusters of duplicate images, each with one image chosen to keep

import Foundation

// MARK: - Union-Find

/// A disjoint set data structure. Helps with given a bunch of items
/// and statements like "item A is connected to item B", which items end up in the same connected group?
///
/// Each item points to a "parent" item. Follow the parent links upward and you eventually reach an item
/// that points to itself -- that item is the group's "leader" (root). Two items are in the same group if,
/// and only if, they have the same leader. The leader is just an internal label, it means nothing about
/// the items themselves
struct UnionFind {
    // parent[i] is the item that i points to. When parent[i] == i, item i is a leader.
    private var parent : [Int];
    // size[root] is how many items are in that root's group. Only meaningful for items that are leaders
    // Used to keep the trees shallow (see union), which keeps lookups fast
    private var size : [Int];

    init(count: Int) {
        // Array(0..<count) turns the range 0,1,2,...,count-1 into the array [0, 1, 2, ..., count-1].
        // So initially parent[i] == i for every i: every item is its own leader (every item is its own group).
        self.parent = Array(0..<count);
        // Every group starts with exactly one item, so every size is 1.
        self.size = Array(repeating: 1, count: count);
    }

    /// Returns the leader (root) of the group containing x. i.e., for x, find it's leader
    mutating func find(_ x: Int) -> Int {
        // Step 1: walk up the parent chain until reaching a self-pointing item -- that's the leader.
        var root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        // Step 2 (path compression): walk the chain again and point every item we passed directly at the
        // root. Next time we call find on any of them, it's a single hop instead of a long walk.
        var current = x;
        while self.parent[current] != current {
            let next = self.parent[current]; // remember where the next node is...
            self.parent[current] = root;     // ...then re-point this item straight at the root
            current = next;
        }
        return root;
    }

    /// Merges the group containing a with the group containing b
    mutating func union(_ a: Int, _ b: Int) {
        let rootA = self.find(a);
        let rootB = self.find(b);
        // `guard` runs its else block and exits early if the condition is false. Here: if a and b already
        // share a leader they're already in the same group, so there's nothing to merge -- just return
        guard rootA != rootB else { return; }

        // "Union by size": attach the smaller group beneath the larger group's leader. Keeping the bigger
        // tree on top stops the parent chains from growing long, which keeps find fast
        if self.size[rootA] < self.size[rootB] {
            self.parent[rootA] = rootB;      // rootA is no longer a leader; it now points at rootB
            self.size[rootB] += self.size[rootA];
        } else {
            self.parent[rootB] = rootA;
            self.size[rootA] += self.size[rootB];
        }
    }
}

// MARK: - Duplicate grouping

/// A cluster of duplicate images: one to keep, the rest suggested for removal
/// `Identifiable` (a UUID `id`) lets SwiftUI tell groups apart in a List/ForEach without extra work
struct DuplicateGroup : Identifiable {
    let id = UUID();
    let keep : Int;             // Index (into the original images array) of the image to keep
    let duplicates : [Int];     // Indices of the other images in the cluster, suggested for removal
    var all : [Int] { [self.keep] + self.duplicates; }  // A computed property (recalculated each time it's read)
}

/// Clusters images from a symmetric ZNCC similarity matrix into duplicate groups
/// - Parameters:
///     - matrix: an NxN matrix where matrix[i][j] is the similarity of image i and image j, in [-1, 1]
///               It is symmetric (matrix[i][j] == matrix[j][i]) with 1.0 on the diagonal
///     - threshold: pairs whose similarity is >= this value are treated as duplicates of each other
///     - quality: per-image quality signals, index-aligned with the matrix. Empty when unavailable, in
///                which case a quality-aware strategy falls back to the medoid
///     - strategy: how to choose each cluster's keeper. Defaults to the medoid (matrix-only) behavior
/// - Returns: one DuplicateGroup per cluster containing 2+ images. Images with no duplicates are omitted
func duplicateGroups(matrix: [[Float]], threshold: Float, quality: [ImageQuality] = [], strategy: KeeperStrategy = MedoidStrategy()) -> [DuplicateGroup] {
    let n = matrix.count;
    guard n > 1 else { return []; } // Need at least two images for a duplicate to exist

    // Connect every pair whose similarity clears the threshold
    // Only scan the upper triangle (j starts at i+1) because the matrix is symmetric -- matrix[i][j]
    // and matrix[j][i] are equal, so checking one side is enough and we skip the i == i diagonal
    // The `where` clause is a filter on the loop: the body only runs for j values that satisfy it
    var unionFind = UnionFind(count: n);
    for i in 0..<n {
        for j in (i + 1)..<n where matrix[i][j] >= threshold {
            unionFind.union(i, j);
        }
    }

    // Bucket every image under its leader, images sharing a leader collect into the same array
    var buckets : [Int: [Int]] = [:];
    for i in 0..<n {
        buckets[unionFind.find(i), default: []].append(i);
    }

    // Convert each multi-image bucket into a DuplicateGroup
    return buckets.values
        // .filter keeps only the elements for which the closure returns true. Inside a trailing closure,
        // `$0` is the shorthand name for the current element -- here, one bucket (an [Int]). We keep only
        // buckets with more than one image, since a lone image isn't a duplicate of anything
        .filter { $0.count > 1 }
        // .map transforms each element into something new, producing a new array. Here each bucket
        // (named `members` for clarity instead of using $0) becomes a DuplicateGroup
        .map { members in
            let keeper = strategy.keeper(from: members, matrix: matrix, quality: quality);
            // members.filter { $0 != keeper } is every index in the cluster except the one we keep
            return DuplicateGroup(keep: keeper, duplicates: members.filter { $0 != keeper });
        }
        // Dictionaries have no guaranteed order, so sort the groups for a stable, predictable UI
        // { $0.all.min()! < $1.all.min()! } orders groups by their smallest image index; here $0 and $1
        // are the two groups being compared, and the closure returns true if $0 should come first
        .sorted { $0.all.min()! < $1.all.min()!; };
}

// MARK: - Keeper selection helpers

/// Average similarity of one image to the other members of its cluster.
func averageSimilarity(of index: Int, within members: [Int], matrix: [[Float]]) -> Float {
    let others = members.filter { $0 != index }; // everyone in the cluster except `index` itself
    guard !others.isEmpty else { return 0; }
    // .reduce combines a collection into a single value. The first argument (Float(0)) is the starting
    // total; the closure is called once per element with ($0 = running total so far, $1 = current element),
    // and whatever it returns becomes the new running total. So this sums matrix[index][other] over others
    let total = others.reduce(Float(0)) { $0 + matrix[index][$1]; };
    return total / Float(others.count); // divide the sum by the count to get the mean
}
