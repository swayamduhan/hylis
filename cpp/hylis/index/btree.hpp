// index/btree.hpp
//
// In-memory B+ tree.
//
// Structure
// ---------
// Internal nodes hold only separator keys and child pointers; *all* values
// live in the leaves. Leaves are chained left-to-right, so a range scan
// descends once to find the start and then walks the chain sideways instead
// of re-descending per key. That leaf chain is the main reason a B+ tree is
// preferred over a plain B-tree for range queries, which is exactly the
// workload the query planner cares about.
//
// Node representation
// -------------------
// LeafNode and InternalNode are distinct classes over a small polymorphic
// Node base, rather than one node type with a tagged union of payloads. That
// costs a vtable pointer per node, but each class then holds only the fields
// it actually uses and the split/merge code reads as two clearly separate
// cases. For a project whose point is to *explain* the algorithm, that
// clarity is worth more than the pointer.
//
// Navigation
// ----------
// Nodes carry no parent pointer. Instead every mutating operation records the
// (node, child-index) path it took down from the root and walks that path
// back up to rebalance. This keeps splits and merges from having to fix up
// parent links everywhere, and — unlike re-deriving the parent by searching
// for a node's own key — it still works when a node has been emptied.
//
// Separators
// ----------
// A separator key is a routing *boundary*, not a copy that must stay in sync
// with the data: keys[i] guarantees children[i] < keys[i] <= children[i+1],
// and nothing more. See the note on InternalNode for why deletes are allowed
// to leave a separator pointing at a key that no longer exists.
//
// Order
// -----
// `order` (m) is the maximum number of children an internal node may have.
// An internal node holds at most m-1 keys; a leaf holds at most m-1 entries.
// A non-root node must keep at least ceil(m/2)-1 keys. The root is exempt:
// once it drops below that the tree simply collapses a level.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// CompareOp is shared with the learned index — see the header for why the
// two structures must present the same predicate interface.
#include "index/compare_op.hpp"

namespace hylis::index {

template <typename Key = std::int64_t, typename Value = std::int64_t>
class BPlusTree {
public:
    explicit BPlusTree(std::size_t order = 32) : order_(order) {
        if (order_ < 3) throw std::invalid_argument("BPlusTree: order must be >= 3");
        root_ = std::make_unique<LeafNode>();
    }

    BPlusTree(const BPlusTree&) = delete;
    BPlusTree& operator=(const BPlusTree&) = delete;
    BPlusTree(BPlusTree&&) noexcept = default;
    BPlusTree& operator=(BPlusTree&&) noexcept = default;

    std::size_t size() const { return count_; }
    bool empty() const { return count_ == 0; }
    std::size_t order() const { return order_; }

    // Height in nodes: a tree whose root is a leaf has height 1.
    std::size_t height() const {
        std::size_t h = 1;
        const Node* n = root_.get();
        while (!n->is_leaf) {
            n = static_cast<const InternalNode*>(n)->children[0].get();
            ++h;
        }
        return h;
    }

    // Insert or overwrite. Returns true if a new key was added, false if an
    // existing key's value was replaced.
    bool insert(const Key& key, const Value& value) {
        Path path;
        LeafNode* leaf = descend(key, &path);
        const std::size_t pos = lower_bound_idx(leaf->keys, key);

        if (pos < leaf->keys.size() && leaf->keys[pos] == key) {
            leaf->values[pos] = value;
            return false;
        }

        leaf->keys.insert(leaf->keys.begin() + static_cast<std::ptrdiff_t>(pos), key);
        leaf->values.insert(leaf->values.begin() + static_cast<std::ptrdiff_t>(pos), value);
        ++count_;

        if (leaf->keys.size() > max_keys()) {
            auto [separator, right] = split_leaf(leaf);
            insert_into_parent(path, separator, std::move(right));
        }
        return true;
    }

    const Value* find(const Key& key) const {
        const LeafNode* leaf = descend(key);
        const std::size_t pos = lower_bound_idx(leaf->keys, key);
        if (pos < leaf->keys.size() && leaf->keys[pos] == key) return &leaf->values[pos];
        return nullptr;
    }

    bool contains(const Key& key) const { return find(key) != nullptr; }

    // Remove a key. Returns true if it was present.
    bool erase(const Key& key) {
        Path path;
        LeafNode* leaf = descend(key, &path);
        const std::size_t pos = lower_bound_idx(leaf->keys, key);
        if (pos >= leaf->keys.size() || !(leaf->keys[pos] == key)) return false;

        leaf->keys.erase(leaf->keys.begin() + static_cast<std::ptrdiff_t>(pos));
        leaf->values.erase(leaf->values.begin() + static_cast<std::ptrdiff_t>(pos));
        --count_;

        // Note: erasing a leaf's smallest key leaves an ancestor separator
        // pointing at a key that no longer exists. That is deliberate and
        // harmless — see the separator invariant documented on InternalNode.
        rebalance(path, leaf);
        return true;
    }

    // Values for all keys in [lo, hi], ascending. Walks the leaf chain.
    std::vector<Value> range(const Key& lo, const Key& hi) const {
        std::vector<Value> out;
        if (hi < lo) return out;

        const LeafNode* leaf = descend(lo);
        std::size_t i = lower_bound_idx(leaf->keys, lo);
        for (; leaf != nullptr; leaf = leaf->next, i = 0) {
            for (; i < leaf->keys.size(); ++i) {
                if (hi < leaf->keys[i]) return out;
                out.push_back(leaf->values[i]);
            }
        }
        return out;
    }

    // Uniform predicate interface shared with the learned index (see
    // CompareOp). Results are ascending by key.
    std::vector<Value> range_query(CompareOp op, const Key& value) const {
        switch (op) {
            case CompareOp::Eq: {
                std::vector<Value> out;
                if (const Value* v = find(value)) out.push_back(*v);
                return out;
            }
            case CompareOp::Lt: return collect_until(value, /*inclusive=*/false);
            case CompareOp::Le: return collect_until(value, /*inclusive=*/true);
            case CompareOp::Gt: return collect_from(value, /*inclusive=*/false);
            case CompareOp::Ge: return collect_from(value, /*inclusive=*/true);
        }
        return {};
    }

    // All (key, value) pairs in ascending key order.
    std::vector<std::pair<Key, Value>> items() const {
        std::vector<std::pair<Key, Value>> out;
        out.reserve(count_);
        for (const LeafNode* leaf = first_leaf(); leaf; leaf = leaf->next) {
            for (std::size_t i = 0; i < leaf->keys.size(); ++i) {
                out.emplace_back(leaf->keys[i], leaf->values[i]);
            }
        }
        return out;
    }

    std::vector<Key> keys() const {
        std::vector<Key> out;
        out.reserve(count_);
        for (const LeafNode* leaf = first_leaf(); leaf; leaf = leaf->next) {
            out.insert(out.end(), leaf->keys.begin(), leaf->keys.end());
        }
        return out;
    }

    void clear() {
        root_ = std::make_unique<LeafNode>();
        count_ = 0;
    }

    // Approximate resident footprint, in bytes: every node object plus the
    // heap its vectors actually hold (capacity, not size — that is what is
    // really occupied).
    //
    // Exists so a tree can be compared against a learned index on memory as
    // well as speed, which is half the argument for learned indexes: a tree's
    // internal nodes grow with n, while an RMI's models do not. Approximate
    // because it cannot see the allocator's own per-block overhead.
    std::size_t memory_bytes() const { return node_bytes(root_.get()); }

    // Full invariant check, for tests. Throws std::logic_error describing the
    // first violation found: key ordering within and across nodes, uniform
    // leaf depth, occupancy bounds, separator correctness, leaf-chain
    // consistency, and that the cached size matches reality.
    void validate() const {
        std::size_t leaf_depth = 0;
        bool leaf_depth_set = false;
        std::size_t counted = 0;
        validate_node(root_.get(), /*depth=*/1, /*is_root=*/true,
                      nullptr, nullptr, leaf_depth, leaf_depth_set, counted);

        if (counted != count_) {
            throw std::logic_error("validate: size " + std::to_string(count_) +
                                   " != counted " + std::to_string(counted));
        }

        // The leaf chain must visit every entry in ascending key order, and
        // prev/next must agree with each other.
        std::size_t chained = 0;
        const Key* prev_key = nullptr;
        const LeafNode* prev_leaf = nullptr;
        for (const LeafNode* leaf = first_leaf(); leaf; leaf = leaf->next) {
            if (leaf->prev != prev_leaf) {
                throw std::logic_error("validate: leaf chain prev/next disagree");
            }
            for (const auto& k : leaf->keys) {
                if (prev_key != nullptr && !(*prev_key < k)) {
                    throw std::logic_error("validate: leaf chain not ascending");
                }
                prev_key = &k;
                ++chained;
            }
            prev_leaf = leaf;
        }
        if (chained != count_) {
            throw std::logic_error("validate: leaf chain covers " +
                                   std::to_string(chained) + " of " +
                                   std::to_string(count_) + " entries");
        }
    }

private:
    // ---------------------------------------------------------------- nodes
    struct Node {
        explicit Node(bool leaf) : is_leaf(leaf) {}
        virtual ~Node() = default;
        bool is_leaf;
        std::vector<Key> keys;
    };

    struct LeafNode : Node {
        LeafNode() : Node(true) {}
        std::vector<Value> values;
        LeafNode* next = nullptr;   // right sibling; the range-scan chain
        LeafNode* prev = nullptr;   // left sibling; keeps unlink O(1)
    };

    struct InternalNode : Node {
        InternalNode() : Node(false) {}
        // Invariants: children.size() == keys.size() + 1, and keys[i] is a
        // routing boundary — every key in children[i] is < keys[i], and every
        // key in children[i+1] is >= keys[i].
        //
        // Note this is a *bound*, not an identity: keys[i] need not still be
        // present in children[i+1]. When a subtree's smallest key is erased
        // the separator is deliberately left behind, which costs nothing —
        // routing stays correct (everything remaining in the right subtree is
        // strictly greater), and a later insert of that same key routes right
        // and restores the exact match. Chasing the tighter
        // separator-equals-leftmost invariant would mean fixing up ancestors
        // on delete paths that empty a node, for no functional gain.
        std::vector<std::unique_ptr<Node>> children;
    };

    // The chain of (node, index-of-child-taken) pairs from root down to a
    // leaf, root first.
    using Path = std::vector<std::pair<InternalNode*, std::size_t>>;

    std::size_t order_;
    std::unique_ptr<Node> root_;
    std::size_t count_ = 0;

    // Internal nodes hold at most order_-1 keys; leaves hold at most order_-1
    // entries, so both overflow at the same threshold.
    std::size_t max_keys() const { return order_ - 1; }
    // Minimum keys for a *non-root* node: ceil(m/2) - 1.
    std::size_t min_keys() const { return (order_ + 1) / 2 - 1; }

    static std::size_t node_bytes(const Node* n) {
        if (n == nullptr) return 0;
        std::size_t total = n->keys.capacity() * sizeof(Key);
        if (n->is_leaf) {
            const auto* leaf = static_cast<const LeafNode*>(n);
            total += sizeof(LeafNode) + leaf->values.capacity() * sizeof(Value);
            return total;
        }
        const auto* internal = static_cast<const InternalNode*>(n);
        total += sizeof(InternalNode) +
                 internal->children.capacity() * sizeof(std::unique_ptr<Node>);
        for (const auto& child : internal->children) total += node_bytes(child.get());
        return total;
    }

    static std::size_t lower_bound_idx(const std::vector<Key>& ks, const Key& k) {
        return static_cast<std::size_t>(
            std::lower_bound(ks.begin(), ks.end(), k) - ks.begin());
    }

    // Index of the child to descend into. keys[i] is the smallest key in
    // children[i+1], so a key equal to a separator goes right.
    static std::size_t child_idx(const InternalNode* n, const Key& key) {
        return static_cast<std::size_t>(
            std::upper_bound(n->keys.begin(), n->keys.end(), key) - n->keys.begin());
    }

    // Walk to the leaf that would hold `key`, optionally recording the path.
    LeafNode* descend(const Key& key, Path* path = nullptr) {
        Node* n = root_.get();
        while (!n->is_leaf) {
            auto* in = static_cast<InternalNode*>(n);
            const std::size_t i = child_idx(in, key);
            if (path) path->emplace_back(in, i);
            n = in->children[i].get();
        }
        return static_cast<LeafNode*>(n);
    }

    const LeafNode* descend(const Key& key) const {
        const Node* n = root_.get();
        while (!n->is_leaf) {
            const auto* in = static_cast<const InternalNode*>(n);
            n = in->children[child_idx(in, key)].get();
        }
        return static_cast<const LeafNode*>(n);
    }

    const LeafNode* first_leaf() const {
        const Node* n = root_.get();
        while (!n->is_leaf) n = static_cast<const InternalNode*>(n)->children[0].get();
        return static_cast<const LeafNode*>(n);
    }

    // ------------------------------------------------------------- insert
    //
    // Leaf split: **copy-up**. The leaf keeps the left half; the right half
    // moves to a new leaf, and the new leaf's first key is *copied* up as the
    // parent's separator. The key stays in the leaf too, because leaves must
    // hold every value.
    std::pair<Key, std::unique_ptr<Node>> split_leaf(LeafNode* leaf) {
        const std::size_t mid = leaf->keys.size() / 2;

        auto right = std::make_unique<LeafNode>();
        right->keys.assign(leaf->keys.begin() + static_cast<std::ptrdiff_t>(mid),
                           leaf->keys.end());
        right->values.assign(leaf->values.begin() + static_cast<std::ptrdiff_t>(mid),
                             leaf->values.end());
        leaf->keys.resize(mid);
        leaf->values.resize(mid);

        right->next = leaf->next;
        right->prev = leaf;
        if (leaf->next) leaf->next->prev = right.get();
        leaf->next = right.get();

        const Key separator = right->keys.front();   // copied up, not moved
        return {separator, std::move(right)};
    }

    // Internal split: **push-up**. The middle key leaves this node entirely
    // and becomes the parent's separator — it is not duplicated, because
    // internal nodes only route, they do not store data. That asymmetry with
    // the leaf case is the detail worth being able to explain.
    std::pair<Key, std::unique_ptr<Node>> split_internal(InternalNode* node) {
        const std::size_t mid = node->keys.size() / 2;
        const Key up = node->keys[mid];              // pushed up, removed here

        auto right = std::make_unique<InternalNode>();
        right->keys.assign(node->keys.begin() + static_cast<std::ptrdiff_t>(mid) + 1,
                           node->keys.end());
        right->children.reserve(node->children.size() - mid - 1);
        for (std::size_t i = mid + 1; i < node->children.size(); ++i) {
            right->children.push_back(std::move(node->children[i]));
        }
        node->keys.resize(mid);
        node->children.resize(mid + 1);

        return {up, std::move(right)};
    }

    // Insert (separator, right sibling) into the deepest node on `path`,
    // splitting and cascading upward for as long as nodes overflow. If the
    // cascade runs past the root, the tree gains a level.
    void insert_into_parent(Path& path, Key separator, std::unique_ptr<Node> right) {
        while (!path.empty()) {
            auto [parent, idx] = path.back();
            path.pop_back();

            parent->keys.insert(parent->keys.begin() + static_cast<std::ptrdiff_t>(idx),
                                std::move(separator));
            parent->children.insert(
                parent->children.begin() + static_cast<std::ptrdiff_t>(idx) + 1,
                std::move(right));

            if (parent->keys.size() <= max_keys()) return;

            auto [up, new_right] = split_internal(parent);
            separator = up;
            right = std::move(new_right);
        }

        // Cascaded past the root: the old root and its new sibling become
        // children of a fresh root.
        auto new_root = std::make_unique<InternalNode>();
        new_root->keys.push_back(std::move(separator));
        new_root->children.push_back(std::move(root_));
        new_root->children.push_back(std::move(right));
        root_ = std::move(new_root);
    }

    // ------------------------------------------------------------- erase
    //
    // Walk back up the recorded path fixing underflow. Preference is borrow,
    // then merge: borrowing touches two nodes and leaves the height alone,
    // whereas merging consumes a separator from the parent and can cascade
    // upward, so it is the more expensive option.
    void rebalance(Path& path, Node* node) {
        while (!path.empty()) {
            if (node->keys.size() >= min_keys()) return;

            auto [parent, idx] = path.back();
            path.pop_back();

            if (idx > 0) {
                Node* left = parent->children[idx - 1].get();
                if (left->keys.size() > min_keys()) {
                    borrow_from_left(parent, idx, node, left);
                    return;
                }
            }
            if (idx + 1 < parent->children.size()) {
                Node* right = parent->children[idx + 1].get();
                if (right->keys.size() > min_keys()) {
                    borrow_from_right(parent, idx, node, right);
                    return;
                }
            }

            // No sibling can spare a key. Merge the pair, always folding the
            // right node into the left one so the separator at `sep_idx` is
            // unambiguously the one consumed.
            const std::size_t sep_idx = (idx > 0) ? idx - 1 : idx;
            merge_children(parent, sep_idx);

            node = parent;   // the parent lost a key; it may now underflow
        }

        collapse_root_if_needed();
    }

    void borrow_from_left(InternalNode* parent, std::size_t idx,
                          Node* node, Node* left) {
        if (node->is_leaf) {
            auto* n = static_cast<LeafNode*>(node);
            auto* l = static_cast<LeafNode*>(left);
            n->keys.insert(n->keys.begin(), l->keys.back());
            n->values.insert(n->values.begin(), l->values.back());
            l->keys.pop_back();
            l->values.pop_back();
            // Separator is a copy of the receiving leaf's new smallest key.
            parent->keys[idx - 1] = n->keys.front();
        } else {
            auto* n = static_cast<InternalNode*>(node);
            auto* l = static_cast<InternalNode*>(left);
            // Rotate through the parent: the separator descends into `node`
            // to describe the child moving across, and the sibling's last
            // key rises to take its place.
            n->keys.insert(n->keys.begin(), parent->keys[idx - 1]);
            n->children.insert(n->children.begin(), std::move(l->children.back()));
            l->children.pop_back();
            parent->keys[idx - 1] = l->keys.back();
            l->keys.pop_back();
        }
    }

    void borrow_from_right(InternalNode* parent, std::size_t idx,
                           Node* node, Node* right) {
        if (node->is_leaf) {
            auto* n = static_cast<LeafNode*>(node);
            auto* r = static_cast<LeafNode*>(right);
            n->keys.push_back(r->keys.front());
            n->values.push_back(r->values.front());
            r->keys.erase(r->keys.begin());
            r->values.erase(r->values.begin());
            parent->keys[idx] = r->keys.front();
        } else {
            auto* n = static_cast<InternalNode*>(node);
            auto* r = static_cast<InternalNode*>(right);
            n->keys.push_back(parent->keys[idx]);
            n->children.push_back(std::move(r->children.front()));
            r->children.erase(r->children.begin());
            parent->keys[idx] = r->keys.front();
            r->keys.erase(r->keys.begin());
        }
    }

    // Fold children[sep_idx + 1] into children[sep_idx], consuming the
    // separator between them.
    void merge_children(InternalNode* parent, std::size_t sep_idx) {
        Node* left = parent->children[sep_idx].get();
        std::unique_ptr<Node> right = std::move(parent->children[sep_idx + 1]);

        if (left->is_leaf) {
            auto* l = static_cast<LeafNode*>(left);
            auto* r = static_cast<LeafNode*>(right.get());
            l->keys.insert(l->keys.end(), r->keys.begin(), r->keys.end());
            l->values.insert(l->values.end(), r->values.begin(), r->values.end());
            // The separator is simply dropped — leaves carry their own keys,
            // so nothing is lost by discarding the routing copy.
            l->next = r->next;
            if (r->next) r->next->prev = l;
        } else {
            auto* l = static_cast<InternalNode*>(left);
            auto* r = static_cast<InternalNode*>(right.get());
            // Here the separator must be pulled *down* between the two key
            // runs: it is the only description of the boundary between the
            // last child of `l` and the first child of `r`.
            l->keys.push_back(parent->keys[sep_idx]);
            l->keys.insert(l->keys.end(), r->keys.begin(), r->keys.end());
            for (auto& c : r->children) l->children.push_back(std::move(c));
        }

        parent->keys.erase(parent->keys.begin() + static_cast<std::ptrdiff_t>(sep_idx));
        parent->children.erase(parent->children.begin() +
                               static_cast<std::ptrdiff_t>(sep_idx) + 1);
    }

    // A root that has lost all its keys has exactly one child left, so that
    // level is redundant and the child becomes the new root. An empty *leaf*
    // root is legal — that is simply an empty tree.
    void collapse_root_if_needed() {
        while (!root_->is_leaf && root_->keys.empty()) {
            auto* in = static_cast<InternalNode*>(root_.get());
            root_ = std::move(in->children[0]);
        }
    }

    // ------------------------------------------------------------ queries
    std::vector<Value> collect_until(const Key& bound, bool inclusive) const {
        std::vector<Value> out;
        for (const LeafNode* leaf = first_leaf(); leaf; leaf = leaf->next) {
            for (std::size_t i = 0; i < leaf->keys.size(); ++i) {
                const Key& k = leaf->keys[i];
                if (bound < k || (!inclusive && k == bound)) return out;
                out.push_back(leaf->values[i]);
            }
        }
        return out;
    }

    std::vector<Value> collect_from(const Key& bound, bool inclusive) const {
        std::vector<Value> out;
        const LeafNode* leaf = descend(bound);
        std::size_t i = lower_bound_idx(leaf->keys, bound);
        for (; leaf != nullptr; leaf = leaf->next, i = 0) {
            for (; i < leaf->keys.size(); ++i) {
                if (!inclusive && leaf->keys[i] == bound) continue;
                out.push_back(leaf->values[i]);
            }
        }
        return out;
    }

    // ----------------------------------------------------------- validate
    // `lo`/`hi` are the bounds inherited from ancestors: every key in this
    // subtree must satisfy lo <= k < hi (nullptr meaning unbounded).
    void validate_node(const Node* n, std::size_t depth, bool is_root,
                       const Key* lo, const Key* hi,
                       std::size_t& leaf_depth, bool& leaf_depth_set,
                       std::size_t& counted) const {
        for (std::size_t i = 1; i < n->keys.size(); ++i) {
            if (!(n->keys[i - 1] < n->keys[i])) {
                throw std::logic_error("validate: keys not strictly ascending");
            }
        }
        for (const auto& k : n->keys) {
            if (lo && k < *lo) throw std::logic_error("validate: key below subtree bound");
            if (hi && !(k < *hi)) throw std::logic_error("validate: key above subtree bound");
        }

        if (!is_root && n->keys.size() < min_keys()) {
            throw std::logic_error("validate: node underflow (" +
                                   std::to_string(n->keys.size()) + " < " +
                                   std::to_string(min_keys()) + ")");
        }
        if (n->keys.size() > max_keys()) {
            throw std::logic_error("validate: node overflow");
        }

        if (n->is_leaf) {
            const auto* leaf = static_cast<const LeafNode*>(n);
            if (leaf->keys.size() != leaf->values.size()) {
                throw std::logic_error("validate: leaf key/value size mismatch");
            }
            if (!leaf_depth_set) {
                leaf_depth = depth;
                leaf_depth_set = true;
            } else if (depth != leaf_depth) {
                throw std::logic_error("validate: leaves at differing depths");
            }
            counted += leaf->keys.size();
            return;
        }

        const auto* in = static_cast<const InternalNode*>(n);
        if (in->children.size() != in->keys.size() + 1) {
            throw std::logic_error("validate: children != keys + 1");
        }
        if (is_root && in->children.size() < 2) {
            throw std::logic_error("validate: internal root with < 2 children");
        }
        for (std::size_t i = 0; i < in->children.size(); ++i) {
            const Key* child_lo = (i == 0) ? lo : &in->keys[i - 1];
            const Key* child_hi = (i == in->keys.size()) ? hi : &in->keys[i];
            const Node* child = in->children[i].get();
            // The lo/hi bounds threaded through this recursion are exactly
            // the separator invariant: any key on the wrong side of an
            // ancestor's separator is caught at the node that holds it.
            validate_node(child, depth + 1, /*is_root=*/false, child_lo, child_hi,
                          leaf_depth, leaf_depth_set, counted);
        }
    }
};

} // namespace hylis::index
