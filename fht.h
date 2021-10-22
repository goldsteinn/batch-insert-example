#ifndef _FHT_H_
#define _FHT_H_

#include <assert.h>
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <cstddef>
#include <type_traits>
#include <utility>

// dbg marcos
#define FHT_PRINT(...) fprintf(stderr, __VA_ARGS__)
#define FHT_SHOW_V(X)  show_v(X, FHT_VSTR(X))

#define FHT_ASSERT(X)                                                          \
    _FHT_ASSERT_(X);                                                           \
    FHT_GURANTEE(X)

#define _FHT_ASSERT_(X)  // assert(X)

// attribute macros
#define FHT_INLINE   inline __attribute__((always_inline))
#define FHT_NOINLINE __attribute__((noinline))
#define FHT_ALIGN(X) __attribute__((aligned(X)))
#define FHT_CONST    __attribute__((const))
#define FHT_PURE     __attribute__((pure))

#define _FHT_VSTR(X)      #X
#define FHT_VSTR(X)       _FHT_VSTR(X)
#define FHT_CODE_ALIGN(X) asm volatile(".p2align " FHT_VSTR(X) " \n\t" :::)

// compiler helpers
#define FHT_LIKELY(X)   __builtin_expect(!!(X), 1)
#define FHT_UNLIKELY(X) __builtin_expect(!!(X), 0)
#define FHT_GURANTEE(X)                                                        \
    _FHT_ASSERT_(X);                                                           \
    if (!(X)) {                                                                \
        __builtin_unreachable();                                               \
    }


#define FHT_ALIGNED(ptr, alignment) __builtin_assume_aligned(ptr, alignment)
#define FHT_CACHE_ALIGNED(ptr)      FHT_ALIGNED(ptr, L1_LOAD_SIZE)
#define FHT_PREFETCH(ptr)           _mm_prefetch((void * const)(ptr), _MM_HINT_T0)
#define FHT_COND_PREFETCH(cond, ptr)                                           \
    if constexpr (cond) {                                                      \
        FHT_PREFETCH(ptr)                                                      \
    }


// just hate having to write this
#define FHT_NEW(type, dst, src) (new ((void * const)(dst)) type(std::move(src)))

namespace fht {

using tag_t = std::byte;

static constexpr uint32_t PAGE_SIZE       = 4096;
static constexpr uint32_t CACHE_LINE_SIZE = 64;
static constexpr uint32_t L1_LOAD_SIZE    = CACHE_LINE_SIZE;
static constexpr uint32_t L2_LOAD_SIZE    = 2 * CACHE_LINE_SIZE;

static constexpr uint32_t ELE_P_NODE = CACHE_LINE_SIZE;


void
show_v(__m128i v, const char * hdr) {
    uint8_t arr[16];
    _mm_storeu_si128((__m128i *)arr, v);
    fprintf(stderr, "%s -> [%x", hdr, arr[0] & 0xff);
    for (uint32_t i = 1; i < 16; ++i) {
        fprintf(stderr, ", %x", arr[i] & 0xff);
    }
    fprintf(stderr, "]\n");
}

void
show_v(__m256i v, const char * hdr) {
    uint8_t arr[32];
    _mm256_storeu_si256((__m256i *)arr, v);
    fprintf(stderr, "%s -> [%x", hdr, arr[0] & 0xff);
    for (uint32_t i = 1; i < 32; ++i) {
        fprintf(stderr, ", %x", arr[i] & 0xff);
    }
    fprintf(stderr, "]\n");
}


template<typename T, uint8_t b>
void FHT_INLINE
init_chunks(T * const start_ptr, T * end_ptr) {
#if defined(__AVX512F__) && defined(USE_AVX512)
    // generally inadvisable to use AVX512 as will lower chip
    // frequency and this isnt really super critical code
    __m512i set_bytes = _mm512_set1_epi8(b);
#elif defined(__AVX2__)
    // this is best
    __m256i set_bytes = _mm256_set1_epi8(b);
#else
    // gcc really has an insane impl of builtin memset so still use
    // this
    __m128i set_bytes = _mm_set1_epi8(b);
#endif
    for (; end_ptr > start_ptr; end_ptr -= 2) {

#if defined(__AVX512F__) && defined(USE_AVX512)
        _mm512_store_si512((__m512i *)end_ptr, set_bytes);
        _mm512_store_si512((__m512i *)(end_ptr - 1), set_bytes);
#elif defined(__AVX2__)
        _mm256_store_si256((__m256i *)end_ptr, set_bytes);
        _mm256_store_si256(((__m256i *)end_ptr) + 1, set_bytes);
        _mm256_store_si256((__m256i *)(end_ptr - 1), set_bytes);
        _mm256_store_si256(((__m256i *)(end_ptr - 1)) + 1, set_bytes);
#else
        _mm_store_si128((__m128i *)end_ptr, set_bytes);
        _mm_store_si128(((__m128i *)end_ptr) + 1, set_bytes);
        _mm_store_si128(((__m128i *)end_ptr) + 2, set_bytes);
        _mm_store_si128(((__m128i *)end_ptr) + 3, set_bytes);
        _mm_store_si128((__m128i *)(end_ptr - 1), set_bytes);
        _mm_store_si128(((__m128i *)(end_ptr - 1)) + 1, set_bytes);
        _mm_store_si128(((__m128i *)(end_ptr - 1)) + 2, set_bytes);
        _mm_store_si128(((__m128i *)(end_ptr - 1)) + 3, set_bytes);
#endif
    }
}

struct SIMD_helper_128 {
    using vec_t = __m128i;

    static vec_t FHT_INLINE FHT_CONST
    set_vec(const tag_t b) {
        return _mm_set1_epi8(uint8_t(b));
    }

    static vec_t FHT_INLINE FHT_CONST
    set_zero() {
        // vpxor
        return _mm_setzero_si128();
    }

    static uint32_t FHT_INLINE FHT_CONST
    is_zero(vec_t v) {
        return _mm_testz_si128(v, v);
    }

    static vec_t FHT_INLINE FHT_PURE
    load_vec(const tag_t * const b) {
        FHT_GURANTEE(((uint64_t)b) % sizeof(vec_t) == 0);
        return _mm_load_si128((vec_t *)b);
    }

    static void FHT_INLINE
    store_vec(const tag_t * const b, vec_t v) {
        FHT_GURANTEE(((uint64_t)b) % sizeof(vec_t) == 0);
        _mm_store_si128((vec_t *)b, v);
    }

    static vec_t FHT_INLINE FHT_CONST
    match(vec_t v1, vec_t v2) {
        return _mm_cmpeq_epi8(v1, v2);
    }

    static vec_t FHT_INLINE FHT_CONST
    less_than(vec_t v1, vec_t v2) {
        return _mm_cmplt_epi8(v1, v2);
    }

    static FHT_INLINE FHT_CONST uint32_t
    byte_mask(vec_t v) {
        return _mm_movemask_epi8(v);
    }

    static FHT_INLINE FHT_CONST uint32_t
    match_mask(vec_t v1, vec_t v2) {
        return byte_mask(match(v1, v2));
    }

    static FHT_INLINE FHT_CONST uint32_t
    lt_mask(vec_t v1, vec_t v2) {
        return byte_mask(less_than(v1, v2));
    }
};

struct SIMD_helper_256 {
    using vec_t = __m256i;

    static vec_t FHT_INLINE FHT_CONST
    set_vec(const tag_t b) {
        return _mm256_set1_epi8(uint8_t(b));
    }

    static vec_t FHT_INLINE FHT_CONST
    set_zero() {
        // vpxor
        return _mm256_setzero_si256();
    }

    static vec_t FHT_INLINE FHT_PURE
    load_vec(const tag_t * const b) {
        FHT_GURANTEE(((uint64_t)b) % sizeof(vec_t) == 0);
        return _mm256_load_si256((vec_t *)b);
    }

    static void FHT_INLINE
    store_vec(const tag_t * const b, vec_t v) {
        FHT_GURANTEE(((uint64_t)b) % sizeof(vec_t) == 0);
        _mm256_store_si256((vec_t *)b, v);
    }

    static vec_t FHT_INLINE FHT_CONST
    match(vec_t v1, vec_t v2) {
        return _mm256_cmpeq_epi8(v1, v2);
    }

    static vec_t FHT_INLINE FHT_CONST
    less_than(vec_t v1, vec_t v2) {
        return _mm256_cmpgt_epi8(v1, v2);
    }

    static FHT_INLINE FHT_CONST uint32_t
    byte_mask(vec_t v) {
        return _mm256_movemask_epi8(v);
    }

    static FHT_INLINE FHT_CONST uint32_t
    match_mask(vec_t v1, vec_t v2) {
        return byte_mask(match(v1, v2));
    }

    static FHT_INLINE FHT_CONST uint32_t
    lt_mask(vec_t v1, vec_t v2) {
        return ~byte_mask(less_than(v1, v2));
    }
};
using SIMD_helper = SIMD_helper_128;

template<typename T>
struct BITS_helper {

    // find first one
    static T FHT_INLINE FHT_CONST
    ff1(const T v) {
        if constexpr (sizeof(T) == sizeof(uint64_t)) {
            return _tzcnt_u64(v);
        }
        else {
            return _tzcnt_u32(v);
        }
    }

    // drop first one
    static T FHT_INLINE FHT_CONST
    df1(const T v) {
        return v & (v - 1);
    }

    static T FHT_INLINE FHT_CONST
    bitcount(const T v) {
        if constexpr (sizeof(T) == sizeof(uint64_t)) {
            return _popcnt64(v);
        }
        else {
            return _popcnt32(v);
        }
    }

    static T FHT_INLINE FHT_CONST
    first_nbits(const T v, const uint32_t n) {
        return v & (T(1) << n);
    }

    static T FHT_INLINE FHT_CONST
    nth_bit(const T v, const uint32_t n) {
        return (v >> n) & 0x1;
    }

    // intentionally not specify for inline as this is far from critical path
    // and
    // compiler knows better if its going to bloat executable
    static constexpr T FHT_CONST
    log_p2(T v) {
        T s = 0, t = 0;
        if constexpr (sizeof(T) == sizeof(uint64_t)) {
            t = (v > 0xffffffffUL) << 5;
            v >>= t;
        }
        t = (v > 0xffffUL) << 4;
        v >>= t;
        s = (v > 0xffUL) << 3;
        v >>= s, t |= s;
        s = (v > 0xfUL) << 2;
        v >>= s, t |= s;
        s = (v > 0x3UL) << 1;
        v >>= s, t |= s;
        return (t | (v >> 1));
    }

    // intentionally not specify for inline as this is far from critical path
    // and compiler knows better if its going to bloat executable
    static constexpr T FHT_CONST
    next_p2(T v) {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        if constexpr (sizeof(T) == sizeof(uint64_t)) {
            v |= v >> 32;
        }
        v++;
        return v;
    }
};

struct fht_tag_ops {
    using vec_t = SIMD_helper::vec_t;

    static constexpr tag_t invalid = tag_t(0xff);
    static constexpr tag_t erased  = tag_t(0xc0);
    static constexpr tag_t content = tag_t(0x7f);

    static constexpr uint32_t content_bits = 7;


    static uint32_t FHT_INLINE FHT_CONST
    match_tag_vecs(vec_t v1, vec_t v2) {
        return SIMD_helper::match_mask(v1, v2);
    }

    template<uint32_t unique = 0>
    static uint32_t FHT_INLINE FHT_CONST
    get_available_mask(vec_t v) {
        if constexpr (unique) {
            return get_placeable_mask(v);
        }
        else {
            return SIMD_helper::match_mask(v, SIMD_helper::set_vec(invalid));
        }
    }


    static uint32_t FHT_INLINE FHT_CONST
    get_placeable_mask(vec_t v) {
        return SIMD_helper::byte_mask(v);
    }

    static uint32_t FHT_INLINE FHT_CONST
    resize_skip(tag_t tag) {
        return uint8_t((tag & tag_t(0x80))) != 0;
    }
};

template<typename hash_t>
struct fht_hash_val_ops {
    using bits_helper = BITS_helper<hash_t>;
    using vec_t       = SIMD_helper::vec_t;

    static constexpr uint32_t tag_bits = fht_tag_ops::content_bits;
    static constexpr uint32_t start_idx_bits =
        bits_helper::log_p2(ELE_P_NODE / sizeof(vec_t));

    static constexpr hash_t   tag_mask        = (hash_t(1) << tag_bits) - 1;
    static constexpr uint32_t start_idx_shift = tag_bits;
    static constexpr hash_t start_idx_mask = (hash_t(1) << start_idx_bits) - 1;

    static constexpr uint32_t meta_data_bits = tag_bits + start_idx_bits;
    static constexpr hash_t meta_data_mask = (hash_t(1) << meta_data_bits) - 1;

    static constexpr hash_t FHT_INLINE FHT_CONST
    get_not_slot(const hash_t hv) {
        return hv & meta_data_mask;
    }
    static constexpr hash_t FHT_INLINE FHT_CONST
    get_slot_n(const hash_t old_slot, const hash_t mask, const uint32_t n) {
        return ((old_slot + n)) & mask;
    }

    static constexpr tag_t FHT_INLINE FHT_CONST
    get_tag(const hash_t hv) {
        return tag_t(hv & tag_mask);
    }

    static constexpr uint32_t FHT_INLINE FHT_CONST
    get_resize_start_idx(const hash_t hv) {
        return (uint32_t(hv >> start_idx_shift) & start_idx_mask);
    }

    static constexpr uint32_t FHT_INLINE FHT_CONST
    get_start_idx(const hash_t hv) {
        return sizeof(vec_t) * get_resize_start_idx(hv);
    }


    static constexpr hash_t FHT_INLINE FHT_CONST
    get_slot_shift(const hash_t hv) {
        return hv >> meta_data_bits;
    }

    static constexpr hash_t FHT_INLINE FHT_CONST
    get_slot(const hash_t hv, const hash_t mask) {
        return get_slot_shift(hv) & mask;
    }

    static constexpr hash_t FHT_INLINE FHT_CONST
    next_mask(const hash_t old_mask) {
        return (old_mask << 1) | hash_t(1);
    }

    static constexpr hash_t FHT_INLINE FHT_CONST
    next_mask_n(const hash_t old_mask, const uint32_t n) {
        return ((old_mask + hash_t(1)) << n) - 1;
    }
};

template<typename K, typename V>
struct fht_pair {
    K k;
    V v;
};

template<typename K, typename V>
struct fht_node2 {
    fht_pair<K, V> kv[ELE_P_NODE];

    void FHT_INLINE
    prefetch_idx(const uint32_t n) const {
        FHT_PREFETCH(kv + n);
    }

    K FHT_INLINE FHT_PURE
    get_key_n(const uint32_t n) const {
        return kv[n].k;
    }

    FHT_INLINE FHT_PURE K *
                        get_key_n_ptr(const uint32_t n) {
        return &(kv[n].k);
    }

    V FHT_INLINE FHT_PURE
    get_val_n(const uint32_t n) const {
        return kv[n].v;
    }

    FHT_INLINE FHT_PURE V *
                        get_val_n_ptr(const uint32_t n) {
        return &(kv[n].v);
    }

    void FHT_INLINE
    copy_idx(const fht_node2<K, V> * const other,
             const uint32_t                from_idx,
             const uint32_t                to_idx) {
        __builtin_memcpy(
            FHT_ALIGNED(kv + to_idx, sizeof(fht_pair<K, V>)),
            FHT_ALIGNED(other->kv + from_idx, sizeof(fht_pair<K, V>)),
            sizeof(fht_pair<K, V>));
    }
};

template<typename K, typename V>
struct fht_node {
    K keys[ELE_P_NODE];
    V vals[ELE_P_NODE];

    void FHT_INLINE
    prefetch_idx(const uint32_t n) const {
        FHT_PREFETCH(keys + n);
        FHT_PREFETCH(vals + n);
    }

    K FHT_INLINE FHT_PURE
    get_key_n(const uint32_t n) const {
        return keys[n];
    }

    FHT_INLINE FHT_PURE K *
                        get_key_n_ptr(const uint32_t n) {
        return keys + n;
    }

    V FHT_INLINE FHT_PURE
    get_val_n(const uint32_t n) const {
        return vals[n];
    }

    FHT_INLINE FHT_PURE V *
                        get_val_n_ptr(const uint32_t n) {
        return vals + n;
    }

    void FHT_INLINE
    copy_idx(const fht_node<K, V> * const other,
             const uint32_t               from_idx,
             const uint32_t               to_idx) {
        __builtin_memcpy(FHT_ALIGNED(keys + to_idx, sizeof(K)),
                         other->keys + from_idx, sizeof(K));
        __builtin_memcpy(FHT_ALIGNED(vals + to_idx, sizeof(V)),
                         other->vals + from_idx, sizeof(V));
    }
};


template<typename K, typename V>
struct fht_chunk {
    using vec_t = SIMD_helper::vec_t;

    using node_t = fht_node2<K, V>;
    // No reason for bytes to be aliasing
    tag_t  tags[ELE_P_NODE];
    node_t nodes;

    //////////////////////////////////////////////////////////////////////
    // Prefetch
    void FHT_INLINE
    prefetch_idx(const uint32_t n) const {
        nodes.prefetch_idx(n);
    }
    //////////////////////////////////////////////////////////////////////
    // Getters
    tag_t FHT_INLINE FHT_PURE
    get_tag_n(const uint32_t n) const {
        FHT_GURANTEE(n < ELE_P_NODE);

        return tags[n];
    }

    uint32_t FHT_INLINE FHT_PURE
    skip_tag_n(const uint32_t n) const {
        FHT_GURANTEE(n < ELE_P_NODE);

        return fht_tag_ops::resize_skip(tags[n]);
    }

    K FHT_INLINE FHT_PURE
    get_key_n(const uint32_t n) const {
        FHT_GURANTEE(n < ELE_P_NODE);
        return nodes.get_key_n(n);
    }

    V FHT_INLINE FHT_PURE
    get_val_n(const uint32_t n) const {
        FHT_GURANTEE(n < ELE_P_NODE);
        return nodes.get_val_n(n);
    }

    void FHT_INLINE
    copy_idx(const fht_chunk<K, V> * const other,
             const uint32_t                from_idx,
             const uint32_t                to_idx) {
        set_tag_n(other->get_tag_n(from_idx), to_idx);
        nodes.copy_idx(&(other->nodes), from_idx, to_idx);
    }

    FHT_INLINE FHT_PURE K *
                        get_key_n_ptr(const uint32_t n) {
        FHT_GURANTEE(n < ELE_P_NODE);
        return nodes.get_key_n_ptr(n);
    }

    FHT_INLINE FHT_PURE V *
                        get_val_n_ptr(const uint32_t n) {
        FHT_GURANTEE(n < ELE_P_NODE);
        return nodes.get_val_n_ptr(n);
    }


    //////////////////////////////////////////////////////////////////////
    // Setter
    void FHT_INLINE
    set_tag_n(const tag_t tag, const uint32_t n) {
        FHT_GURANTEE(n < ELE_P_NODE);
        tags[n] = tag;
    }

    void FHT_INLINE
    set_tag_n_v(const vec_t tag_v, const uint32_t off) {
        _mm_store_si128((__m128i *)(tags + off), tag_v);
    }

    //////////////////////////////////////////////////////////////////////
    // Vec logic
    vec_t FHT_INLINE FHT_PURE
    load_tags_o(const uint32_t offset) const {
        FHT_GURANTEE(offset % sizeof(vec_t) == 0);
        FHT_GURANTEE(((uint64_t)tags) % sizeof(vec_t) == 0);

        return SIMD_helper::load_vec(tags + offset);
    }

    uint64_t FHT_INLINE FHT_PURE
    get_valid_indexes() const {
        FHT_GURANTEE(((uint64_t)tags) % L1_LOAD_SIZE == 0);
        __m256i  v0 = _mm256_load_si256((__m256i *)tags);
        __m256i  v1 = _mm256_load_si256((__m256i *)(tags + sizeof(__m256i)));
        uint64_t m0 = _mm256_movemask_epi8(v0);
        uint64_t m1 = _mm256_movemask_epi8(v1);
        return ~((m0 & 0xffffffff) | (m1 << sizeof(__m256i)));
    }


    // if sizeof(K) + sizeof(V) is odd we would want L2_LOAD_SIZE
} FHT_ALIGN(L1_LOAD_SIZE);


template<typename K, typename V, typename Hasher, uint64_t rehash_attempts = 3>
struct fht_table {
    static constexpr uint32_t failure = 1;
    static constexpr uint32_t success = 0;

    static constexpr uint64_t default_init_size = 64;

    static constexpr Hasher hasher = Hasher{};

    using key_t = K;
    using val_t = V;

    using chunk_t                          = fht_chunk<K, V>;
    using vec_t                            = SIMD_helper::vec_t;
    static constexpr uint32_t LINES_P_NODE = ELE_P_NODE / sizeof(vec_t);

    using hash_t       = uint64_t;
    using hash_val_ops = fht_hash_val_ops<hash_t>;
    static_assert(((1u) << hash_val_ops::start_idx_bits) == LINES_P_NODE);

    using tag_ops = fht_tag_ops;

    hash_t    cur_mask;
    chunk_t * chunks;

    void
    init() {
        init(default_init_size);
    }

    void
    init(const uint64_t isize) {
        const uint64_t ichunks =
            (BITS_helper<uint64_t>::next_p2(isize) / ELE_P_NODE);

        chunks =
            (chunk_t *)aligned_alloc(L1_LOAD_SIZE, ichunks * sizeof(chunk_t));
        assert(chunks != NULL);

        if (ichunks < 2) {
            for (uint32_t i = 0; i < ichunks; ++i) {
                __builtin_memset(FHT_CACHE_ALIGNED(chunks[i].tags),
                                 uint8_t(fht_tag_ops::invalid), ELE_P_NODE);
            }
        }
        else {
            init_chunks<chunk_t, uint8_t(fht_tag_ops::invalid)>(
                chunks, chunks + (ichunks - 1));
        }

        cur_mask = (ichunks - 1);
    }

    void
    destroy() {
        if (chunks) {
            free(chunks);
        }
    }

    void
    prep_insert_n(const uint32_t n) {
        const hash_t _cur_mask = cur_mask;

        if ((n / ELE_P_NODE) > cur_mask) {
            const uint32_t scale =
                (n / ELE_P_NODE) >> BITS_helper<hash_t>::bitcount(_cur_mask);
            const uint32_t log_scale =
                BITS_helper<hash_t>::log_p2((scale << 1) - 1);
            resize(log_scale);
        }
    }


    chunk_t * FHT_NOINLINE
    resize(const uint32_t skips) {
        // No batching here as the VAST VAST majority of inserts find
        // space with first attempt so for the most part we have high
        // cache hit rate on placement

        const hash_t   new_mask    = hash_val_ops::next_mask_n(cur_mask, skips);
        const uint32_t old_nchunks = cur_mask + 1;
        const uint32_t new_nchunks = old_nchunks << skips;
        FHT_GURANTEE(new_nchunks);

        const chunk_t * const old_chunks = chunks;
        chunk_t * const       new_chunks = (chunk_t * const)aligned_alloc(
            L1_LOAD_SIZE, new_nchunks * sizeof(chunk_t));


        // might get some slightly better caching behavior here
#if 0
                for (uint32_t i = new_nchunks; i; --i) {
            __builtin_memset(FHT_CACHE_ALIGNED((new_chunks + (i - 1))->tags),
                             uint8_t(fht_tag_ops::invalid), ELE_P_NODE);
        }
#else
        init_chunks<chunk_t, uint8_t(fht_tag_ops::invalid)>(
            new_chunks, new_chunks + (new_nchunks - 1));
#endif

        FHT_PREFETCH(old_chunks);
        FHT_PREFETCH(new_chunks + old_nchunks);
        FHT_GURANTEE(old_nchunks);

        for (uint32_t i = 0; i < old_nchunks; ++i) {
            const chunk_t * const old_chunk = old_chunks + i;

            FHT_PREFETCH(new_chunks + i + 1);
            FHT_PREFETCH(new_chunks + old_nchunks + i + 1);

            uint64_t valid_indexes = old_chunk->get_valid_indexes();
            for (; valid_indexes;) {
                uint32_t j    = BITS_helper<uint64_t>::ff1(valid_indexes);
                valid_indexes = BITS_helper<uint64_t>::df1(valid_indexes);

                const hash_t raw_hash = hasher(old_chunk->get_key_n(j));
                hash_t       slot = hash_val_ops::get_slot(raw_hash, new_mask);

                const uint32_t start_idx =
                    hash_val_ops::get_start_idx(raw_hash);
                chunk_t * const new_chunk = new_chunks + slot;

                uint32_t attempt = 0;
                for (;;) {
                    for (uint32_t k = 0; k < ELE_P_NODE; k += sizeof(vec_t)) {
                        const uint32_t cur_offset =
                            (k + start_idx) % ELE_P_NODE;

                        vec_t c_tags_v = new_chunk->load_tags_o(cur_offset);
                        const uint32_t available_map =
                            fht_tag_ops::get_placeable_mask(c_tags_v);

                        if (FHT_LIKELY(available_map != 0)) {
                            const uint32_t available_slot =
                                BITS_helper<uint32_t>::ff1(available_map) +
                                cur_offset;
                            new_chunk->copy_idx(old_chunk, j, available_slot);
                            FHT_ASSERT(new_chunk->get_tag_n(available_slot) ==
                                       old_chunk->get_tag_n(j));
                            FHT_ASSERT(
                                !(uint8_t(old_chunk->get_tag_n(j)) & 0x80));
                            goto endloop;
                        }
                    }
                    if (attempt == rehash_attempts) {
                        //                        break;
                        FHT_GURANTEE(0);
                    }
                    ++attempt;
                    slot = hash_val_ops::get_slot_n(slot, new_mask, attempt);
                }
                FHT_GURANTEE(0);

            endloop:
                __attribute__((hot));
            }
        }

        free(chunks);
        cur_mask = new_mask;
        chunks   = new_chunks;
        return new_chunks;
    }


    val_t *
    emplace(key_t k, val_t v) {
        return insert(k, v);
    }


    struct insert_frame {
        using vec_t = SIMD_helper::vec_t;

        static constexpr uint64_t att_shift = 48;
        static constexpr uint64_t slot_mask =
            rehash_attempts ? ((1ul << att_shift) - 1) : (~(0UL));

        uint64_t raw_slot_and_att;
        uint32_t tag_and_start;
        uint32_t idx;


        static constexpr FHT_INLINE FHT_CONST uint64_t
        reset_slot(const uint64_t slot, const uint32_t _att) {
            if constexpr (rehash_attempts == 0) {
                return slot;
            }
            else if constexpr (rehash_attempts == 1) {
                return (slot - _att) & slot_mask;
            }
            else if constexpr (rehash_attempts == 2) {
                return (slot - (_att | (_att >> 1))) & slot_mask;
            }
            else {
                // static constexpr uint32_t att_map[] = { 0, 1, 3, 6, 10 };
                return (slot - ((_att * (_att + 1)) / 2)) & slot_mask;
            }
        }

        static constexpr FHT_INLINE FHT_CONST uint64_t
        get_slot(const uint64_t slot, const uint64_t mask) {
            return slot & mask;
        }

        static constexpr FHT_INLINE FHT_CONST uint32_t
        get_att(const uint64_t slot) {
            if constexpr (rehash_attempts == 0) {
                return 0;
            }
            else {
                return uint32_t(slot >> att_shift);
            }
        }

        static constexpr FHT_INLINE FHT_CONST uint64_t
        next_att(const uint32_t att) {
            if constexpr (rehash_attempts == 0) {
                return 0;
            }
            else {
                return (uint64_t(1) << att_shift) | uint64_t(att);
            }
        }

        static constexpr FHT_INLINE FHT_CONST uint32_t
        get_start_idx(const uint32_t _tag_and_start) {
            return sizeof(vec_t) *
                   (_tag_and_start >> hash_val_ops::start_idx_shift);
        }

        static constexpr FHT_INLINE FHT_CONST tag_t
        get_tag(const uint32_t _tag_and_start) {
            return tag_t(_tag_and_start & hash_val_ops::tag_mask);
        }
    };

    template<typename batch_t, uint32_t stream_n, uint32_t unique = 0>
    FHT_ALIGN(32)
    FHT_NOINLINE void batch_insert(const batch_t  batch,
                                   const uint32_t n,
                                   val_t ** const ret) {
        FHT_GURANTEE(n > 0);
        static_assert(stream_n <= 64 && stream_n > 1);

        using bvec_t =
            typename std::conditional_t<(stream_n > 32), uint64_t, uint32_t>;

        constexpr bvec_t all_frames = stream_n == 8 * sizeof(bvec_t)
                                          ? ~bvec_t(0)
                                          : (bvec_t(1) << stream_n) - 1;

        bvec_t outstanding_frames = all_frames;

        hash_t    _cur_mask = cur_mask;
        chunk_t * _chunks   = chunks;

        insert_frame frames[stream_n];
        uint32_t     batch_idx = 0;

        // scope incase compiler fails to see that init_nbound is
        // unused after init
        {
            uint32_t init_nbound = stream_n;
            if (n < stream_n) {
                init_nbound = n;
                outstanding_frames >>= (stream_n - n);
            }


            for (; batch_idx < init_nbound; ++batch_idx) {
                const hash_t   raw_hash = hasher(batch.key(batch_idx));
                const uint64_t raw_slot =
                    hash_val_ops::get_slot_shift(raw_hash);
                const uint32_t not_slot = hash_val_ops::get_not_slot(raw_hash);

                const uint32_t slot =
                    insert_frame::get_slot(raw_slot, _cur_mask);


                FHT_PREFETCH(_chunks + slot);
                FHT_PREFETCH(uint64_t((_chunks + slot)) + ELE_P_NODE +
                             sizeof(typename chunk_t::node_t) *
                                 hash_val_ops::get_start_idx(raw_hash));


                frames[batch_idx] = { .raw_slot_and_att = raw_slot,
                                      .tag_and_start    = not_slot,
                                      .idx              = batch_idx };
            }
        }

        for (;;) {
            uint32_t frame_idx;
            bvec_t   b;
            for (frame_idx = 0, b = 1; frame_idx < stream_n;
                 ++frame_idx, b <<= 1) {
                if (FHT_UNLIKELY(!(outstanding_frames & b))) {
                    continue;
                }

                const uint64_t raw_slot = frames[frame_idx].raw_slot_and_att;
                const uint32_t tag_and_start = frames[frame_idx].tag_and_start;
                const uint32_t idx           = frames[frame_idx].idx;


                const tag_t tag   = insert_frame::get_tag(tag_and_start);
                const vec_t tag_v = SIMD_helper::set_vec(tag);

                const uint32_t start_idx =
                    insert_frame::get_start_idx(tag_and_start);
                chunk_t * const chunk =
                    _chunks + insert_frame::get_slot(raw_slot, _cur_mask);

                uint32_t cur_offset = start_idx;

                for (;;) {
                    const vec_t ctags_v = chunk->load_tags_o(cur_offset);
                    if constexpr (!unique) {
                        uint32_t matches_map =
                            fht_tag_ops::match_tag_vecs(ctags_v, tag_v);

                        for (; matches_map;
                             matches_map =
                                 BITS_helper<uint32_t>::df1(matches_map)) {
                            const uint32_t match_idx =
                                BITS_helper<uint32_t>::ff1(matches_map) +
                                cur_offset;


                            FHT_ASSERT(tag == chunk->get_tag_n(match_idx));
                            FHT_GURANTEE(match_idx < ELE_P_NODE);
                            if (FHT_LIKELY(chunk->get_key_n(match_idx) ==
                                           batch.key(idx))) {
                                if (ret) {
                                    ret[idx] = chunk->get_val_n_ptr(match_idx);
                                }
                                goto endloop;
                            }
                        }
                    }

                    const uint32_t available_map =
                        fht_tag_ops::get_available_mask<unique>(ctags_v);
                    if (FHT_LIKELY(available_map != 0)) {
                        const uint32_t available_slot =
                            BITS_helper<uint32_t>::ff1(available_map) +
                            cur_offset;
                        // store
                        FHT_GURANTEE(available_slot < ELE_P_NODE);
                        chunk->set_tag_n(tag, available_slot);
                        FHT_NEW(key_t, chunk->get_key_n_ptr(available_slot),
                                batch.key(idx));
                        FHT_NEW(val_t, chunk->get_val_n_ptr(available_slot),
                                batch.val(idx));

                        FHT_ASSERT(chunk->get_tag_n(available_slot) == tag);
                        goto endloop;
                    }

                    cur_offset = (cur_offset + sizeof(vec_t)) % ELE_P_NODE;
                    if (FHT_UNLIKELY(cur_offset == start_idx)) {
                        break;
                    }
                }
                {
                    const uint32_t att = insert_frame::get_att(raw_slot);
                    FHT_GURANTEE(att <= rehash_attempts);
                    if (rehash_attempts == 0 ||
                        FHT_UNLIKELY(att == rehash_attempts)) {
                        // to resize need to recompute slot for all active
                        // frames

                        _chunks   = resize(1);
                        _cur_mask = hash_val_ops::next_mask(_cur_mask);
                        for (frame_idx = 0, b = 1; frame_idx < stream_n;
                             ++frame_idx, b <<= 1) {
                            if (FHT_UNLIKELY(!(outstanding_frames & b))) {
                                continue;
                            }

                            const uint64_t old_raw_slot =
                                frames[frame_idx].raw_slot_and_att;
                            const uint64_t new_raw_slot =
                                insert_frame::reset_slot(
                                    old_raw_slot,
                                    insert_frame::get_att(old_raw_slot));

                            const uint64_t new_slot =
                                insert_frame::get_slot(new_raw_slot, _cur_mask);


                            FHT_PREFETCH(_chunks + new_slot);
                            if constexpr (rehash_attempts != 0) {
                                frames[frame_idx].raw_slot_and_att =
                                    new_raw_slot;
                            }
                        }
                        continue;
                    }
                    else {

                        const uint64_t new_slot = hash_val_ops::get_slot_n(
                            raw_slot, _cur_mask, att + 1);

                        FHT_PREFETCH(_chunks + new_slot);
                        FHT_PREFETCH(
                            uint64_t((_chunks + new_slot)) + ELE_P_NODE +
                            sizeof(typename chunk_t::node_t) * start_idx);

                        frames[frame_idx].raw_slot_and_att +=
                            insert_frame::next_att(att + 1);


                        continue;
                    }
                }

            endloop:
                __attribute__((hot));
                FHT_ASSERT(batch_idx <= n);
                if (FHT_UNLIKELY(batch_idx >= n)) {
                    // this might be necessary for the compiler to
                    // know that the while loop will exit
                    if ((outstanding_frames -= b) == 0) {
                        return;
                    }
                }
                else {
                    const hash_t   new_raw_hash = hasher(batch.key(batch_idx));
                    const uint64_t new_raw_slot =
                        hash_val_ops::get_slot_shift(new_raw_hash);
                    const uint32_t not_slot =
                        hash_val_ops::get_not_slot(new_raw_hash);

                    FHT_PREFETCH(_chunks + insert_frame::get_slot(new_raw_slot,
                                                                  _cur_mask));


                    frames[frame_idx] = { .raw_slot_and_att = new_raw_slot,
                                          .tag_and_start    = not_slot,
                                          .idx              = batch_idx };

                    ++batch_idx;
                }
            }
        }
        FHT_GURANTEE(0);
    }

    template<uint32_t unique = 0>
    FHT_NOINLINE
    FHT_ALIGN(32) val_t * insert(key_t k, val_t v) {
        chunk_t * const _chunks   = chunks;
        const hash_t    _cur_mask = cur_mask;
        const hash_t    raw_hash  = hasher(k);
        hash_t          slot      = hash_val_ops::get_slot(raw_hash, _cur_mask);
        FHT_PREFETCH(chunks + slot);
        const uint32_t start_idx = hash_val_ops::get_start_idx(raw_hash);
        chunk_t *      chunk     = _chunks + slot;
        chunk->prefetch_idx(start_idx);

        const tag_t tag     = hash_val_ops::get_tag(raw_hash);
        const vec_t tag_v   = SIMD_helper::set_vec(tag);
        uint32_t    attempt = 0;
        for (;;) {


            uint32_t cur_offset = start_idx;
            do {
                vec_t ctags_v = chunk->load_tags_o(cur_offset);

                if constexpr (!unique) {
                    uint32_t matches_map =
                        fht_tag_ops::match_tag_vecs(ctags_v, tag_v);

                    for (; matches_map;
                         matches_map =
                             BITS_helper<uint32_t>::df1(matches_map)) {
                        const uint32_t match_idx =
                            BITS_helper<uint32_t>::ff1(matches_map) +
                            cur_offset;


                        if (FHT_LIKELY(chunk->get_key_n(match_idx) == k)) {
                            return chunk->get_val_n_ptr(match_idx);
                        }
                    }
                }

                const uint32_t available_map =
                    fht_tag_ops::get_available_mask<unique>(ctags_v);
                if (FHT_LIKELY(available_map != 0)) {
                    const uint32_t available_slot =
                        BITS_helper<uint32_t>::ff1(available_map) + cur_offset;
                    // store
                    chunk->set_tag_n(tag, available_slot);
                    FHT_NEW(key_t, chunk->get_key_n_ptr(available_slot), k);
                    FHT_NEW(val_t, chunk->get_val_n_ptr(available_slot), v);

                    return NULL;
                }

                cur_offset = (cur_offset + sizeof(vec_t)) % ELE_P_NODE;
            } while (FHT_UNLIKELY(cur_offset != start_idx));

            if (FHT_UNLIKELY(attempt == rehash_attempts)) {
                break;
            }
            ++attempt;
            slot  = hash_val_ops::get_slot_n(slot, _cur_mask, attempt);
            chunk = _chunks + slot;

            FHT_PREFETCH(chunk);
        }

        const hash_t new_mask = hash_val_ops::next_mask(_cur_mask);
        slot                  = hash_val_ops::get_slot(raw_hash, new_mask);
        chunk                 = resize(1) + slot;

        attempt = 0;
        for (;;) {
            uint32_t cur_offset = start_idx;

            do {
                vec_t          c_tags_v = chunk->load_tags_o(cur_offset);
                const uint32_t available_map =
                    fht_tag_ops::get_placeable_mask(c_tags_v);

                if (FHT_LIKELY(available_map != 0)) {
                    const uint32_t available_slot =
                        BITS_helper<uint32_t>::ff1(available_map) + cur_offset;

                    chunk->set_tag_n(tag, available_slot);
                    FHT_NEW(key_t, chunk->get_key_n_ptr(available_slot), k);
                    FHT_NEW(val_t, chunk->get_val_n_ptr(available_slot), v);

                    return NULL;
                }

                cur_offset = (cur_offset + sizeof(vec_t)) % ELE_P_NODE;
            } while (FHT_UNLIKELY(cur_offset != start_idx));


            if (FHT_UNLIKELY(attempt == rehash_attempts)) {
                FHT_GURANTEE(0);
            }

            slot  = hash_val_ops::get_slot_n(slot, new_mask, ++attempt);
            chunk = chunks + slot;
            FHT_PREFETCH(chunk);
        }
    }

    FHT_PURE FHT_NOINLINE
    FHT_ALIGN(32) V * find(key_t k) const {
        chunk_t * const _chunks   = chunks;
        const hash_t    _cur_mask = cur_mask;

        const hash_t raw_hash = hasher(k);
        hash_t       slot     = hash_val_ops::get_slot(raw_hash, _cur_mask);
        // FHT_PREFETCH(chunks + slot);

        const uint32_t start_idx = hash_val_ops::get_start_idx(raw_hash);

        const vec_t tag_v =
            SIMD_helper::set_vec(hash_val_ops::get_tag(raw_hash));
        uint32_t attempt = 0;

        for (;;) {
            chunk_t * const chunk      = _chunks + slot;
            uint32_t        cur_offset = start_idx;
            for (;;) {
                vec_t    ctags_v = chunk->load_tags_o(cur_offset);
                uint32_t matches_map =
                    fht_tag_ops::match_tag_vecs(ctags_v, tag_v);
                for (; matches_map;
                     matches_map = BITS_helper<uint32_t>::df1(matches_map)) {
                    const uint32_t match_idx =
                        BITS_helper<uint32_t>::ff1(matches_map) + cur_offset;


                    if (FHT_LIKELY(chunk->get_key_n(match_idx) == k)) {
                        return chunk->get_val_n_ptr(match_idx);
                    }
                }

                const uint32_t available_map =
                    fht_tag_ops::get_available_mask(ctags_v);
                if (FHT_LIKELY(available_map != 0)) {
                    return NULL;
                }
                cur_offset = (cur_offset + sizeof(vec_t)) % ELE_P_NODE;
                if (FHT_UNLIKELY(cur_offset == start_idx)) {
                    break;
                }
            }
            if (FHT_UNLIKELY(attempt == rehash_attempts)) {
                return NULL;
            }
            slot = hash_val_ops::get_slot_n(slot, _cur_mask, ++attempt);
        }
    }

    uint32_t
    FHT_ALIGN(32) erase(key_t k) {

        const hash_t _cur_mask = cur_mask;

        const hash_t raw_hash = hasher(k);
        hash_t       slot     = hash_val_ops::get_slot(raw_hash, _cur_mask);
        FHT_PREFETCH(chunks + slot);

        const uint32_t start_idx = hash_val_ops::get_start_idx(raw_hash);
        const tag_t    tag       = hash_val_ops::get_tag(raw_hash);

        uint32_t attempt = 0;
        for (;;) {
            FHT_ASSERT(start_idx < ELE_P_NODE);
            FHT_ASSERT((tag & tag_t(0x7f)) == tag);

            chunk_t * const chunk = chunks + slot;

            vec_t tag_v = SIMD_helper::set_vec(tag);


            for (uint32_t i = 0; i < ELE_P_NODE; i += sizeof(vec_t)) {
                const uint32_t cur_offset = (i + start_idx) % ELE_P_NODE;


                vec_t ctags_v = chunk->load_tags_o(cur_offset);

                uint32_t matches_map =
                    fht_tag_ops::match_tag_vecs(ctags_v, tag_v);
                for (; matches_map;
                     matches_map = BITS_helper<uint32_t>::df1(matches_map)) {
                    const uint32_t match_idx =
                        BITS_helper<uint32_t>::ff1(matches_map) + cur_offset;

                    FHT_ASSERT(tag == chunk->get_tag_n(match_idx));

                    if (FHT_LIKELY(chunk->get_key_n(match_idx) == k)) {
                        if ((!attempt) && (i == 0) &&
                            (fht_tag_ops::get_available_mask(ctags_v) != 0)) {
                            chunk->set_tag_n(fht_tag_ops::invalid, match_idx);
                        }
                        else {
                            chunk->set_tag_n(fht_tag_ops::erased, match_idx);
                        }
                        return success;
                    }
                }

                const uint32_t available_map =
                    fht_tag_ops::get_available_mask(ctags_v);
                if (FHT_LIKELY(available_map != 0)) {
                    return failure;
                }
            }
            if (attempt == rehash_attempts) {
                break;
            }
            ++attempt;
            slot = hash_val_ops::get_slot_n(slot, _cur_mask, attempt);
        }
        return failure;
    }
    void
    man_count() {
        uint64_t hits = 0;
        for (uint32_t i = 0; i < cur_mask + 1; ++i) {
            for (uint32_t j = 0; j < 64; ++j) {
                if (uint8_t((chunks + i)->get_tag_n(j)) & 0x80) {
                    continue;
                }
                ++hits;
            }
        }
        fprintf(stderr, "%ld / %ld\n", hits, cur_mask);
    }

};  // namespace fht

}  // namespace fht


#endif
