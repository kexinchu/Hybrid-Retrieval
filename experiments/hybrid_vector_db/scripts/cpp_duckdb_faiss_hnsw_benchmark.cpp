#include "duckdb.h"

#include <faiss/Index.h>
#include <faiss/impl/HNSW.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/index_io.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Filter {
    const char* name;
    const char* label;
    const char* predicate;
};

struct FBinMMap {
    int fd = -1;
    size_t bytes = 0;
    void* mapping = nullptr;
    int32_t n = 0;
    int32_t d = 0;
    const float* data = nullptr;

    explicit FBinMMap(const std::string& path) {
        fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("failed to open fbin: " + path);
        }
        struct stat st {};
        if (fstat(fd, &st) != 0) {
            throw std::runtime_error("failed to stat fbin: " + path);
        }
        bytes = static_cast<size_t>(st.st_size);
        mapping = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapping == MAP_FAILED) {
            throw std::runtime_error("failed to mmap fbin: " + path);
        }
        auto* header = static_cast<const int32_t*>(mapping);
        n = header[0];
        d = header[1];
        data = reinterpret_cast<const float*>(static_cast<const char*>(mapping) + 8);
    }

    ~FBinMMap() {
        if (mapping && mapping != MAP_FAILED) {
            munmap(mapping, bytes);
        }
        if (fd >= 0) {
            close(fd);
        }
    }

    const float* row(int64_t id) const {
        return data + id * d;
    }
};

struct TruthRow {
    int query_id = -1;
    std::vector<int64_t> truth_ids;
};

struct Metrics {
    double recall_sum = 0.0;
    double latency_sum = 0.0;
    double vector_sum = 0.0;
    double rerank_sum = 0.0;
    double returned_sum = 0.0;
    double intersection_sum = 0.0;
    int count = 0;
};

const std::vector<Filter> filters = {
    {"popular_ge1000", "50.32%", "item_rating_number >= 1000"},
    {"price_10_to_20", "21.89%", "has_price AND price > 10 AND price <= 20"},
    {"rating5_price_le10", "9.59%", "has_price AND price <= 10 AND rating = 5"},
    {"long_review_ge500", "5.88%", "review_text_len >= 500"},
    {"grocery_rating5", "2.34%", "main_category = 'Grocery' AND rating = 5"},
    {"grocery_helpful", "1.01%", "main_category = 'Grocery' AND helpful_vote >= 1"},
    {"helpful_ge20", "0.61%", "helpful_vote >= 20"},
    {"grocery_long500", "0.21%", "main_category = 'Grocery' AND review_text_len >= 500"},
};

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::vector<std::string_view> split_csv_line(std::string& line) {
    std::vector<std::string_view> fields;
    fields.reserve(20);
    bool in_quotes = false;
    size_t start = 0;
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '"') {
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                ++i;
            } else {
                in_quotes = !in_quotes;
            }
        } else if (c == ',' && !in_quotes) {
            fields.emplace_back(line.data() + start, i - start);
            start = i + 1;
        }
    }
    fields.emplace_back(line.data() + start, line.size() - start);
    return fields;
}

std::string unquote(std::string_view v) {
    if (v.size() >= 2 && v.front() == '"' && v.back() == '"') {
        v = v.substr(1, v.size() - 2);
    }
    return std::string(v);
}

std::vector<int64_t> parse_id_list(std::string_view value) {
    std::string s = unquote(value);
    std::vector<int64_t> ids;
    size_t start = 0;
    while (start < s.size()) {
        size_t comma = s.find(',', start);
        std::string_view part(s.data() + start, (comma == std::string::npos ? s.size() : comma) - start);
        if (!part.empty()) {
            ids.push_back(std::atoll(std::string(part).c_str()));
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return ids;
}

std::unordered_map<std::string, std::vector<TruthRow>> load_truth(const std::string& path, int queries) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open truth csv: " + path);
    }
    std::unordered_map<std::string, std::vector<TruthRow>> out;
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
        auto fields = split_csv_line(line);
        if (fields.size() < 17) {
            continue;
        }
        std::string method(fields[6]);
        if (method != "pre_filter_exact") {
            continue;
        }
        std::string filter_name(fields[2]);
        auto& rows = out[filter_name];
        if (static_cast<int>(rows.size()) >= queries) {
            continue;
        }
        TruthRow row;
        row.query_id = std::atoi(std::string(fields[1]).c_str());
        row.truth_ids = parse_id_list(fields[16]);
        rows.push_back(std::move(row));
    }
    return out;
}

void run_query(duckdb_connection conn, const std::string& query) {
    duckdb_result result;
    if (duckdb_query(conn, query.c_str(), &result) == DuckDBError) {
        std::string err = duckdb_result_error(&result);
        duckdb_destroy_result(&result);
        throw std::runtime_error(err);
    }
    duckdb_destroy_result(&result);
}

std::vector<uint8_t> pack_bitstring_to_faiss_bitmap(const char* bits, size_t bit_count) {
    std::vector<uint8_t> bitmap((bit_count + 7) / 8, 0);
    const size_t full_bytes = bit_count / 8;
    for (size_t byte = 0; byte < full_bytes; ++byte) {
        const size_t i = byte * 8;
        bitmap[byte] =
            static_cast<uint8_t>((bits[i + 0] == '1') << 0) |
            static_cast<uint8_t>((bits[i + 1] == '1') << 1) |
            static_cast<uint8_t>((bits[i + 2] == '1') << 2) |
            static_cast<uint8_t>((bits[i + 3] == '1') << 3) |
            static_cast<uint8_t>((bits[i + 4] == '1') << 4) |
            static_cast<uint8_t>((bits[i + 5] == '1') << 5) |
            static_cast<uint8_t>((bits[i + 6] == '1') << 6) |
            static_cast<uint8_t>((bits[i + 7] == '1') << 7);
    }
    for (size_t i = full_bytes * 8; i < bit_count; ++i) {
        bitmap[i >> 3] |= static_cast<uint8_t>((bits[i] == '1') << (i & 7));
    }
    return bitmap;
}

bool bitmap_contains(const std::vector<uint8_t>& bitmap, int64_t id) {
    return bool(bitmap[id >> 3] & (1 << (id & 7)));
}

float l2_distance(const float* query, const float* vec, int d) {
    float query_norm = 0.0f;
    float vec_norm = 0.0f;
    float dot = 0.0f;
    for (int i = 0; i < d; ++i) {
        query_norm += query[i] * query[i];
        vec_norm += vec[i] * vec[i];
        dot += vec[i] * query[i];
    }
    return vec_norm + query_norm - 2.0f * dot;
}

std::vector<int64_t> exact_rerank(const FBinMMap& xb, const float* query, const std::vector<int64_t>& candidates, int k) {
    std::vector<std::pair<float, int64_t>> scored;
    scored.reserve(candidates.size());
    for (int64_t id : candidates) {
        scored.emplace_back(l2_distance(query, xb.row(id), xb.d), id);
    }
    const size_t take = std::min<size_t>(k, scored.size());
    if (take < scored.size()) {
        std::nth_element(scored.begin(), scored.begin() + take, scored.end());
        scored.resize(take);
    }
    std::sort(scored.begin(), scored.end());
    std::vector<int64_t> out;
    out.reserve(take);
    for (const auto& item : scored) {
        out.push_back(item.second);
    }
    return out;
}

double recall_at_k(const std::vector<int64_t>& ids, const std::vector<int64_t>& truth, int k) {
    if (truth.empty()) {
        return 0.0;
    }
    int hit = 0;
    const int limit = std::min<int>(k, truth.size());
    for (int i = 0; i < std::min<int>(k, ids.size()); ++i) {
        for (int j = 0; j < limit; ++j) {
            if (ids[i] == truth[j]) {
                ++hit;
                break;
            }
        }
    }
    return static_cast<double>(hit) / static_cast<double>(limit);
}

std::vector<int64_t> faiss_search(faiss::Index* index, const float* query, int k, int ef_search, faiss::IDSelector* selector, double& ms) {
    std::vector<float> distances(k, 0.0f);
    std::vector<faiss::idx_t> labels(k, -1);
    faiss::SearchParametersHNSW params;
    params.efSearch = ef_search;
    params.sel = selector;
    const auto start = Clock::now();
    index->search(1, query, k, distances.data(), labels.data(), &params);
    const auto end = Clock::now();
    ms = elapsed_ms(start, end);
    std::vector<int64_t> out;
    out.reserve(k);
    for (auto label : labels) {
        if (label >= 0) {
            out.push_back(static_cast<int64_t>(label));
        }
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string db_path = argc > 1 ? argv[1] : "data/duckdb/amazon_grocery_10m.duckdb";
    const std::string table = argc > 2 ? argv[2] : "amazon_grocery_reviews_10m";
    const std::string fbin_path = argc > 3 ? argv[3] : "data/amazon_reviews_2023/processed/grocery_reviews_10m_tfidf_svd128.fbin";
    const std::string index_path = argc > 4 ? argv[4] : "data/faiss/amazon_grocery_10m_tfidf_svd128_hnsw_m16.index";
    const std::string truth_path = argc > 5 ? argv[5] : "results/hybrid_vector_db/faiss_hnsw_sql_attribute_filter_10m_q100_20260602.csv";
    const int total_rows = argc > 6 ? std::atoi(argv[6]) : 10000000;
    const int queries = argc > 7 ? std::atoi(argv[7]) : 100;
    const int vector_topn = argc > 8 ? std::atoi(argv[8]) : 50000;
    const int ef_search = argc > 9 ? std::atoi(argv[9]) : 1000;
    const int k = argc > 10 ? std::atoi(argv[10]) : 10;
    const int threads = argc > 11 ? std::atoi(argv[11]) : 8;

    duckdb_database db;
    duckdb_connection conn;
    if (duckdb_open(db_path.c_str(), &db) == DuckDBError || duckdb_connect(db, &conn) == DuckDBError) {
        std::cerr << "failed to open/connect DuckDB\n";
        return 1;
    }
    run_query(conn, "PRAGMA threads=" + std::to_string(threads));

    FBinMMap xb(fbin_path);
    faiss::Index* index = faiss::read_index(index_path.c_str());
    auto truth = load_truth(truth_path, queries);

    std::cout << "cpp_duckdb_faiss_hnsw queries=" << queries
              << " vector_topn=" << vector_topn
              << " efSearch=" << ef_search
              << " k=" << k
              << " threads=" << threads << "\n";
    std::cout << "| Filter | Parallel recall | Parallel latency ms | Parallel SQL ms | Parallel vector ms | Pre recall | Pre latency ms | Pre SQL ms | Pre build ms | Pre search ms |\n";
    std::cout << "| ------ | --------------- | ------------------- | --------------- | ------------------ | ---------- | -------------- | ---------- | ------------ | ------------- |\n";

    for (const auto& filter : filters) {
        const std::string query =
            "SELECT count(*)::BIGINT, CAST(bitstring_agg(id, 0, " + std::to_string(total_rows - 1) + ") AS VARCHAR) "
            "FROM " + table + " WHERE " + filter.predicate;

        duckdb_result result;
        const auto sql_start = Clock::now();
        if (duckdb_query(conn, query.c_str(), &result) == DuckDBError) {
            std::cerr << "query error: " << duckdb_result_error(&result) << "\n";
            return 1;
        }
        int64_t sql_rows = duckdb_value_int64(&result, 0, 0);
        char* bits = duckdb_value_varchar(&result, 1, 0);
        const auto sql_end = Clock::now();

        const auto build_start = Clock::now();
        std::vector<uint8_t> bitmap = pack_bitstring_to_faiss_bitmap(bits, total_rows);
        faiss::IDSelectorBitmap selector(bitmap.size(), bitmap.data());
        const auto build_end = Clock::now();
        duckdb_free(bits);
        duckdb_destroy_result(&result);

        const double sql_ms = elapsed_ms(sql_start, sql_end);
        const double build_ms = elapsed_ms(build_start, build_end);

        Metrics parallel;
        Metrics pre;
        const auto& rows = truth[filter.name];
        for (const auto& row : rows) {
            const float* q = xb.row(row.query_id);

            double vec_ms = 0.0;
            std::vector<int64_t> vec_ids = faiss_search(index, q, vector_topn, ef_search, nullptr, vec_ms);

            std::vector<int64_t> intersection;
            intersection.reserve(vec_ids.size());
            for (int64_t id : vec_ids) {
                if (bitmap_contains(bitmap, id)) {
                    intersection.push_back(id);
                }
            }

            const auto rerank_start = Clock::now();
            std::vector<int64_t> parallel_ids = exact_rerank(xb, q, intersection, k);
            const auto rerank_end = Clock::now();
            const double rerank_ms = elapsed_ms(rerank_start, rerank_end);

            double pre_ms = 0.0;
            std::vector<int64_t> pre_ids = faiss_search(index, q, k, ef_search, &selector, pre_ms);

            parallel.recall_sum += recall_at_k(parallel_ids, row.truth_ids, k);
            parallel.latency_sum += std::max(sql_ms, vec_ms) + rerank_ms;
            parallel.vector_sum += vec_ms;
            parallel.rerank_sum += rerank_ms;
            parallel.returned_sum += parallel_ids.size();
            parallel.intersection_sum += intersection.size();
            parallel.count += 1;

            pre.recall_sum += recall_at_k(pre_ids, row.truth_ids, k);
            pre.latency_sum += sql_ms + build_ms + pre_ms;
            pre.vector_sum += pre_ms;
            pre.returned_sum += pre_ids.size();
            pre.intersection_sum += pre_ids.size();
            pre.count += 1;
        }

        std::cout << "| " << filter.label
                  << " | " << std::fixed << std::setprecision(3) << (parallel.recall_sum / parallel.count)
                  << " | " << std::setprecision(2) << (parallel.latency_sum / parallel.count)
                  << " | " << sql_ms
                  << " | " << (parallel.vector_sum / parallel.count)
                  << " | " << std::setprecision(3) << (pre.recall_sum / pre.count)
                  << " | " << std::setprecision(2) << (pre.latency_sum / pre.count)
                  << " | " << sql_ms
                  << " | " << build_ms
                  << " | " << (pre.vector_sum / pre.count)
                  << " |\n";
    }

    delete index;
    duckdb_disconnect(&conn);
    duckdb_close(&db);
    return 0;
}
