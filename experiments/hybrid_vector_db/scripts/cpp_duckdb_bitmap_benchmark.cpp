#include "duckdb.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Filter {
    const char *name;
    const char *label;
    const char *predicate;
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

void run_query(duckdb_connection conn, const std::string &query) {
    duckdb_result result;
    if (duckdb_query(conn, query.c_str(), &result) == DuckDBError) {
        std::string err = duckdb_result_error(&result);
        duckdb_destroy_result(&result);
        throw std::runtime_error(err);
    }
    duckdb_destroy_result(&result);
}

std::vector<uint8_t> pack_bitstring_to_faiss_bitmap(const char *bits, size_t bit_count) {
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

}  // namespace

int main(int argc, char **argv) {
    const std::string db_path = argc > 1 ? argv[1] : "data/duckdb/amazon_grocery_10m.duckdb";
    const std::string table = argc > 2 ? argv[2] : "amazon_grocery_reviews_10m";
    const int64_t total_rows = argc > 3 ? std::atoll(argv[3]) : 10000000LL;
    const int threads = argc > 4 ? std::atoi(argv[4]) : 8;

    duckdb_database db;
    duckdb_connection conn;

    if (duckdb_open(db_path.c_str(), &db) == DuckDBError) {
        std::cerr << "failed to open DuckDB database: " << db_path << "\n";
        return 1;
    }
    if (duckdb_connect(db, &conn) == DuckDBError) {
        std::cerr << "failed to connect DuckDB database\n";
        duckdb_close(&db);
        return 1;
    }

    try {
        run_query(conn, "PRAGMA threads=" + std::to_string(threads));
        run_query(conn, "SELECT count(*) FROM " + table);
    } catch (const std::exception &ex) {
        std::cerr << "setup error: " << ex.what() << "\n";
        duckdb_disconnect(&conn);
        duckdb_close(&db);
        return 1;
    }

    std::cout << "duckdb_c_api_db=" << db_path << " table=" << table
              << " total_rows=" << total_rows << " threads=" << threads << "\n";
    std::cout << "filter,label,rows,selectivity,sql_bitstring_ms,bitmap_build_ms,total_ms,bitmap_bytes\n";

    for (const auto &filter : filters) {
        const std::string query =
            "SELECT count(*)::BIGINT, CAST(bitstring_agg(id, 0, " + std::to_string(total_rows - 1) + ") AS VARCHAR) "
            "FROM " + table + " WHERE " + filter.predicate;

        duckdb_result result;
        const auto sql_start = Clock::now();
        if (duckdb_query(conn, query.c_str(), &result) == DuckDBError) {
            std::cerr << "query error for " << filter.name << ": " << duckdb_result_error(&result) << "\n";
            duckdb_destroy_result(&result);
            duckdb_disconnect(&conn);
            duckdb_close(&db);
            return 1;
        }
        int64_t rows = duckdb_value_int64(&result, 0, 0);
        char *bits = duckdb_value_varchar(&result, 1, 0);
        const auto sql_end = Clock::now();

        const auto build_start = Clock::now();
        std::vector<uint8_t> bitmap = pack_bitstring_to_faiss_bitmap(bits, static_cast<size_t>(total_rows));
        const auto build_end = Clock::now();

        duckdb_free(bits);
        duckdb_destroy_result(&result);

        const double sql_ms = elapsed_ms(sql_start, sql_end);
        const double build_ms = elapsed_ms(build_start, build_end);
        std::cout << filter.name << ","
                  << filter.label << ","
                  << rows << ","
                  << std::fixed << std::setprecision(6)
                  << static_cast<double>(rows) / static_cast<double>(total_rows) << ","
                  << std::setprecision(2)
                  << sql_ms << ","
                  << build_ms << ","
                  << (sql_ms + build_ms) << ","
                  << bitmap.size() << "\n";
    }

    duckdb_disconnect(&conn);
    duckdb_close(&db);
    return 0;
}
