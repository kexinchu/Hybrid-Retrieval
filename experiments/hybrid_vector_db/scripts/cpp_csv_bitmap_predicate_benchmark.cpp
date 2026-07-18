#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Row {
    int64_t id = -1;
    double rating = 0.0;
    int helpful_vote = 0;
    int review_text_len = 0;
    std::string_view main_category;
    double price = 0.0;
    bool has_price = false;
    int item_rating_number = 0;
};

struct Filter {
    const char *name;
    const char *target;
    bool (*match)(const Row &row);
};

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::vector<std::string_view> split_csv_line(std::string &line) {
    std::vector<std::string_view> fields;
    fields.reserve(16);
    bool in_quotes = false;
    size_t start = 0;
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
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    fields.emplace_back(line.data() + start, line.size() - start);
    return fields;
}

bool parse_bool(std::string_view value) {
    return value == "True" || value == "true" || value == "1";
}

int parse_int(std::string_view value) {
    return value.empty() ? 0 : std::atoi(std::string(value).c_str());
}

int64_t parse_i64(std::string_view value) {
    return value.empty() ? 0 : std::atoll(std::string(value).c_str());
}

double parse_double(std::string_view value) {
    return value.empty() ? 0.0 : std::atof(std::string(value).c_str());
}

Row parse_row(const std::vector<std::string_view> &f) {
    Row row;
    row.id = parse_i64(f[0]);
    row.rating = parse_double(f[3]);
    row.helpful_vote = parse_int(f[6]);
    row.review_text_len = parse_int(f[7]);
    row.main_category = f[9];
    row.price = parse_double(f[11]);
    row.has_price = parse_bool(f[12]);
    row.item_rating_number = parse_int(f[14]);
    return row;
}

void set_bit(std::vector<uint8_t> &bitmap, int64_t id) {
    bitmap[static_cast<size_t>(id) >> 3] |= static_cast<uint8_t>(1u << (id & 7));
}

bool popular_ge1000(const Row &r) { return r.item_rating_number >= 1000; }
bool price_10_to_20(const Row &r) { return r.has_price && r.price > 10.0 && r.price <= 20.0; }
bool rating5_price_le10(const Row &r) { return r.has_price && r.price <= 10.0 && r.rating == 5.0; }
bool long_review_ge500(const Row &r) { return r.review_text_len >= 500; }
bool grocery_rating5(const Row &r) { return r.main_category == "Grocery" && r.rating == 5.0; }
bool grocery_helpful(const Row &r) { return r.main_category == "Grocery" && r.helpful_vote >= 1; }
bool helpful_ge20(const Row &r) { return r.helpful_vote >= 20; }
bool grocery_long500(const Row &r) { return r.main_category == "Grocery" && r.review_text_len >= 500; }

const std::vector<Filter> filters = {
    {"popular_ge1000", "50%", popular_ge1000},
    {"price_10_to_20", "20%", price_10_to_20},
    {"rating5_price_le10", "10%", rating5_price_le10},
    {"long_review_ge500", "5%", long_review_ge500},
    {"grocery_rating5", "2%", grocery_rating5},
    {"grocery_helpful", "1%", grocery_helpful},
    {"helpful_ge20", "0.5%", helpful_ge20},
    {"grocery_long500", "0.2%", grocery_long500},
};

}  // namespace

int main(int argc, char **argv) {
    const std::string csv_path = argc > 1 ? argv[1] : "data/amazon_reviews_2023/processed/grocery_reviews_10m_hybrid_sql.csv";
    const int64_t total_rows = argc > 2 ? std::atoll(argv[2]) : 10000000LL;

    std::cout << "csv=" << csv_path << " total_rows=" << total_rows << "\n";
    std::cout << "filter,target,rows,selectivity,bitmap_bytes,elapsed_ms\n";

    for (const auto &filter : filters) {
        std::ifstream in(csv_path);
        if (!in) {
            std::cerr << "failed to open " << csv_path << "\n";
            return 1;
        }

        std::vector<uint8_t> bitmap(static_cast<size_t>((total_rows + 7) / 8), 0);
        std::string line;
        std::getline(in, line);

        int64_t matched = 0;
        int64_t parsed = 0;
        const auto start = Clock::now();
        while (std::getline(in, line)) {
            auto fields = split_csv_line(line);
            if (fields.size() < 15) {
                continue;
            }
            Row row = parse_row(fields);
            if (filter.match(row)) {
                set_bit(bitmap, row.id);
                ++matched;
            }
            ++parsed;
        }
        const auto end = Clock::now();

        std::cout << filter.name << ","
                  << filter.target << ","
                  << matched << ","
                  << std::fixed << std::setprecision(6) << (static_cast<double>(matched) / static_cast<double>(parsed)) << ","
                  << bitmap.size() << ","
                  << std::setprecision(2) << elapsed_ms(start, end) << "\n";
    }

    std::ifstream in(csv_path);
    if (!in) {
        std::cerr << "failed to open " << csv_path << "\n";
        return 1;
    }
    std::vector<std::vector<uint8_t>> bitmaps;
    std::vector<int64_t> matches(filters.size(), 0);
    bitmaps.reserve(filters.size());
    for (size_t i = 0; i < filters.size(); ++i) {
        bitmaps.emplace_back(static_cast<size_t>((total_rows + 7) / 8), 0);
    }

    std::string line;
    std::getline(in, line);
    int64_t parsed = 0;
    const auto start = Clock::now();
    while (std::getline(in, line)) {
        auto fields = split_csv_line(line);
        if (fields.size() < 15) {
            continue;
        }
        Row row = parse_row(fields);
        for (size_t i = 0; i < filters.size(); ++i) {
            if (filters[i].match(row)) {
                set_bit(bitmaps[i], row.id);
                ++matches[i];
            }
        }
        ++parsed;
    }
    const auto end = Clock::now();
    std::cout << "single_pass_all_filters,total_rows=" << parsed
              << ",elapsed_ms=" << std::fixed << std::setprecision(2)
              << elapsed_ms(start, end) << "\n";
    for (size_t i = 0; i < filters.size(); ++i) {
        std::cout << "single_pass," << filters[i].name << ","
                  << matches[i] << ","
                  << std::fixed << std::setprecision(6)
                  << (static_cast<double>(matches[i]) / static_cast<double>(parsed)) << "\n";
    }

    return 0;
}
