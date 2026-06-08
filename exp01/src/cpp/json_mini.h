// json_mini.h -- minimal reader for the flat model.json produced by train.py.
// Not a general JSON parser; it knows exactly the keys we export.
#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>

// forward-declared in fhe_server.cpp
struct Model;

namespace jm {

inline std::string slurp(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

// find `"key"` and return the position just after the following ':'
inline size_t findKey(const std::string& s, const std::string& key, size_t from = 0) {
    std::string pat = "\"" + key + "\"";
    size_t p = s.find(pat, from);
    if (p == std::string::npos) throw std::runtime_error("key not found: " + key);
    p = s.find(':', p + pat.size());
    return p + 1;
}

inline double readNumber(const std::string& s, size_t pos) {
    return std::stod(s.substr(pos));
}

// read a flat [a, b, c, ...] array starting at/after pos
inline std::vector<double> readArray(const std::string& s, size_t pos) {
    size_t l = s.find('[', pos);
    size_t r = s.find(']', l);
    std::vector<double> out;
    std::string body = s.substr(l + 1, r - l - 1);
    std::stringstream ss(body);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        // strip whitespace
        size_t a = tok.find_first_not_of(" \n\t\r");
        if (a == std::string::npos) continue;
        out.push_back(std::stod(tok.substr(a)));
    }
    return out;
}

// read a 2-D array [[...],[...],...] starting at/after pos
inline std::vector<std::vector<double>> readArray2D(const std::string& s, size_t pos) {
    size_t l = s.find('[', pos);          // outer open
    std::vector<std::vector<double>> out;
    size_t i = l + 1;
    while (true) {
        size_t open = s.find('[', i);
        size_t outerClose = s.find(']', i);
        if (open == std::string::npos || open > outerClose) break;  // done
        size_t close = s.find(']', open);
        out.push_back(readArray(s, open));
        i = close + 1;
    }
    return out;
}

inline std::vector<std::string> readStringArray(const std::string& s, size_t pos) {
    size_t l = s.find('[', pos);
    size_t r = s.find(']', l);
    std::vector<std::string> out;
    std::string body = s.substr(l + 1, r - l - 1);
    size_t i = 0;
    while (true) {
        size_t q1 = body.find('"', i);
        if (q1 == std::string::npos) break;
        size_t q2 = body.find('"', q1 + 1);
        out.push_back(body.substr(q1 + 1, q2 - q1 - 1));
        i = q2 + 1;
    }
    return out;
}

} // namespace jm

// Definition matching the struct in fhe_server.cpp
struct Model {
    int input_dim, hidden_dim, num_classes, packed_dim;
    std::vector<std::string> classes;
    std::vector<double> mean, stdv, b1, b2;
    std::vector<std::vector<double>> W1_diag, W2_diag;
};

inline Model loadModel(const std::string& path) {
    std::string s = jm::slurp(path);
    Model m;
    m.input_dim   = (int)jm::readNumber(s, jm::findKey(s, "input_dim"));
    m.hidden_dim  = (int)jm::readNumber(s, jm::findKey(s, "hidden_dim"));
    m.num_classes = (int)jm::readNumber(s, jm::findKey(s, "num_classes"));
    m.packed_dim  = (int)jm::readNumber(s, jm::findKey(s, "packed_dim"));
    m.classes = jm::readStringArray(s, jm::findKey(s, "classes"));
    m.mean    = jm::readArray(s, jm::findKey(s, "mean"));
    m.stdv    = jm::readArray(s, jm::findKey(s, "std"));
    m.b1      = jm::readArray(s, jm::findKey(s, "b1"));
    m.b2      = jm::readArray(s, jm::findKey(s, "b2"));
    m.W1_diag = jm::readArray2D(s, jm::findKey(s, "W1_diag"));
    m.W2_diag = jm::readArray2D(s, jm::findKey(s, "W2_diag"));
    return m;
}
