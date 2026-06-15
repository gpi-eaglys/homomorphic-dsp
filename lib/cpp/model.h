// model.h -- shared Model struct and JSON loader (nlohmann/json)
#pragma once
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

struct Model {
    int input_dim, hidden, num_classes, packed_dim;
    std::vector<std::string>         classes;
    std::vector<double>              mean, stdv;
    std::vector<std::vector<double>> W1_diag, W2_diag, W3_diag;
    std::vector<double>              b1, b2, b3;
};

inline Model loadModel(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open model: " + path);
    nlohmann::json j = nlohmann::json::parse(f);

    Model m;
    m.input_dim   = j["input_dim"];
    m.hidden      = j["hidden"];
    m.num_classes = j["num_classes"];
    m.packed_dim  = j["packed_dim"];
    m.classes     = j["classes"].get<std::vector<std::string>>();
    m.mean        = j["mean"].get<std::vector<double>>();
    m.stdv        = j["std"].get<std::vector<double>>();
    m.W1_diag     = j["W1_diag"].get<std::vector<std::vector<double>>>();
    m.b1          = j["b1"].get<std::vector<double>>();
    m.W2_diag     = j["W2_diag"].get<std::vector<std::vector<double>>>();
    m.b2          = j["b2"].get<std::vector<double>>();
    m.W3_diag     = j["W3_diag"].get<std::vector<std::vector<double>>>();
    m.b3          = j["b3"].get<std::vector<double>>();
    return m;
}
