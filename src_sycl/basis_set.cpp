/*
 * GANSU: GPU Acclerated Numerical Simulation Utility
 *
 * Copyright (c) 2025, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * You may obtain a copy of the license in the LICENSE file
 * located in the root directory of this source tree or at:
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */


#include <fstream>
#include <sstream> // std::istringstream
#include <cctype> // std::isalpha
#include <algorithm> // std::replace
#include <utility> // std::pair

#include "basis_set.hpp"
#include "utils.hpp" // THROW_EXCEPTION

namespace gansu{

/**
 * @brief Construct of a basis set from gbs file
 * @param filename Basis set file name (Gaussian basis set file)
 * @return Basis set
 */

BasisSet BasisSet::construct_from_gbs(const std::string& filename){
    std::ifstream ifs(filename);
    if(!ifs){
        THROW_EXCEPTION("Cannot open basis set file: " + filename);
    }


    BasisSet basis_set;
    std::string line;

    ElementBasisSet current_element_basis_set;

    // ---------- 前処理：ヘッダー・コメントをスキップ ----------
    while (std::getline(ifs, line)) {
        // 改行コード対策
        if (!line.empty() && line.back() == '\r') line.pop_back();

        // 空行や空白行をスキップ
        if (line.empty() || std::all_of(line.begin(), line.end(), [](unsigned char c) { return std::isspace(c); }))
            continue;

        // コメントやメタ情報(!や-で始まる行)をスキップ
        char c = static_cast<unsigned char>(line[0]);
        if (!std::isalpha(c)) continue;

        // 最初のアルファベット行が出たら、それを最初の元素行として扱う
        break;
    }

    // ---------- メインループ ----------
    while (true) {
        // EOFチェック
        if (!ifs || line.empty()) break;

        // 前回の要素を登録
        if (!current_element_basis_set.get_element_name().empty()) {
            basis_set.add_element_basis_set(current_element_basis_set);
            current_element_basis_set = ElementBasisSet();
        }

        // ---- 元素名行 (例: "H 0") ----
        {
            // 改行コード対策
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::istringstream iss(line);
            std::string element_name;
            iss >> element_name;
            if (element_name.empty())
                THROW_EXCEPTION("Missing element name in basis file.");

            current_element_basis_set.set_element_name(element_name);
        }

        // ---- 基底関数ブロック ----
        while (std::getline(ifs, line)) {
            // 改行コード対策
            if (!line.empty() && line.back() == '\r') line.pop_back();

            // 空行スキップ
            if (line.empty() || std::all_of(line.begin(), line.end(), [](unsigned char c) { return std::isspace(c); }))
                continue;

            // ブロック終端（次の元素へ）
            if (line == "****") {
                if (!std::getline(ifs, line)) line.clear(); // 次の元素行へ
                break;
            }

            std::istringstream iss(line);
            std::string type;
            size_t num_primitives = 0;
            iss >> type >> num_primitives;

            if (type.empty())
                THROW_EXCEPTION("Missing basis function type for element " + current_element_basis_set.get_element_name());

            // ---- 単独タイプ (S, P, D, F, ...) ----
            if (type.length() == 1) {
                ContractedGauss contracted_gauss(type);

                for (size_t i = 0; i < num_primitives; i++) {
                    if (!std::getline(ifs, line))
                        THROW_EXCEPTION("Unexpected EOF reading " + type + " primitives.");

                    if (!line.empty() && line.back() == '\r') line.pop_back();
                    std::replace(line.begin(), line.end(), 'D', 'E'); // Fortran exponent

                    std::istringstream iss_line(line);
                    double exponent = 0.0, coefficient = 0.0;
                    iss_line >> exponent >> coefficient;
                    contracted_gauss.add_primitive_gauss(exponent, coefficient);
                }

                current_element_basis_set.add_contracted_gauss(contracted_gauss);
            }
            // ---- 混合タイプ (SPなど) ----
            else if (type.length() == 2) {
                ContractedGauss contracted_gauss0(std::string(1, type[0]));
                ContractedGauss contracted_gauss1(std::string(1, type[1]));

                for (size_t i = 0; i < num_primitives; i++) {
                    if (!std::getline(ifs, line))
                        THROW_EXCEPTION("Unexpected EOF reading " + type + " primitives.");

                    if (!line.empty() && line.back() == '\r') line.pop_back();
                    std::replace(line.begin(), line.end(), 'D', 'E');

                    std::istringstream iss_line(line);
                    double exponent = 0.0, coeff0 = 0.0, coeff1 = 0.0;
                    iss_line >> exponent >> coeff0 >> coeff1;

                    contracted_gauss0.add_primitive_gauss(exponent, coeff0);
                    contracted_gauss1.add_primitive_gauss(exponent, coeff1);
                }

                current_element_basis_set.add_contracted_gauss(contracted_gauss0);
                current_element_basis_set.add_contracted_gauss(contracted_gauss1);
            }
            else {
                THROW_EXCEPTION("Invalid basis function name: " + type);
            }
        }

        // EOFまで来たら終了
        if (!ifs) break;
    }

    // 最後の要素を登録
    if (!current_element_basis_set.get_element_name().empty()) {
        basis_set.add_element_basis_set(current_element_basis_set);
    }

    return basis_set;
    }


/*

    BasisSet basis_set;
    std::string line;

    ElementBasisSet current_element_basis_set;

    // Read lines until the first charactor of the line is an alphabet.
    while(std::getline(ifs, line)){
        if(std::isalpha(line[0])){
            // unread the line
            ifs.seekg(ifs.tellg() - static_cast<std::streamoff>(line.size() + 1));
            break;
        }
    }

    while(!ifs.eof()){
        if(!current_element_basis_set.get_element_name().empty()){
            basis_set.add_element_basis_set(current_element_basis_set);
            current_element_basis_set = ElementBasisSet();
        }

        
        { // Read a line for Element name
            std::getline(ifs, line);
            std::istringstream iss(line);
            // Get element name (H, He, Li, ...)
            std::string element_name;
            iss >> element_name;
            current_element_basis_set.set_element_name(element_name);
        }

        // Read lines for basis functions
        while(std::getline(ifs, line)){
            // If the line is "****", the end of the basis functions
            if(line == "****"){
                break;
            }

            std::istringstream iss(line);

            // Get the type of the basis functions and the number of primitive Gaussians
            std::string type;
            size_t num_primitives;
            iss >> type >> num_primitives;


            if(type.length() == 1){ // S, P, D, F, ...
                ContractedGauss contracted_gauss(type);
                for(size_t i = 0; i < num_primitives; i++){
                    std::getline(ifs, line);
                    // Replace all "D"s to "E"s for the exponential notation
                    std::replace(line.begin(), line.end(), 'D', 'E');

                    std::istringstream iss(line);
                    double exponent, coefficient;
                    iss >> exponent >> coefficient;
                    contracted_gauss.add_primitive_gauss(exponent, coefficient);
                }
                current_element_basis_set.add_contracted_gauss(contracted_gauss);
            }else if(type.length() == 2){ // SP, ??, ...
                ContractedGauss contracted_gauss0(std::string(1,type[0]));
                ContractedGauss contracted_gauss1(std::string(1,type[1]));
                for(size_t i = 0; i < num_primitives; i++){
                    std::getline(ifs, line);
                    // Replace all "D"s to "E"s for the exponential notation
                    std::replace(line.begin(), line.end(), 'D', 'E');

                    std::istringstream iss(line);
                    double exponent, coefficient0, coefficient1;
                    iss >> exponent >> coefficient0 >> coefficient1;
                    contracted_gauss0.add_primitive_gauss(exponent, coefficient0);
                    contracted_gauss1.add_primitive_gauss(exponent, coefficient1);
                }
                current_element_basis_set.add_contracted_gauss(contracted_gauss0);
                current_element_basis_set.add_contracted_gauss(contracted_gauss1);
            }else{ // could not find, or three or more characters
                THROW_EXCEPTION("Invalid basis function name: " + type);
            }
        }
    }

    // The last element basis set is added
    if(!current_element_basis_set.get_element_name().empty()){
        basis_set.add_element_basis_set(current_element_basis_set);
    }

    return basis_set;
}
*/

}
