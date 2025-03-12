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


/**
 * @file builder.hpp 
 */

#pragma once

#include "hf.hpp"
#include "rhf.hpp"
#include "uhf.hpp"
#include "rohf.hpp"
#include "utils.hpp" // THROW_EXCEPTION

#include "parameter_manager.hpp"
#include "gpu_manager.hpp"

namespace gansu{

/**
 * @brief HFBuilder class
 * @details This class is a virtual class for building the HF class.
 */
class HFBuilder {
public:
    /**
     * @brief Build of the HF class (Factory method)
     */
    static std::unique_ptr<HF> buildHF(const ParameterManager& arg_parameters){

        ParameterManager parameters; // Create a new ParameterManager without setting default values

        // Load the parameters if the parameter file is given
        if(arg_parameters.contains("parameter_file")){
            const std::string parameter_file = arg_parameters.get<std::string>(std::string("parameter_file"));
            parameters.load_from_file(parameter_file);
        }

        // add the command line arguments, which will overwrite the parameters from the file if the same parameter is specified.
        for(const auto& key : arg_parameters.keys()){
            parameters[key] = arg_parameters.get<std::string>(key);
        }


        if(parameters.contains("verbose")){
            std::cout << "Enumerating all parameters:" << std::endl;
            std::vector<std::string> key_list = parameters.keys();
            for (const auto& key : key_list) {
                std::cout << "parameters[" << key << "] = " << parameters.get<std::string>(key) << std::endl;
            }
        }

        // Check if required parameters are set
        if(!parameters.contains("xyzfilename")){
            THROW_EXCEPTION("xyzfilename is not set");
        }
        if(!parameters.contains("gbsfilename")){
            THROW_EXCEPTION("gbsfilename is not set");
        }

        // Set default values for unspecified parameters
        parameters.set_default_values_to_unspecfied_parameters();


        // Check the emvironment
        {
            gpu::cusolverManager cusolver;
            gpu::cublasManager cublas;
        }


        // Read the molecular information and create the Molecular class
        const std::string xyzfilename = parameters.get<std::string>("xyzfilename");
        const std::string gbsfilename = parameters.get<std::string>("gbsfilename");
        const int charge = parameters.get<int>("charge");
        const int beta_to_alpha = parameters.get<int>("beta_to_alpha");
        
        Molecular molecular(xyzfilename, gbsfilename, charge, beta_to_alpha);

        // Select and build the HF class
        const std::string method = parameters.get<std::string>("method");
        if(method == "rhf"){
            return std::make_unique<RHF>(molecular, parameters);
        }else if(method == "uhf"){
            return std::make_unique<UHF>(molecular, parameters);
        }else if(method == "rohf"){
            return std::make_unique<ROHF>(molecular, parameters);
        }else{
            THROW_EXCEPTION("Invalid method: " + method);
        }
    }

};

} // namespace gansu