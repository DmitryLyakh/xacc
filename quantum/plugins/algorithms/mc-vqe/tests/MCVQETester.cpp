/*******************************************************************************
 * Copyright (c) 2019 UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * and Eclipse Distribution License v1.0 which accompanies this
 * distribution. The Eclipse Public License is available at
 * http://www.eclipse.org/legal/epl-v10.html and the Eclipse Distribution
 *License is available at https://eclipse.org/org/documents/edl-v10.php
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *******************************************************************************/
#include <gtest/gtest.h>

#include "xacc.hpp"
#include "xacc_service.hpp"
#include "Optimizer.hpp"
#include "Algorithm.hpp"
#include <random>
#include <fstream>
#include <sstream>

using namespace xacc;

/*
TEST(mcVqeTester, check4qubitExaTN) {

    std::string path = "/home/cades/dev/xacc/quantum/plugins/algorithms/mc-vqe/tests/datafile.txt";
    auto acc = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor","exatn")});
    auto buffer = xacc::qalloc(4);
    auto optimizer = xacc::getOptimizer("nlopt", {std::make_pair("nlopt-maxeval", 1)});
    auto mc_vqe = xacc::getService<Algorithm>("mc-vqe");
    EXPECT_TRUE(mc_vqe->initialize({std::make_pair("accelerator",acc),
                                std::make_pair("optimizer",optimizer),
                                std::make_pair("data-path", path),
                                std::make_pair("nChromophores", 4)}));
    mc_vqe->execute(buffer);

}

TEST(mcVqeTester, check4qubitITensorMPS) {

    std::string path = "/home/cades/dev/xacc/quantum/plugins/algorithms/mc-vqe/tests/datafile.txt";
    auto acc = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor","itensor-mps")});
    auto buffer = xacc::qalloc(4);
    auto optimizer = xacc::getOptimizer("nlopt", {std::make_pair("nlopt-maxeval", 1)});
    auto mc_vqe = xacc::getService<Algorithm>("mc-vqe");
    EXPECT_TRUE(mc_vqe->initialize({std::make_pair("accelerator",acc),
                                std::make_pair("optimizer",optimizer),
                                std::make_pair("data-path", path),
                                std::make_pair("nChromophores", 4)}));
    mc_vqe->execute(buffer);

}
*/

TEST(mcVqeTester, checkHPCVirt) {

  int n_virt_qpus = 2, exatnLogLevel = 2, mcvqeLogLevel = 1, n_chromophores = 4, exatnBufferSize = 2, opt_maxiter = 1, n_states = 1, n_cycles = 1;
  std::string acc = "tnqvm";

  //xacc::logToFile(true);
  xacc::setLoggingLevel(exatnLogLevel);

  // pseudo random number generator
  auto rand = [](){

    std::random_device rnd_device;
    std::mt19937 mersenne_engine{rnd_device()}; // Generates random integers
    double l_range = -1.0, r_range = 0.0;
    std::uniform_real_distribution<double> dist{l_range, r_range};
    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
    std::vector<double> vec(1);
    std::generate(vec.begin(), vec.end(), gen);
    return vec[0];

  };

    std::ofstream datafile("datafile.txt", std::ofstream::out);
    std::stringstream ss;
    for (int i = 0; i < n_chromophores; i++) {
       ss << i << "\n";
       ss << "Ground state energy: " << rand() << "\n";
       ss << "Excited state energy: " << rand() << "\n";
       ss << "Center of mass: " << rand() << "," << rand() << "," << rand() << "\n";
       ss << "Ground state dipole moment: " << rand() << "," << rand() << "," << rand() << "\n";
       ss << "Excited state dipole moment: " << rand() << "," << rand() << "," << rand() << "\n";
       ss << "Transition dipole moment: " << rand() << "," << rand() << "," << rand() << "\n";
       ss << "\n";
    }  

  datafile << ss.str();
  datafile.close();
  std::string path = "./datafile.txt";
  auto optimizer = xacc::getOptimizer("nlopt", {{"nlopt-maxeval", opt_maxiter}});

  // ExaTN visitor
    auto accelerator = xacc::getAccelerator(
        "tnqvm", {{"tnqvm-visitor", "exatn"},
                  {"exatn-buffer-size-gb", exatnBufferSize}});

  // decorate accelerator
  accelerator = xacc::getAcceleratorDecorator(
      "hpc-virtualization", accelerator, {{"n-virtual-qpus", n_virt_qpus}});

  auto mc_vqe = xacc::getAlgorithm("mc-vqe");
  mc_vqe->initialize(
      {{"accelerator", accelerator},
       {"optimizer", optimizer},
       {"interference", false}, {"n-states", n_states},
       {"data-path", path}, {"cyclic", true},
       {"log-level", mcvqeLogLevel}, {"tnqvm-log", true},
       {"nChromophores", n_chromophores}});

  for(int i = 0; i < n_cycles; i++) {
    auto q = xacc::qalloc(n_chromophores);
    xacc::ScopeTimer timer("mpi_timing", false);
    mc_vqe->execute(q);
    auto run_time = timer.getDurationMs();

    // Print the result
    if (q->hasExtraInfoKey("rank") ? ((*q)["rank"].as<int>() == 0) : true) {
      std::cout << "Energy: "
                << q->getInformation("opt-average-energy").as<double>()
                << " \n";
      std::cout << "Circuit depth: " << q->getInformation("circuit-depth").as<int>() << ".\n";
      std::cout << "Total number of gates: " << q->getInformation("n-gates").as<int>() << ".\n";
      std::cout << "Runtime: " << run_time << " ms.\n";
    }
  }
}

int main(int argc, char **argv) {
  xacc::Initialize(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
}
