/*******************************************************************************
 * Copyright (c) 2020 UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * and Eclipse Distribution License v1.0 which accompanies this
 * distribution. The Eclipse Public License is available at
 * http://www.eclipse.org/legal/epl-v10.html and the Eclipse Distribution
 *License is available at https://eclipse.org/org/documents/edl-v10.php
 *
 * Contributors:
 *   Daniel Claudino - initial API and implementation
 *******************************************************************************/
#ifndef XACC_ALGORITHM_MC_VQE_HPP_
#define XACC_ALGORITHM_MC_VQE_HPP_

#include "Algorithm.hpp"
#include "xacc_service.hpp"
#include <Eigen/Dense>
#include <chrono>
#include <memory>
#include <vector>
#include "AlgorithmGradientStrategy.hpp"

namespace xacc {
namespace algorithm {
class MC_VQE : public Algorithm {
protected:
  Optimizer *optimizer;
  Accelerator *accelerator;
  HeterogeneousMap parameters;
  std::shared_ptr<AlgorithmGradientStrategy> gradientStrategy;

  // MC-VQE variables

  // number of chromophores
  int nChromophores;
  // true if the molecular system is cyclic
  bool isCyclic;
  // if false, will not compute interference matrix
  bool doInterference = true;
  // state preparation angles
  Eigen::MatrixXd CISGateAngles, CISEigenstates;
  // AIEM Hamiltonian
  std::shared_ptr<Observable> observable;
  // # number of CIS states = nChromophores + 1
  int nStates;
  // # of parameters in a single entangler
  // after merging adjacent Ry gates
  const int NPARAMSENTANGLER = 4;
  // angstrom to bohr
  const double ANGSTROM2BOHR = 1.8897161646320724;
  // D to a.u.
  const double DEBYE2AU = 0.393430307;
  // path to file with quantum chemistry data
  std::string dataPath;
  // start time of simulation
  std::chrono::system_clock::time_point start;

  // constructs CIS state preparation circiuit
  std::shared_ptr<CompositeInstruction>
  statePreparationCircuit(const Eigen::VectorXd &stateAngles) const;

  // constructs entangler portion of MC-VQE circuit
  std::shared_ptr<CompositeInstruction> entanglerCircuit() const;

  // processes quantum chemistry data and returns the Hamiltonian
  // and the gates for state preparation
  void preProcessing();

  Eigen::MatrixXd
  statePreparationAngles(const Eigen::MatrixXd CoefficientMatrix);

  double vqeWrapper(const std::shared_ptr<Observable> observable,
                    const std::shared_ptr<CompositeInstruction> kernel,
                    const std::vector<double> x) const;

  // controls the level of printing
  int logLevel = 1;
  bool tnqvmLog = false;
  void logControl(const std::string message, const int level) const;

  // response/gradient
  std::vector<Eigen::VectorXd>
  getUnrelaxed1PDM(const std::string pauliTerm,
                   const std::shared_ptr<CompositeInstruction> entangler,
                   const Eigen::MatrixXd subspaceRotation,
                   const std::vector<double> x);

  std::vector<Eigen::MatrixXd>
  getUnrelaxed2PDM(const std::string pauliTerm,
                   const std::shared_ptr<CompositeInstruction> entangler,
                   const Eigen::MatrixXd subspaceRotation,
                   const std::vector<double> x);

  Eigen::VectorXd
  getVQE1PDM(const std::string pauliTerm,
             const std::shared_ptr<CompositeInstruction> entangler,
             const Eigen::MatrixXd subspaceRotation,
             const std::vector<double> x);

  Eigen::MatrixXd
  getVQE2PDM(const std::string pauliTerm,
             const std::shared_ptr<CompositeInstruction> entangler,
             const Eigen::MatrixXd subspaceRotation,
             const std::vector<double> x);

public:
  bool initialize(const HeterogeneousMap &parameters) override;
  const std::vector<std::string> requiredParameters() const override;
  void execute(const std::shared_ptr<AcceleratorBuffer> buffer) const override;
  std::vector<double> execute(const std::shared_ptr<AcceleratorBuffer> buffer,
                              const std::vector<double> &parameters) override;
  const std::string name() const override { return "mc-vqe"; }
  const std::string description() const override { return ""; }
  DEFINE_ALGORITHM_CLONE(MC_VQE)
};
} // namespace algorithm
} // namespace xacc
#endif