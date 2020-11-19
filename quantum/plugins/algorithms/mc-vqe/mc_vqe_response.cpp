#include "mc-vqe.hpp"
#include "PauliOperator.hpp"
#include <memory>
#include "xacc.hpp"

using namespace xacc;
using namespace xacc::quantum;

namespace xacc {
namespace algorithm {

std::vector<Eigen::VectorXd>
MC_VQE::getUnrelaxed1PDM(const std::string pauliTerm,
                         const std::shared_ptr<CompositeInstruction> entangler,
                         const Eigen::MatrixXd subspaceRotation,
                         const std::vector<double> x) {

  Eigen::MatrixXd rotatedEigenstates = Eigen::MatrixXd::Zero(nStates, nStates);
  Eigen::MatrixXd gateAngles = statePreparationAngles(rotatedEigenstates);

  std::vector<Eigen::VectorXd> unrelaxed1PDM;
  for (int state = 0; state < nStates; state++) {

    // prepare interference state and append entangler
    auto kernel = statePreparationCircuit(gateAngles.col(state));
    kernel->addVariables(entangler->getVariables());
    for (auto &inst : entangler->getInstructions()) {
      kernel->addInstruction(inst);
    }

    Eigen::VectorXd stateDensityMatrix = Eigen::VectorXd::Zero(nChromophores);
    for (int A = 0; A < nChromophores; A++) {
      auto term = PauliOperator({{A, pauliTerm}});
      stateDensityMatrix(A) =
          vqeWrapper(std::make_shared<PauliOperator>(term), kernel, x);
    }

    unrelaxed1PDM.push_back(stateDensityMatrix);
  }

  return unrelaxed1PDM;
}

std::vector<Eigen::MatrixXd>
MC_VQE::getUnrelaxed2PDM(const std::string pauliTerm,
                         const std::shared_ptr<CompositeInstruction> entangler,
                         const Eigen::MatrixXd subspaceRotation,
                         const std::vector<double> x) {

  Eigen::MatrixXd rotatedEigenstates = Eigen::MatrixXd::Zero(nStates, nStates);
  Eigen::MatrixXd gateAngles = statePreparationAngles(rotatedEigenstates);

  // stores the indices of the valid chromophore pairs
  std::vector<std::vector<int>> pairs(nChromophores);
  for (int A = 0; A < nChromophores; A++) {
    if (A == 0 && isCyclic) {
      pairs[A] = {A + 1, nChromophores - 1};
    } else if (A == nChromophores - 1) {
      pairs[A] = {A - 1, 0};
    } else {
      pairs[A] = {A - 1, A + 1};
    }
  }

  std::vector<Eigen::MatrixXd> unrelaxed2PDM;
  for (int state = 0; state < nStates; state++) {

    // prepare interference state and append entangler
    auto kernel = statePreparationCircuit(gateAngles.col(state));
    kernel->addVariables(entangler->getVariables());
    for (auto &inst : entangler->getInstructions()) {
      kernel->addInstruction(inst);
    }

    Eigen::VectorXd stateDensityMatrix = Eigen::VectorXd::Zero(nChromophores);
    for (int A = 0; A < nChromophores; A++) {
      for (int B : pairs[A]) {
        auto term = PauliOperator({{A, pauliTerm[0]}, {B, pauliTerm[1]}});
        // auto term = PauliOperator({{A, "X"}, {B, "Y"}});
        stateDensityMatrix(A, B) =
            vqeWrapper(std::make_shared<PauliOperator>(term), kernel, x);
      }
    }

    unrelaxed2PDM.push_back(stateDensityMatrix);
  }

  return unrelaxed2PDM;
}

Eigen::VectorXd
MC_VQE::getVQE1PDM(const std::string pauliTerm,
                   const std::shared_ptr<CompositeInstruction> entangler,
                   const Eigen::MatrixXd subspaceRotation,
                   const std::vector<double> x) {

  Eigen::MatrixXd rotatedEigenstates = Eigen::MatrixXd::Zero(nStates, nStates);
  Eigen::MatrixXd gateAngles = statePreparationAngles(rotatedEigenstates);
  int nParams = x.size();
  std::vector<double> tmp_x;

  std::vector<Eigen::VectorXd> gradients;
  // compute gradient for each state energy w.r.t. to entangler parameters
  for (int state = 0; state < nStates; state++) {

    // prepare interference state and append entangler
    auto kernel = statePreparationCircuit(gateAngles.col(state));
    kernel->addVariables(entangler->getVariables());
    for (auto &inst : entangler->getInstructions()) {
      kernel->addInstruction(inst);
    }

    // Eq. 124
    Eigen::VectorXd stateDensityMatrix = Eigen::VectorXd::Zero(nParams);
    for (int g = 0; g < nParams; g++) {
      tmp_x = x;
      tmp_x[g] += xacc::constants::pi / 4.0;
      stateDensityMatrix(g) = vqeWrapper(observable, kernel, tmp_x);

      tmp_x = x;
      tmp_x[g] -= xacc::constants::pi / 4.0;
      stateDensityMatrix(g) -= vqeWrapper(observable, kernel, tmp_x);
    }

    gradients.push_back(stateDensityMatrix);
  }

  // Now compute Hessian
  Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(nParams, nParams);
  for (int state = 0; state < nStates; state++) {

    // prepare interference state and append entangler
    auto kernel = statePreparationCircuit(CISGateAngles.col(state));
    kernel->addVariables(entangler->getVariables());
    for (auto &inst : entangler->getInstructions()) {
      kernel->addInstruction(inst);
    }

    // Eq. 124
    // diagonal terms
    for (int g = 0; g < nParams; g++) {

      // Eq. 125 term #1
      tmp_x = x;
      tmp_x[g] += xacc::constants::pi / 2.0;
      hessian(g, g) += vqeWrapper(observable, kernel, tmp_x);

      // Eq. 125 term #2
      hessian(g, g) += vqeWrapper(observable, kernel, x);

      // Eq. 125 term #3
      tmp_x = x;
      tmp_x[g] -= xacc::constants::pi / 2.0;
      hessian(g, g) -= vqeWrapper(observable, kernel, tmp_x);
    }

    // off-diagonal terms

    for (int g = 1; g < nParams; g++) {
      for (int gp = g; gp < nParams - 1; gp++) {

        tmp_x = x;
        tmp_x[g] += xacc::constants::pi / 4.0;
        tmp_x[gp] += xacc::constants::pi / 4.0;
        hessian(g, gp) += vqeWrapper(observable, kernel, tmp_x);

        tmp_x = x;
        tmp_x[g] += xacc::constants::pi / 4.0;
        tmp_x[gp] -= xacc::constants::pi / 4.0;
        hessian(g, gp) -= vqeWrapper(observable, kernel, tmp_x);

        tmp_x = x;
        tmp_x[g] -= xacc::constants::pi / 4.0;
        tmp_x[gp] += xacc::constants::pi / 4.0;
        hessian(g, gp) -= vqeWrapper(observable, kernel, tmp_x);

        tmp_x = x;
        tmp_x[g] -= xacc::constants::pi / 4.0;
        tmp_x[gp] -= xacc::constants::pi / 4.0;
        hessian(g, gp) += vqeWrapper(observable, kernel, tmp_x);
      }
    }
  }

  hessian /= nStates;

  // Eq. 128
  Eigen::VectorXd vqe1PDM = Eigen::VectorXd::Zero(nChromophores);
  for (int A = 0; A < nChromophores; A++) {

    auto term = PauliOperator({{A, pauliTerm}});

    for (int state = 0; state < nStates; state++) {

      // prepare interference state and append entangler
      auto kernel = statePreparationCircuit(CISGateAngles.col(state));
      kernel->addVariables(entangler->getVariables());
      for (auto &inst : entangler->getInstructions()) {
        kernel->addInstruction(inst);
      }

      Eigen::VectorXd theta_g =
          hessian.colPivHouseholderQr().solve(-gradients[state]);

      for (int g = 0; g < nParams; g++) {

        tmp_x = x;
        tmp_x[g] += xacc::constants::pi / 4.0;
        vqe1PDM(A) +=
            theta_g(g) *
            vqeWrapper(std::make_shared<PauliOperator>(term), kernel, tmp_x);

        tmp_x = x;
        tmp_x[g] -= xacc::constants::pi / 4.0;
        vqe1PDM(A) -=
            theta_g(g) *
            vqeWrapper(std::make_shared<PauliOperator>(term), kernel, tmp_x);
      }
    }
  }

  vqe1PDM /= nStates;

  return vqe1PDM;
}

Eigen::MatrixXd
MC_VQE::getVQE2PDM(const std::string pauliTerm,
                   const std::shared_ptr<CompositeInstruction> entangler,
                   const Eigen::MatrixXd subspaceRotation,
                   const std::vector<double> x) {

  Eigen::MatrixXd rotatedEigenstates = Eigen::MatrixXd::Zero(nStates, nStates);
  Eigen::MatrixXd gateAngles = statePreparationAngles(rotatedEigenstates);
  int nParams = x.size();
  std::vector<double> tmp_x;

  // stores the indices of the valid chromophore pairs
  std::vector<std::vector<int>> pairs(nChromophores);
  for (int A = 0; A < nChromophores; A++) {
    if (A == 0 && isCyclic) {
      pairs[A] = {A + 1, nChromophores - 1};
    } else if (A == nChromophores - 1) {
      pairs[A] = {A - 1, 0};
    } else {
      pairs[A] = {A - 1, A + 1};
    }
  }

  std::vector<Eigen::VectorXd> gradients;
  // compute gradient for each state energy w.r.t. to entangler parameters
  for (int state = 0; state < nStates; state++) {

    // prepare interference state and append entangler
    auto kernel = statePreparationCircuit(gateAngles.col(state));
    kernel->addVariables(entangler->getVariables());
    for (auto &inst : entangler->getInstructions()) {
      kernel->addInstruction(inst);
    }

    // Eq. 124
    Eigen::VectorXd stateDensityMatrix = Eigen::VectorXd::Zero(nParams);
    for (int g = 0; g < nParams; g++) {
      tmp_x = x;
      tmp_x[g] += xacc::constants::pi / 4.0;
      stateDensityMatrix(g) = vqeWrapper(observable, kernel, tmp_x);

      tmp_x = x;
      tmp_x[g] -= xacc::constants::pi / 4.0;
      stateDensityMatrix(g) -= vqeWrapper(observable, kernel, tmp_x);
    }

    gradients.push_back(stateDensityMatrix);
  }

  // Now compute Hessian
  Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(nParams, nParams);
  for (int state = 0; state < nStates; state++) {

    // prepare interference state and append entangler
    auto kernel = statePreparationCircuit(CISGateAngles.col(state));
    kernel->addVariables(entangler->getVariables());
    for (auto &inst : entangler->getInstructions()) {
      kernel->addInstruction(inst);
    }

    // Eq. 124
    // diagonal terms
    for (int g = 0; g < nParams; g++) {

      // Eq. 125 term #1
      tmp_x = x;
      tmp_x[g] += xacc::constants::pi / 2.0;
      hessian(g, g) += vqeWrapper(observable, kernel, tmp_x);

      // Eq. 125 term #2
      hessian(g, g) += vqeWrapper(observable, kernel, x);

      // Eq. 125 term #3
      tmp_x = x;
      tmp_x[g] -= xacc::constants::pi / 2.0;
      hessian(g, g) -= vqeWrapper(observable, kernel, tmp_x);
    }

    // off-diagonal terms

    for (int g = 1; g < nParams; g++) {
      for (int gp = g; gp < nParams - 1; gp++) {

        tmp_x = x;
        tmp_x[g] += xacc::constants::pi / 4.0;
        tmp_x[gp] += xacc::constants::pi / 4.0;
        hessian(g, gp) += vqeWrapper(observable, kernel, tmp_x);

        tmp_x = x;
        tmp_x[g] += xacc::constants::pi / 4.0;
        tmp_x[gp] -= xacc::constants::pi / 4.0;
        hessian(g, gp) -= vqeWrapper(observable, kernel, tmp_x);

        tmp_x = x;
        tmp_x[g] -= xacc::constants::pi / 4.0;
        tmp_x[gp] += xacc::constants::pi / 4.0;
        hessian(g, gp) -= vqeWrapper(observable, kernel, tmp_x);

        tmp_x = x;
        tmp_x[g] -= xacc::constants::pi / 4.0;
        tmp_x[gp] -= xacc::constants::pi / 4.0;
        hessian(g, gp) += vqeWrapper(observable, kernel, tmp_x);
      }
    }
  }

  hessian /= nStates;

  // Eq. 128
  Eigen::VectorXd vqe1PDM = Eigen::VectorXd::Zero(nChromophores);
  for (int A = 0; A < nChromophores; A++) {

    for (int B : pairs[A]) {

      auto term = PauliOperator({{A, pauliTerm[0]}, {B, pauliTerm[1]}});

      for (int state = 0; state < nStates; state++) {

        // prepare interference state and append entangler
        auto kernel = statePreparationCircuit(CISGateAngles.col(state));
        kernel->addVariables(entangler->getVariables());
        for (auto &inst : entangler->getInstructions()) {
          kernel->addInstruction(inst);
        }

        Eigen::VectorXd theta_g =
            hessian.colPivHouseholderQr().solve(-gradients[state]);

        for (int g = 0; g < nParams; g++) {

          tmp_x = x;
          tmp_x[g] += xacc::constants::pi / 4.0;
          vqe1PDM(A) +=
              theta_g(g) *
              vqeWrapper(std::make_shared<PauliOperator>(term), kernel, tmp_x);

          tmp_x = x;
          tmp_x[g] -= xacc::constants::pi / 4.0;
          vqe1PDM(A) -=
              theta_g(g) *
              vqeWrapper(std::make_shared<PauliOperator>(term), kernel, tmp_x);
        }
      }
    }
  }

  vqe1PDM /= nStates;

  return vqe1PDM;
}

} // namespace algorithm
} // namespace xacc
