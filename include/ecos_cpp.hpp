#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <cassert>
#include <ecos.h>
#include <iostream>
#include <numeric>
#include <vector>
namespace ecos
{
struct ConeDescription {
    idxint dim_Rplus;
    std::vector<idxint> dim_quadratic_cones;
    idxint dim_exponential_cones;

    idxint dimension() const
    {
        return dim_Rplus + std::accumulate(dim_quadratic_cones.begin(), dim_quadratic_cones.end(), 0) + dim_exponential_cones;
    }
};
struct ConeConstraint {
    // h - Gx ∈ K
    Eigen::VectorXd h;
    Eigen::SparseMatrix<double, Eigen::ColMajor> G;
    ConeDescription K;
    ConeConstraint() = default;
    ConeConstraint(const ConeConstraint&) = default;
    ConeConstraint(ConeConstraint&&) = default;
    ConeConstraint& operator=(const ConeConstraint&) = default;
    ConeConstraint& operator=(ConeConstraint&&) = default;

    ConeConstraint(idxint x_num, idxint constraint_dim) : h(constraint_dim), G(constraint_dim, x_num), K{} {}
    ConeConstraint(idxint x_num, ConeDescription cone) : ConeConstraint(x_num, cone.dimension())
    {
        K = std::move(cone);
    }

    idxint variableDimension() const { return G.cols(); }
    idxint constraintDimension() const
    {
        assert(h.rows() == G.rows() && h.rows() == K.dimension());
        return h.rows();
    }
};

struct LinearConstraint {
    // Ax - b = 0
    Eigen::SparseMatrix<double, Eigen::ColMajor> A;
    Eigen::VectorXd b;
    LinearConstraint() = default;
    LinearConstraint(const LinearConstraint&) = default;
    LinearConstraint(LinearConstraint&&) = default;
    LinearConstraint& operator=(const LinearConstraint&) = default;
    LinearConstraint& operator=(LinearConstraint&&) = default;
    LinearConstraint(idxint x_num, idxint constraint_dim) : A(constraint_dim, x_num), b(constraint_dim) {}

    idxint variableDimension() const { return A.cols(); }
    idxint constraintDimension() const
    {
        assert(A.rows() == b.rows());
        return A.rows();
    }
};

struct SOCPProblem {
    // minimize c^T x
    //    s.t.  (h - Gx) ∈ K, Ax - b = 0

    Eigen::VectorXd target;   // =: c
    ConeConstraint cone;      // =: (h, G, K)
    LinearConstraint linear;  // =: (A, b)
    SOCPProblem() = default;
    SOCPProblem(const SOCPProblem&) = default;
    SOCPProblem(SOCPProblem&&) = default;
    SOCPProblem& operator=(const SOCPProblem&) = default;
    SOCPProblem& operator=(SOCPProblem&&) = default;

    SOCPProblem(idxint x_num, ConeDescription cone_description, idxint linear_constraint_dim = 0)
        : target(x_num), cone(x_num, std::move(cone_description)), linear(x_num, linear_constraint_dim) {}

    SOCPProblem(idxint x_num, idxint cone_constraint_dim, idxint linear_constraint_dim = 0)
        : target(x_num), cone(x_num, cone_constraint_dim), linear(x_num, linear_constraint_dim) {}

    SOCPProblem(Eigen::VectorXd target, ConeConstraint cone, LinearConstraint linear)
        : target(std::move(target)), cone(std::move(cone)), linear(std::move(linear)) {}

    idxint variableDimension() const
    {
        assert(target.rows() == cone.variableDimension() && target.rows() == linear.variableDimension());
        return target.rows();
    }
    idxint coneConstraintDimension() const
    {
        return cone.constraintDimension();
    }

    idxint linearConstraintDimension() const
    {
        return linear.constraintDimension();
    }

    friend std::ostream& operator<<(std::ostream& os, const SOCPProblem& problem)
    {
        return os << "A:\n"
                  << Eigen::MatrixXd(problem.linear.A) << "\n"
                  << "b:\n"
                  << problem.linear.b << "\n"
                  << "G:\n"
                  << Eigen::MatrixXd(problem.cone.G) << "\n"
                  << "h:\n"
                  << problem.cone.h << "\n"
                  << "c: " << problem.target.transpose() << "\n";
    }
};

enum class SOCPResult {
    Optimal = 0,
    Infeasible = 1,
    DualInfeasible = 2,
    OptimalInaccurate = 10,
    InfeasibleInaccurate = 11,
    DualInfeasibleInaccurate = 12,
    MaximumIteration = -1,
    NumericalIssue = -2,
    OutsideCone = -3,
    Interrupted = -4,
    FatalError = -7
};
struct SOCPSolver {

    SOCPProblem problem;
    pwork* impl = nullptr;
    explicit SOCPSolver(SOCPProblem problem_)
        : problem{std::move(problem_)}
    {

        problem.cone.G.makeCompressed();
        problem.linear.A.makeCompressed();

        idxint n = problem.variableDimension();
        idxint m = problem.coneConstraintDimension();
        idxint p = problem.linearConstraintDimension();

        idxint l = problem.cone.K.dim_Rplus;
        idxint ncones = problem.cone.K.dim_quadratic_cones.size();
        const idxint* q = problem.cone.K.dim_quadratic_cones.data();
        idxint e = problem.cone.K.dim_exponential_cones;

        impl = ECOS_setup(n, m, p, l, ncones, const_cast<idxint*>(q), e,
            problem.cone.G.valuePtr(), problem.cone.G.outerIndexPtr(), problem.cone.G.innerIndexPtr(),
            problem.linear.A.valuePtr(), problem.linear.A.outerIndexPtr(), problem.linear.A.innerIndexPtr(),
            problem.target.data(), problem.cone.h.data(), problem.linear.b.data());
    }

    SOCPResult solve()
    {
        return static_cast<SOCPResult>(ECOS_solve(impl));
    }
    /*
     * Updates one element of the RHS vector h of inequalities
     * After the call, w->h[idx] = value (but equilibrated)
     */
    void updatehComponent(idxint idx, double value)
    {
        ecos_updateDataEntry_h(impl, idx, value);
    }


    /*
     * Updates one element of the RHS vector b of equalities
     * After the call, w->b[idx] = value (but equilibrated)
     */
    void updatebComponent(idxint idx, double value)
    {
#if EQUILIBRATE > 0
        impl->b[idx] = value / impl->Aequil[idx];
#else
        impl->b[idx] = value;
#endif
    }
    /*
 * Updates one element of the OBJ vector c of inequalities
 * After the call, w->c[idx] = value (but equilibrated)
 */
    void updateTargetComponent(idxint idx, double value)
    {
        ecos_updateDataEntry_c(impl, idx, value);
    }
    /*
 * Updates numerical data for G, A, c, h, and b,
 * and re-equilibrates.
 * Then updates the corresponding KKT entries.
 */
    void updateProblem()
    {
        ECOS_updateData(impl, problem.cone.G.valuePtr(), problem.linear.A.valuePtr(),
            problem.target.data(), problem.cone.h.data(), problem.linear.b.data());
    }


    Eigen::Map<Eigen::VectorXd> x() const
    {
        assert(impl != nullptr);
        return {impl->x, problem.variableDimension()};
    }

    double cost() const
    {
        assert(impl != nullptr);
        return impl->cx;
    }

    ~SOCPSolver() noexcept
    {
        if (impl != nullptr) {
            ECOS_cleanup(impl, 0);
        }
    }
};

}  // namespace ecos
