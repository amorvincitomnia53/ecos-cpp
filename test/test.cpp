//#define EIGEN_INITIALIZE_MATRICES_BY_ZERO
#include <Eigen/Core>
#include <ecos_cpp.hpp>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/QR>
#include <Eigen/SparseCore>
#include <array>
#include <chrono>
#include <fenv.h>
#include <iostream>

constexpr int N = 4;
template <int H, int W>
std::array<Eigen::Matrix<double, H, W>, N> zeros()
{
    std::array<Eigen::Matrix<double, H, W>, N> ret;
    for (int i = 0; i < N; i++) {
        ret[i].setZero();
    }
    return ret;
}
int main(int argc, char** argv)
{
    //    feenableexcept(FE_INVALID);
    Eigen::Vector2d g_pos = {0.0, 0.0};
    std::array<Eigen::Vector2d, N> pos = {{{1.0, 1.0}, {-1.0, 1.0}, {-1.0, -1.0}, {1.0, -1.0}}};
    for (int i = 0; i < N; i++) {
        pos[i] -= g_pos;
    }
    double g_z = 0.25;

    Eigen::Matrix<double, 3, N> load_distri_pre_mat;
    for (int i = 0; i < N; i++) {
        load_distri_pre_mat.block<3, 1>(0, i) << pos[i], 1.0;
    }
    Eigen::Matrix<double, 3, 2> load_distri_b_mat;
    load_distri_b_mat << 1.0, 0.0,
        0.0, 1.0,
        0.0, 0.0;
    Eigen::Matrix<double, N, 2> load_distri_mat = load_distri_pre_mat.completeOrthogonalDecomposition().solve(load_distri_b_mat);
    std::cerr << load_distri_mat << std::endl;

    std::array<double, N> mu{{1.0, 1.0, 1.0, 1.0}};

    std::vector<Eigen::Triplet<double>> A_triplet;
    std::vector<Eigen::Triplet<double>> G_triplet;
    std::array<Eigen::Matrix<double, 1, 2>, N> q;
    for (int i = 0; i < N; i++) {
        q[i] = -g_z * load_distri_mat.block<1, 2>(i, 0);
        A_triplet.push_back({0, 2 * i + 0, 1.0});
        A_triplet.push_back({1, 2 * i + 1, 1.0});
        A_triplet.push_back({2, 2 * i + 0, -pos[i].y()});
        A_triplet.push_back({2, 2 * i + 1, pos[i].x()});
        G_triplet.push_back({3 * i + 0, 2 * N, -mu[i]});
        G_triplet.push_back({3 * i + 1, 2 * i + 0, -1.0});
        G_triplet.push_back({3 * i + 2, 2 * i + 1, -1.0});
    }

    ecos::SOCPProblem problem(2 * N + 1, {0, std::vector<idxint>(N, 3), 0}, 3);
    problem.linear.A.setFromTriplets(A_triplet.begin(), A_triplet.end());
    problem.linear.b.setZero();
    problem.cone.G.setFromTriplets(G_triplet.begin(), G_triplet.end());
    problem.cone.h.setZero();
    problem.target.setZero();
    problem.target(2 * N) = 1.0;

    ecos::SOCPSolver solver(std::move(problem));

    auto solve = [&](
                     const Eigen::Matrix<double, 3, 1>& h,
                     const Eigen::Matrix<double, 1, 3>& p0_tmp,
                     //  const std::array<Eigen::Matrix<double, 2, 1>, N>& f0 = zeros<2, 1>(),
                     int iter = 10,
                     bool verbose = false) {
        for (int i = 0; i < N; i++) {
            solver.updatehComponent(3 * i + 0, q[i].dot(h.segment<2>(0)));
            solver.updatehComponent(3 * i + 1, 0.0);
            solver.updatehComponent(3 * i + 2, 0.0);
        }
        solver.updatebComponent(0, h(0));
        solver.updatebComponent(1, h(1));
        solver.updatebComponent(2, h(2));


        if (verbose) {
            std::cout << solver.problem << std::endl;
        }
        auto res = solver.solve();

        std::array<Eigen::Matrix<double, 2, 1>, N> f;
        double k = solver.x()(2 * N);
        for (int i = 0; i < N; i++) {
            f[i] = solver.x().segment<2>(2 * i);
        }
        if (verbose) {
            std::cout << "res: " << static_cast<int>(res) << " k: " << k << std::endl;
        }
        return std::make_pair(f, k);
    };


    if (argc > 1) {
        Eigen::Vector3d h0 = {std::atof(argv[1]),
            std::atof(argv[2]),
            std::atof(argv[3])};
        auto [f, k] = solve(h0, h0.transpose(), 100, true);
        Eigen::Matrix<double, 3, 1> h = h0 / k;

        std::cout << "answer:";
        for (int i = 0; i < N; i++) {
            std::cout << " [" << i << "] " << f[i].x() << " " << f[i].y();
        }
        std::cout << "\n";

        std::cout << "h: " << h.transpose() << " k: " << k << "\n";
        return 0;
    }
    constexpr int NTH = 50;
    constexpr int NPHI = 50;

    std::array<std::array<Eigen::Matrix<double, 3, 1>, NPHI>, NTH> data;
    auto start_time = std::chrono::steady_clock::now();
    for (int ith = 0; ith < NTH; ith++) {
        for (int iphi = 0; iphi < NPHI; iphi++) {
            double th = M_PI / NTH * ith;
            double phi = 2 * M_PI / NPHI * iphi;

            Eigen::Vector3d h0 = {std::sin(th) * std::cos(phi),
                std::sin(th) * std::sin(phi),
                std::cos(th)};
            auto [f, k] = solve(h0, h0.transpose(), 10);
            Eigen::Matrix<double, 3, 1> h = h0 / k;
            data[ith][iphi] = h;
        }
    }
    auto end_time = std::chrono::steady_clock::now();
    for (int ith = 0; ith < NTH; ith++) {
        for (int iphi = 0; iphi < NPHI; iphi++) {
            double th = M_PI / NTH * ith;
            double phi = 2 * M_PI / NPHI * iphi;
            Eigen::Vector3d p{std::sin(th) * std::cos(phi),
                std::sin(th) * std::sin(phi),
                std::cos(th)};
            std::cout << "(px,py,pth,ax,ay,ath): " << p.x() << " " << p.y() << " " << p.z() << " " << data[ith][iphi].x() << " " << data[ith][iphi].y() << " " << data[ith][iphi].z() << "\n";
        }
    }
    std::cerr << "average time: " << (end_time - start_time).count() / NPHI / NTH * 1e-3 << "us" << std::endl;
}