// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

/*! \file AlignAngleForceComputeGPU.cc
    \brief Defines AlignAngleForceComputeGPU
*/

#include "AlignAngleForceComputeGPU.h"

using namespace std;

namespace hoomd
    {
namespace md
    {

AlignAngleForceComputeGPU::AlignAngleForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef)
    : AlignAngleForceCompute(sysdef)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error()
            << "Creating a AlignAngleForceComputeGPU with no GPU in the execution configuration"
            << endl;
        throw std::runtime_error("Error initializing AlignAngleForceComputeGPU");
        }

    // Allocate and zero device memory for per-type parameters (Scalar4: K, m, phase, 0)
    GPUArray<Scalar4> params(m_angle_data->getNTypes(), m_exec_conf);
    m_params_gpu.swap(params);

    m_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                   m_exec_conf,
                                   "align_angle"));
    m_autotuners.push_back(m_tuner);
    }

AlignAngleForceComputeGPU::~AlignAngleForceComputeGPU() { }

void AlignAngleForceComputeGPU::setParams(unsigned int type, Scalar K, unsigned int multiplicity, Scalar phase)
    {
    AlignAngleForceCompute::setParams(type, K, multiplicity, phase);

    ArrayHandle<Scalar4> h_params(m_params_gpu, access_location::host, access_mode::readwrite);
    h_params.data[type] = make_scalar4(K, Scalar(multiplicity), phase, Scalar(0.0));
    }

void AlignAngleForceComputeGPU::computeForces(uint64_t timestep)
    {
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(),
                                       access_location::device,
                                       access_mode::read);

    BoxDim box = m_pdata->getGlobalBox();

    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_torque(m_torque, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_params(m_params_gpu, access_location::device, access_mode::read);

    ArrayHandle<AngleData::members_t> d_gpu_anglelist(m_angle_data->getGPUTable(),
                                                      access_location::device,
                                                      access_mode::read);
    ArrayHandle<unsigned int> d_gpu_angle_pos_list(m_angle_data->getGPUPosTable(),
                                                   access_location::device,
                                                   access_mode::read);
    ArrayHandle<unsigned int> d_gpu_n_angles(m_angle_data->getNGroupsArray(),
                                             access_location::device,
                                             access_mode::read);

    // Run the kernel
    m_tuner->begin();
    (void)kernel::gpu_compute_align_angle_forces(d_force.data,
                                           d_torque.data,
                                           d_virial.data,
                                           m_virial.getPitch(),
                                           m_pdata->getN(),
                                           d_pos.data,
                                           d_orientation.data,
                                           box,
                                           d_gpu_anglelist.data,
                                           d_gpu_angle_pos_list.data,
                                           m_angle_data->getGPUTableIndexer().getW(),
                                           d_gpu_n_angles.data,
                                           d_params.data,
                                           m_angle_data->getNTypes(),
                                           m_tuner->getParam()[0]);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();
    }

namespace detail
    {
void export_AlignAngleForceComputeGPU(pybind11::module& m)
    {
    pybind11::class_<AlignAngleForceComputeGPU,
                     AlignAngleForceCompute,
                     std::shared_ptr<AlignAngleForceComputeGPU>>(m, "AlignAngleForceComputeGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
