// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

/*! \file CosineAngleForceComputeGPU.cc
    \brief Defines CosineAngleForceComputeGPU — host-side GPU launcher
*/

#include "CosineAngleForceComputeGPU.h"
#include "MixedPrecisionCompat.h"

#include <math.h>

using namespace std;

namespace hoomd
    {
namespace md
    {

CosineAngleForceComputeGPU::CosineAngleForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef)
    : CosineAngleForceCompute(sysdef)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a CosineAngleForceComputeGPU with no GPU "
                                     "in the execution configuration"
                                  << endl;
        throw std::runtime_error("Error initializing CosineAngleForceComputeGPU");
        }

    // allocate and zero device memory for parameters
    GPUArray<Scalar4> params(m_angle_data->getNTypes(), m_exec_conf);
    m_params.swap(params);

    m_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                   m_exec_conf,
                                   "cosine_angle"));
    m_autotuners.push_back(m_tuner);
    }

CosineAngleForceComputeGPU::~CosineAngleForceComputeGPU() { }

void CosineAngleForceComputeGPU::setParams(unsigned int type, Scalar K, Scalar t_0)
    {
    CosineAngleForceCompute::setParams(type, K, t_0);

    // precompute cos(t_0), sin(t_0) so the kernel needs no trig or acos
    ArrayHandle<Scalar4> h_params(m_params, access_location::host, access_mode::readwrite);
    h_params.data[type] = make_scalar4(K, t_0, cos(t_0), sin(t_0));
    }

void CosineAngleForceComputeGPU::computeForces(uint64_t timestep)
    {
    ArrayHandle<ForceReal4> d_pos(
#ifdef HOOMD_HAS_FORCEREAL
        m_pdata->getPositionsForceReal(),
#else
        m_pdata->getPositions(),
#endif
        access_location::device,
        access_mode::read);

    BoxDim box = m_pdata->getGlobalBox();

    ArrayHandle<ForceReal4> d_force(m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<ForceReal> d_virial(m_virial, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_params(m_params, access_location::device, access_mode::read);

    ArrayHandle<AngleData::members_t> d_gpu_anglelist(m_angle_data->getGPUTable(),
                                                      access_location::device,
                                                      access_mode::read);
    ArrayHandle<unsigned int> d_gpu_angle_pos_list(m_angle_data->getGPUPosTable(),
                                                   access_location::device,
                                                   access_mode::read);
    ArrayHandle<unsigned int> d_gpu_n_angles(m_angle_data->getNGroupsArray(),
                                             access_location::device,
                                             access_mode::read);

    m_tuner->begin();
    kernel::gpu_compute_cosine_angle_forces(d_force.data,
                                            d_virial.data,
                                            m_virial.getPitch(),
                                            m_pdata->getN(),
                                            d_pos.data,
                                            box,
                                            d_gpu_anglelist.data,
                                            d_gpu_angle_pos_list.data,
                                            m_angle_data->getGPUTableIndexer().getW(),
                                            d_gpu_n_angles.data,
                                            d_params.data,
                                            m_angle_data->getNTypes(),
                                            this->m_tuner->getParam()[0]);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();
    }

namespace detail
    {
void export_CosineAngleForceComputeGPU(pybind11::module& m)
    {
    pybind11::class_<CosineAngleForceComputeGPU,
                     CosineAngleForceCompute,
                     std::shared_ptr<CosineAngleForceComputeGPU>>(m, "CosineAngleForceComputeGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("setParams", &CosineAngleForceComputeGPU::setParamsPython)
        .def("getParams", &CosineAngleForceComputeGPU::getParams);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
