// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

/*! \file SinSqDihedralForceComputeGPU.cc
    \brief Defines SinSqDihedralForceComputeGPU — host-side GPU launcher
*/

#include "SinSqDihedralForceComputeGPU.h"
#include "MixedPrecisionCompat.h"

using namespace std;

namespace hoomd
    {
namespace md
    {

SinSqDihedralForceComputeGPU::SinSqDihedralForceComputeGPU(
    std::shared_ptr<SystemDefinition> sysdef)
    : SinSqDihedralForceCompute(sysdef)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error()
            << "Creating a SinSqDihedralForceComputeGPU with no GPU in the execution configuration"
            << endl;
        throw std::runtime_error("Error initializing SinSqDihedralForceComputeGPU");
        }

    GPUArray<Scalar4> params(m_dihedral_data->getNTypes(), m_exec_conf);
    m_params.swap(params);

    m_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                   m_exec_conf,
                                   "sinsq_dihedral"));
    m_autotuners.push_back(m_tuner);
    }

SinSqDihedralForceComputeGPU::~SinSqDihedralForceComputeGPU() { }

void SinSqDihedralForceComputeGPU::setParams(unsigned int type,
                                              Scalar K,
                                              Scalar sign,
                                              int multiplicity,
                                              Scalar phi_0)
    {
    SinSqDihedralForceCompute::setParams(type, K, sign, multiplicity, phi_0);

    ArrayHandle<Scalar4> h_params(m_params, access_location::host, access_mode::readwrite);
    h_params.data[type]
        = make_scalar4(Scalar(K), Scalar(sign), Scalar(multiplicity), Scalar(phi_0));
    }

void SinSqDihedralForceComputeGPU::computeForces(uint64_t timestep)
    {
    ArrayHandle<DihedralData::members_t> d_gpu_dihedral_list(m_dihedral_data->getGPUTable(),
                                                             access_location::device,
                                                             access_mode::read);
    ArrayHandle<unsigned int> d_n_dihedrals(m_dihedral_data->getNGroupsArray(),
                                            access_location::device,
                                            access_mode::read);
    ArrayHandle<unsigned int> d_dihedrals_ABCD(m_dihedral_data->getGPUPosTable(),
                                               access_location::device,
                                               access_mode::read);

    ArrayHandle<ForceReal4> d_pos(
#ifdef HOOMD_HAS_FORCEREAL
        m_pdata->getPositionsForceReal(),
#else
        m_pdata->getPositions(),
#endif
        access_location::device, access_mode::read);
    BoxDim box = m_pdata->getGlobalBox();

    ArrayHandle<ForceReal4> d_force(m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<ForceReal> d_virial(m_virial, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_params(m_params, access_location::device, access_mode::read);

    m_tuner->begin();
    kernel::gpu_compute_sinsq_dihedral_forces(d_force.data,
                                               d_virial.data,
                                               m_virial.getPitch(),
                                               m_pdata->getN(),
                                               d_pos.data,
                                               box,
                                               d_gpu_dihedral_list.data,
                                               d_dihedrals_ABCD.data,
                                               m_dihedral_data->getGPUTableIndexer().getW(),
                                               d_n_dihedrals.data,
                                               d_params.data,
                                               m_dihedral_data->getNTypes(),
                                               this->m_tuner->getParam()[0],
                                               this->m_exec_conf->dev_prop.warpSize);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();
    }

namespace detail
    {
void export_SinSqDihedralForceComputeGPU(pybind11::module& m)
    {
    pybind11::class_<SinSqDihedralForceComputeGPU,
                     SinSqDihedralForceCompute,
                     std::shared_ptr<SinSqDihedralForceComputeGPU>>(
        m,
        "SinSqDihedralForceComputeGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("setParams", &SinSqDihedralForceComputeGPU::setParamsPython)
        .def("getParams", &SinSqDihedralForceComputeGPU::getParams);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
