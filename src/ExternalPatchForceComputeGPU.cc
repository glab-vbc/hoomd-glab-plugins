// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

/*! \file ExternalPatchForceComputeGPU.cc
    \brief Defines ExternalPatchForceComputeGPU — host-side code for the
           GPU external-patch force compute.
*/

#include "ExternalPatchForceComputeGPU.h"
#include "MixedPrecisionCompat.h"

using namespace std;

namespace hoomd
    {
namespace md
    {

ExternalPatchForceComputeGPU::ExternalPatchForceComputeGPU(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<NeighborList> nlist)
    : ExternalPatchForceCompute(sysdef, nlist)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error()
            << "Creating an ExternalPatchForceComputeGPU with no GPU in the "
               "execution configuration"
            << endl;
        throw std::runtime_error("Error initializing ExternalPatchForceComputeGPU");
        }

    m_tuner.reset(new Autotuner<2>({AutotunerBase::makeBlockSizeRange(m_exec_conf),
                                    AutotunerBase::getTppListPow2(m_exec_conf)},
                                   m_exec_conf,
                                   "external_patch"));
    m_autotuners.push_back(m_tuner);
    }

ExternalPatchForceComputeGPU::~ExternalPatchForceComputeGPU() { }

void ExternalPatchForceComputeGPU::computeForces(uint64_t timestep)
    {
    // Update the neighbor list
    m_nlist->compute(timestep);

    // Require full neighbor list
    bool third_law = m_nlist->getStorageMode() == NeighborList::half;
    if (third_law)
        {
        m_exec_conf->msg->error()
            << "ExternalPatchForceComputeGPU requires a full neighbor list" << endl;
        throw std::runtime_error(
            "Error computing forces in ExternalPatchForceComputeGPU");
        }

    // Access particle data on device
    ArrayHandle<ForceReal4> d_pos(
#ifdef HOOMD_HAS_FORCEREAL
        m_pdata->getPositionsForceReal(),
#else
        m_pdata->getPositions(),
#endif
        access_location::device,
        access_mode::read);

    ArrayHandle<unsigned int> d_tag(m_pdata->getTags(),
                                    access_location::device,
                                    access_mode::read);
    ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(),
                                     access_location::device,
                                     access_mode::read);

    // Access partner tags on device
    ArrayHandle<int> d_partners(m_partner_tags,
                                access_location::device,
                                access_mode::read);

    // Access neighbor list on device
    ArrayHandle<unsigned int> d_n_neigh(m_nlist->getNNeighArray(),
                                        access_location::device,
                                        access_mode::read);
    ArrayHandle<unsigned int> d_nlist(m_nlist->getNListArray(),
                                      access_location::device,
                                      access_mode::read);
    ArrayHandle<size_t> d_head_list(m_nlist->getHeadList(),
                                    access_location::device,
                                    access_mode::read);

    BoxDim box = m_pdata->getGlobalBox();

    // Access force/virial output (overwrite — kernel driver zeros first)
    ArrayHandle<ForceReal4> d_force(m_force,
                                    access_location::device,
                                    access_mode::overwrite);
    ArrayHandle<ForceReal> d_virial(m_virial,
                                    access_location::device,
                                    access_mode::overwrite);

    // Launch the kernel
    m_tuner->begin();
    auto param = m_tuner->getParam();
    unsigned int block_size = param[0];
    unsigned int threads_per_particle = param[1];

    (void)kernel::gpu_compute_external_patch_forces(
        d_force.data,
        d_virial.data,
        m_virial.getPitch(),
        m_pdata->getN(),
        m_pdata->getNGhosts(),
        d_pos.data,
        d_tag.data,
        d_rtag.data,
        d_partners.data,
        m_partner_tags_size,
        d_n_neigh.data,
        d_nlist.data,
        d_head_list.data,
        box,
        ForceReal(m_epsilon),
        ForceReal(m_rcutsq),
        ForceReal(m_width),
        block_size,
        threads_per_particle);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();
    }

namespace detail
    {
void export_ExternalPatchForceComputeGPU(pybind11::module& m)
    {
    pybind11::class_<ExternalPatchForceComputeGPU,
                     ExternalPatchForceCompute,
                     std::shared_ptr<ExternalPatchForceComputeGPU>>(
        m,
        "ExternalPatchForceComputeGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<NeighborList>>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
