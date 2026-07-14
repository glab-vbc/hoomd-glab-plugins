set(COMPONENT_NAME glab_forces)

set(_${COMPONENT_NAME}_sources
    module.cc
    AlignAngleForceCompute.cc
    AnisoPotentialPairNematic.cc
    ExternalPatchForceCompute.cc
    SinSqDihedralForceCompute.cc
    SoftHarmonicBond.cc
    SoftHarmonicAngleForceCompute.cc
    CosineAngleForceCompute.cc
    )

set(_${COMPONENT_NAME}_headers
    AlignAngleForceCompute.h
    EvaluatorPairNematic.h
    EvaluatorBondSoftHarmonic.h
    ExternalPatchForceCompute.h
    SinSqDihedralForceCompute.h
    SoftHarmonicTail.h
    SoftHarmonicAngleForceCompute.h
    CosineAngleForceCompute.h
    )

if (ENABLE_HIP)
    list(APPEND _${COMPONENT_NAME}_sources
         AlignAngleForceComputeGPU.cc
         AnisoPotentialPairNematicGPU.cc
         ExternalPatchForceComputeGPU.cc
         SinSqDihedralForceComputeGPU.cc
         SoftHarmonicBondGPU.cc
         SoftHarmonicAngleForceComputeGPU.cc
         CosineAngleForceComputeGPU.cc
         )
    list(APPEND _${COMPONENT_NAME}_headers
         AlignAngleForceComputeGPU.h
         AlignAngleForceGPU.cuh
         ExternalPatchForceComputeGPU.h
         ExternalPatchForceGPU.cuh
         SinSqDihedralForceComputeGPU.h
         SinSqDihedralForceGPU.cuh
         SoftHarmonicAngleForceComputeGPU.h
         SoftHarmonicAngleForceGPU.cuh
         CosineAngleForceComputeGPU.h
         CosineAngleForceGPU.cuh
         )
    set(_${COMPONENT_NAME}_cu_sources
        AlignAngleForceGPU.cu
        AnisoPotentialPairNematicGPUKernel.cu
        ExternalPatchForceGPU.cu
        SinSqDihedralForceGPU.cu
        SoftHarmonicBondGPU.cu
        SoftHarmonicAngleForceGPU.cu
        CosineAngleForceGPU.cu
        )
    set_source_files_properties(${_${COMPONENT_NAME}_cu_sources}
        PROPERTIES LANGUAGE ${HOOMD_DEVICE_LANGUAGE})
endif()

hoomd_add_module(_${COMPONENT_NAME} SHARED
    ${_${COMPONENT_NAME}_sources}
    ${_${COMPONENT_NAME}_cu_sources}
    ${_${COMPONENT_NAME}_headers}
    NO_EXTRAS)

# Link to HOOMD core and MD module
target_link_libraries(_${COMPONENT_NAME} PUBLIC HOOMD::_hoomd HOOMD::_md)

# ---------------------------------------------------------------------------
# ForceReal compatibility (see MixedPrecisionCompat.h)
#
# Some HOOMD builds provide a native `ForceReal` type (mixed-precision force
# evaluation). The hoomd-sloptimize fork also `#define`s HOOMD_HAS_FORCEREAL in
# its headers, but other ForceReal-capable builds (e.g. some upstream 6.x
# installs) provide the type WITHOUT that guard macro. In that case
# MixedPrecisionCompat.h would re-typedef `ForceReal = Scalar` and conflict with
# the native (ShortReal) typedef.
#
# Detect the situation from the installed HOOMDMath.h and define the guard macro
# ourselves only when the type exists but the header does not already define it.
# Builds with no native ForceReal fall through to the shim's Scalar aliases, and
# builds that already define the macro are left untouched (no redefinition).
# ---------------------------------------------------------------------------
get_target_property(_hoomd_inc_dirs HOOMD::_hoomd INTERFACE_INCLUDE_DIRECTORIES)
find_file(HOOMD_MATH_HEADER
    NAMES hoomd/HOOMDMath.h
    PATHS ${_hoomd_inc_dirs}
    NO_DEFAULT_PATH)
if (HOOMD_MATH_HEADER)
    file(READ "${HOOMD_MATH_HEADER}" _hoomd_math_src)
    string(FIND "${_hoomd_math_src}" "ForceReal" _fr_type_pos)
    string(FIND "${_hoomd_math_src}" "define HOOMD_HAS_FORCEREAL" _fr_macro_pos)
    if (NOT _fr_type_pos EQUAL -1 AND _fr_macro_pos EQUAL -1)
        message(STATUS
            "glab_forces: HOOMD provides native ForceReal without HOOMD_HAS_FORCEREAL; defining it")
        target_compile_definitions(_${COMPONENT_NAME} PRIVATE HOOMD_HAS_FORCEREAL)
    endif()
    unset(_hoomd_math_src)
else()
    message(WARNING
        "glab_forces: could not locate hoomd/HOOMDMath.h to probe for native ForceReal; "
        "if the build fails with a conflicting 'ForceReal' typedef, configure with "
        "-DCMAKE_CXX_FLAGS=-DHOOMD_HAS_FORCEREAL -DCMAKE_CUDA_FLAGS=-DHOOMD_HAS_FORCEREAL")
endif()

# Install the compiled module
install(TARGETS _${COMPONENT_NAME}
        LIBRARY DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${COMPONENT_NAME})

# Install Python files
install(FILES __init__.py version.py
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${COMPONENT_NAME})

add_subdirectory(pytest)
