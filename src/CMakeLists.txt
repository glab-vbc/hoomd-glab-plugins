set(COMPONENT_NAME align_angle)

set(_${COMPONENT_NAME}_sources
    module.cc
    AlignAngleForceCompute.cc
    AnisoPotentialPairNematic.cc
    SinSqDihedralForceCompute.cc
    )

set(_${COMPONENT_NAME}_headers
    AlignAngleForceCompute.h
    EvaluatorPairNematic.h
    SinSqDihedralForceCompute.h
    )

if (ENABLE_HIP)
    list(APPEND _${COMPONENT_NAME}_sources
         AlignAngleForceComputeGPU.cc
         AnisoPotentialPairNematicGPU.cc
         SinSqDihedralForceComputeGPU.cc
         )
    list(APPEND _${COMPONENT_NAME}_headers
         AlignAngleForceComputeGPU.h
         AlignAngleForceGPU.cuh
         SinSqDihedralForceComputeGPU.h
         SinSqDihedralForceGPU.cuh
         )
    set(_${COMPONENT_NAME}_cu_sources
        AlignAngleForceGPU.cu
        AnisoPotentialPairNematicGPUKernel.cu
        SinSqDihedralForceGPU.cu
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

# Install the compiled module
install(TARGETS _${COMPONENT_NAME}
        LIBRARY DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${COMPONENT_NAME})

# Install Python files
install(FILES __init__.py version.py
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${COMPONENT_NAME})

add_subdirectory(pytest)
