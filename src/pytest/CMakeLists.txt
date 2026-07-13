set(COMPONENT_NAME align_angle)
install(FILES __init__.py test_align_angle.py test_nematic_pair.py test_sinsq_dihedral.py
              test_soft_harmonic_bond.py test_soft_harmonic_angle.py test_external_patch.py
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${COMPONENT_NAME}/pytest)
