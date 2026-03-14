set(COMPONENT_NAME align_angle)
install(FILES test_align_angle.py test_nematic_pair.py test_sinsq_dihedral.py
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${COMPONENT_NAME}/pytest)
