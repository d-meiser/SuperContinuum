set(NPP_HDRS
    fft.h
   )
install(FILES ${NPP_HDRS} ${PROJECT_BINARY_DIR}/NppExport.h
    DESTINATION include)

