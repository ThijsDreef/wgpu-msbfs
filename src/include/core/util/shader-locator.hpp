
#define USE_FILES

#ifdef USE_FILES
// TODO this requires someone to free the returned pointer
char* getShaderFile(const char* value);
#define LOCATE_SHADER(name) getShaderFile(name)

#else
// Provide some static way to get data.
// We do not want to call free on this method.
#endif
