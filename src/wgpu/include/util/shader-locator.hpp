// For development define USE_FILES here

#ifdef USE_FILES
// TODO this requires someone to free the returned pointer
char* getShaderFile(const char* value);
#define LOCATE_SHADER(name) getShaderFile(name)

#else

char* getShaderPointer(const char* value);
#define LOCATE_SHADER(name) getShaderPointer(name)

#endif
