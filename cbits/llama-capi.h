#ifndef __LLAMA_HASKELL_CBITS_LLAMA_CAPI_H__
#define __LLAMA_HASKELL_CBITS_LLAMA_CAPI_H__

#include "llama.h"

// Some LLAMA.cpp C APIs returns a struct, or take struct as argument,
// which are not supported by Haskell FFI.

#ifdef __cplusplus
extern "C" {
#endif

LLAMA_API struct llama_model *
llama_load_model_from_file_capi(const char *path_model,
                                struct llama_context_params *params);

LLAMA_API struct llama_context *
llama_new_context_with_model_capi(struct llama_model *model,
                                  struct llama_context_params *params);

LLAMA_API void llama_get_timings_capi(struct llama_context *ctx,
                                      struct llama_timings *timings);

LLAMA_API void
llama_context_default_params_capi(struct llama_context_params *params);

LLAMA_API void llama_model_quantize_default_params_capi(
    struct llama_model_quantize_params *params);

#ifdef __cplusplus
}
#endif

#endif
