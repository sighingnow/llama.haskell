#include "llama-capi.h"

LLAMA_API struct llama_model *
llama_load_model_from_file_capi(const char *path_model,
                                struct llama_context_params *params) {
  return llama_load_model_from_file(path_model, *params);
}

LLAMA_API struct llama_context *
llama_new_context_with_model_capi(struct llama_model *model,
                                  struct llama_context_params *params) {
  return llama_new_context_with_model(model, *params);
}

LLAMA_API void llama_get_timings_capi(struct llama_context *ctx,
                                      struct llama_timings *timings) {
  *timings = llama_get_timings(ctx);
}

LLAMA_API void
llama_context_default_params_capi(struct llama_context_params *params) {
  *params = llama_context_default_params();
}

LLAMA_API void llama_model_quantize_default_params_capi(
    struct llama_model_quantize_params *params) {
  *params = llama_model_quantize_default_params();
}
