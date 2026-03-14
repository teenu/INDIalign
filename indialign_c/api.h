#ifndef INDIALIGN_API_H
#define INDIALIGN_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int backend_mode;
    int use_fragment_search;
    int max_iter;
    int dp_iter;
    double max_mem_gb;
    int request_cuda;
} NativeConfig;

typedef struct {
    int length;
    const double *pred_xyz;
    const double *native_xyz;
    const uint8_t *valid_mask;
    const uint8_t *pred_valid_mask;
    const uint8_t *native_valid_mask;
    double lnorm;
} NativePairInput;

typedef struct {
    double score;
    double rotation[9];
    double translation[3];
} NativePairResult;

void indialign_default_config(NativeConfig *cfg);
int  indialign_tmscore_search(const NativePairInput *inp,
                              const NativeConfig *cfg,
                              NativePairResult *out);
int  indialign_tmscore_search_batch(int n_pairs,
                                     const NativePairInput *inputs,
                                     const NativeConfig *cfg,
                                     NativePairResult *outputs);
int  indialign_cuda_available(void);

#ifdef __cplusplus
}
#endif

#endif
