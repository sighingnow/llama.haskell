module AI.LanguageModel.GGML.FFI where

import Data.Int
import Foreign.C.String
import Foreign.C.Types
import Foreign.Ptr
import Foreign.Storable

{- FOURMOLU_DISABLE -}

#include "ggml.h"

{- FOURMOLU_ENABLE -}

type Float16 = CUShort

-- | convert FP16 -> FP32.
foreign import ccall unsafe "ggml_fp16_to_fp32"
    ggml_fp16_to_fp32 :: Float16 -> Float

-- | convert FP16 <- FP32.
foreign import ccall unsafe "ggml_fp32_to_fp16"
    ggml_fp32_to_fp16 :: Float -> Float16

foreign import ccall unsafe "ggml_fp16_to_fp32_row"
    ggml_fp16_to_fp32_row :: Ptr Float16 -> Ptr Float -> CInt -> IO ()

foreign import ccall unsafe "ggml_fp32_to_fp16_row"
    ggml_fp32_to_fp16_row :: Ptr Float -> Ptr Float16 -> CInt -> IO ()

data GGMLType
    = GGMLF32
    | GGMLF16
    | GGMLQ4_0
    | GGMLQ4_1
    | GGMLQ5_0
    | GGMLQ5_1
    | GGMLQ8_0
    | GGMLQ8_1
    | GGMLQ2_K
    | GGMLQ3_K
    | GGMLQ4_K
    | GGMLQ5_K
    | GGMLQ6_K
    | GGMLQ8_K
    | GGMLI8
    | GGMLI16
    | GGMLI32
    deriving (Eq, Show)

instance Enum GGMLType where
    fromEnum GGMLF32 = 0
    fromEnum GGMLF16 = 1
    fromEnum GGMLQ4_0 = 2
    fromEnum GGMLQ4_1 = 3
    fromEnum GGMLQ5_0 = 6
    fromEnum GGMLQ5_1 = 7
    fromEnum GGMLQ8_0 = 8
    fromEnum GGMLQ8_1 = 9
    fromEnum GGMLQ2_K = 10
    fromEnum GGMLQ3_K = 11
    fromEnum GGMLQ4_K = 12
    fromEnum GGMLQ5_K = 13
    fromEnum GGMLQ6_K = 14
    fromEnum GGMLQ8_K = 15
    fromEnum GGMLI8 = 16
    fromEnum GGMLI16 = 17
    fromEnum GGMLI32 = 18

    toEnum 0 = GGMLF32
    toEnum 1 = GGMLF16
    toEnum 2 = GGMLQ4_0
    toEnum 3 = GGMLQ4_1
    toEnum 6 = GGMLQ5_0
    toEnum 7 = GGMLQ5_1
    toEnum 8 = GGMLQ8_0
    toEnum 9 = GGMLQ8_1
    toEnum 10 = GGMLQ2_K
    toEnum 11 = GGMLQ3_K
    toEnum 12 = GGMLQ4_K
    toEnum 13 = GGMLQ5_K
    toEnum 14 = GGMLQ6_K
    toEnum 15 = GGMLQ8_K
    toEnum 16 = GGMLI8
    toEnum 17 = GGMLI16
    toEnum 18 = GGMLI32
    toEnum _ = error "toEnum: invalid GGMLType"

instance Storable GGMLType where
    sizeOf _ = sizeOf (undefined :: CInt)
    alignment _ = alignment (undefined :: CInt)
    peek ptr =
        toEnum . fromIntegral
            <$> peek (castPtr ptr :: Ptr CInt)
    poke ptr level = do
        poke (castPtr ptr :: Ptr CInt) (fromIntegral $ fromEnum level)

data GGMLBackend
    = GGMLBackendCPU
    | GGMLBackendGPU
    | GGMLBackendGPUSplit
    deriving (Eq, Show)

instance Enum GGMLBackend where
    fromEnum GGMLBackendCPU = 0
    fromEnum GGMLBackendGPU = 10
    fromEnum GGMLBackendGPUSplit = 20

    toEnum 0 = GGMLBackendCPU
    toEnum 10 = GGMLBackendGPU
    toEnum 20 = GGMLBackendGPUSplit
    toEnum _ = error "toEnum: invalid GGMLBackend"

instance Storable GGMLBackend where
    sizeOf _ = sizeOf (undefined :: CInt)
    alignment _ = alignment (undefined :: CInt)
    peek ptr =
        toEnum . fromIntegral
            <$> peek (castPtr ptr :: Ptr CInt)
    poke ptr level = do
        poke (castPtr ptr :: Ptr CInt) (fromIntegral $ fromEnum level)

data GGMLFloatType
    = Unknown
    | AllF32
    | MostlyF16
    | MostlyQ4_0
    | MostlyQ4_1
    | MostlyQ4_1SomeF16
    | MostlyQ8_0
    | MostlyQ5_0
    | MostlyQ5_1
    | MostlyQ2_K
    | MostlyQ3_K
    | MostlyQ4_K
    | MostlyQ5_K
    | MostlyQ6_K
    deriving (Eq, Show)

instance Enum GGMLFloatType where
    fromEnum Unknown = -1
    fromEnum AllF32 = 0
    fromEnum MostlyF16 = 1
    fromEnum MostlyQ4_0 = 2
    fromEnum MostlyQ4_1 = 3
    fromEnum MostlyQ4_1SomeF16 = 4
    fromEnum MostlyQ8_0 = 7
    fromEnum MostlyQ5_0 = 8
    fromEnum MostlyQ5_1 = 9
    fromEnum MostlyQ2_K = 10
    fromEnum MostlyQ3_K = 11
    fromEnum MostlyQ4_K = 12
    fromEnum MostlyQ5_K = 13
    fromEnum MostlyQ6_K = 14

    toEnum (-1) = Unknown
    toEnum 0 = AllF32
    toEnum 1 = MostlyF16
    toEnum 2 = MostlyQ4_0
    toEnum 3 = MostlyQ4_1
    toEnum 4 = MostlyQ4_1SomeF16
    toEnum 7 = MostlyQ8_0
    toEnum 8 = MostlyQ5_0
    toEnum 9 = MostlyQ5_1
    toEnum 10 = MostlyQ2_K
    toEnum 11 = MostlyQ3_K
    toEnum 12 = MostlyQ4_K
    toEnum 13 = MostlyQ5_K
    toEnum 14 = MostlyQ6_K
    toEnum _ = error "toEnum: invalid GGMLFloatType"

instance Storable GGMLFloatType where
    sizeOf _ = sizeOf (undefined :: CInt)
    alignment _ = alignment (undefined :: CInt)
    peek ptr =
        toEnum . fromIntegral
            <$> peek (castPtr ptr :: Ptr CInt)
    poke ptr level = do
        poke (castPtr ptr :: Ptr CInt) (fromIntegral $ fromEnum level)

-- | Available tensor operations.
data GGMLOP
    = None
    | Dup
    | Add
    | Add1
    | Acc
    | Sub
    | Mul
    | Div
    | Sqr
    | Sqrt
    | Log
    | Sum
    | SumRows
    | Mean
    | Argmax
    | Repeat
    | RepeatBack
    | Concat
    | SiluBack
    | Norm
    | RmsNorm
    | RmsNormBack
    | GroupNorm
    | MulMat
    | OutProd
    | Scale
    | Set
    | Cpy
    | Cont
    | Reshape
    | View
    | Permute
    | Transpose
    | GetRows
    | GetRowsBack
    | Diag
    | DiagMaskInf
    | DiagMaskZero
    | SoftMax
    | SoftMaxBack
    | Rope
    | RopeBack
    | Alibi
    | Clamp
    | Conv1D
    | Conv2D
    | ConvTranspose2D
    | Pool1D
    | Pool2D
    | Upscale
    | FlashAttn
    | FlashFF
    | FlashAttnBack
    | WinPart
    | WinUnpart
    | GetRelPos
    | AddRelPos
    | Unary
    | MapUnary
    | MapBinary
    | MapCustom1F32
    | MapCustom2F32
    | MapCustom3F32
    | MapCustom1
    | MapCustom2
    | MapCustom3
    | CrossEntropyLoss
    | CrossEntropyLossBack
    deriving (Eq, Show, Enum)

instance Storable GGMLOP where
    sizeOf _ = sizeOf (undefined :: CInt)
    alignment _ = alignment (undefined :: CInt)
    peek ptr =
        toEnum . fromIntegral
            <$> peek (castPtr ptr :: Ptr CInt)
    poke ptr level = do
        poke (castPtr ptr :: Ptr CInt) (fromIntegral $ fromEnum level)

data GGMLUnaryOp
    = Abs
    | Sgn
    | Neg
    | Step
    | Tanh
    | Elu
    | Relu
    | Gelu
    | GeluQuick
    | Silu
    deriving (Eq, Show, Enum)

instance Storable GGMLUnaryOp where
    sizeOf _ = sizeOf (undefined :: CInt)
    alignment _ = alignment (undefined :: CInt)
    peek ptr =
        toEnum . fromIntegral
            <$> peek (castPtr ptr :: Ptr CInt)
    poke ptr level = do
        poke (castPtr ptr :: Ptr CInt) (fromIntegral $ fromEnum level)

data GGMLObjectType = GGMLObjectTensor | GGMLObjectGraph | GGMLObjectWorkBuffer
    deriving (Eq, Show, Enum)

instance Storable GGMLObjectType where
    sizeOf _ = sizeOf (undefined :: CInt)
    alignment _ = alignment (undefined :: CInt)
    peek ptr =
        toEnum . fromIntegral
            <$> peek (castPtr ptr :: Ptr CInt)
    poke ptr level = do
        poke (castPtr ptr :: Ptr CInt) (fromIntegral $ fromEnum level)

data GGMLObject = GGMLObject
    { ggmlObjectOffs :: CSize
    , ggmlObjectSize :: CSize
    , ggmlObjectNext :: Ptr GGMLObject
    , ggmlObjectType :: GGMLObjectType
    , ggmlObjectPadding :: CInt
    }
    deriving (Eq, Show)

{- FOURMOLU_DISABLE -}

instance Storable GGMLObject where
    sizeOf _ = (#size struct ggml_object)
    alignment _ = alignment (undefined :: CSize)
    peek ptr = do
        GGMLObject
            <$> (#peek struct ggml_object, offs) ptr
            <*> (#peek struct ggml_object, size) ptr
            <*> (#peek struct ggml_object, next) ptr
            <*> (#peek struct ggml_object, type) ptr
            <*> (#peek struct ggml_object, padding) ptr
    poke ptr (GGMLObject offs size next typ padding) = do
        (#poke struct ggml_object, offs) ptr offs
        (#poke struct ggml_object, size) ptr size
        (#poke struct ggml_object, next) ptr next
        (#poke struct ggml_object, type) ptr typ
        (#poke struct ggml_object, padding) ptr padding

{- FOURMOLU_ENABLE -}

newtype GGMLContext = GGMLContext (Ptr ()) deriving (Eq, Storable)

newtype GGMLTensor = GGMLTensor (Ptr ()) deriving (Eq, Storable)

-- // n-dimensional tensor
-- struct ggml_tensor {
--     enum ggml_type    type;
--     enum ggml_backend backend;

--     int     n_dims;
--     int64_t ne[GGML_MAX_DIMS]; // number of elements
--     size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
--                                // nb[0] = sizeof(type)
--                                // nb[1] = nb[0]   * ne[0] + padding
--                                // nb[i] = nb[i-1] * ne[i-1]

--     // compute data
--     enum ggml_op op;

--     // op params - allocated as int32_t for alignment
--     int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];

--     bool is_param;

--     struct ggml_tensor * grad;
--     struct ggml_tensor * src[GGML_MAX_SRC];

--     // performance
--     int     perf_runs;
--     int64_t perf_cycles;
--     int64_t perf_time_us;

--     void * data;

--     char name[GGML_MAX_NAME];

--     void * extra; // extra things e.g. for ggml-cuda.cu

--     char padding[4];
-- };

newtype GGMLPlan = GGMLPlan (Ptr ()) deriving (Eq, Storable)

-- // the compute plan that needs to be prepared for ggml_graph_compute()
-- // since https://github.com/ggerganov/ggml/issues/287
-- struct ggml_cplan {
--     size_t    work_size; // size of work buffer, calculated by `ggml_graph_plan()`
--     uint8_t * work_data; // work buffer, to be allocated by caller before calling to `ggml_graph_compute()`

--     int n_threads;

--     // the `n_tasks` of nodes, 1:1 mapping to cgraph nodes
--     int n_tasks[GGML_MAX_NODES];

--     // abort ggml_graph_compute when true
--     bool (*abort_callback)(void * data);
--     void * abort_callback_data;
-- };

newtype GGMLGraph = GGMLGraph (Ptr ()) deriving (Eq, Storable)

-- // computation graph
-- struct ggml_cgraph {
--     int n_nodes;
--     int n_leafs;

--     struct ggml_tensor * nodes[GGML_MAX_NODES];
--     struct ggml_tensor * grads[GGML_MAX_NODES];
--     struct ggml_tensor * leafs[GGML_MAX_NODES];

--     void * visited_hash_table[GGML_GRAPH_HASHTABLE_SIZE];

--     // performance
--     int     perf_runs;
--     int64_t perf_cycles;
--     int64_t perf_time_us;
-- };

data GGMLScratch = GGMLScratch
    { ggmlScratchOffs :: CSize
    , ggmlScratchSize :: CSize
    , ggmlScratchData :: Ptr ()
    }
    deriving (Eq, Show)

{- FOURMOLU_DISABLE -}

instance Storable GGMLScratch where
    sizeOf _ = (#size struct ggml_scratch)
    alignment _ = alignment (undefined :: CSize)
    peek ptr = do
        GGMLScratch
            <$> (#peek struct ggml_scratch, offs) ptr
            <*> (#peek struct ggml_scratch, size) ptr
            <*> (#peek struct ggml_scratch, data) ptr
    poke ptr (GGMLScratch offs size data_) = do
        (#poke struct ggml_scratch, offs) ptr offs
        (#poke struct ggml_scratch, size) ptr size
        (#poke struct ggml_scratch, data) ptr data_

{- FOURMOLU_ENABLE -}

data GGMLInitParameters = GGMLInitParameters
    { ggmlInitParametersMemSize :: CSize
    -- ^ bytes
    , ggmlInitParametersMemBuffer :: Ptr ()
    -- ^ if NULL, memory will be allocated internally
    , ggmlInitParametersNoAlloc :: Bool
    -- ^ don't allocate memory for the tensor data
    }
    deriving (Eq, Show)

{- FOURMOLU_DISABLE -}

instance Storable GGMLInitParameters where
    sizeOf _ = (#size struct ggml_init_params)
    alignment _ = alignment (undefined :: CSize)
    peek ptr = do
        GGMLInitParameters
            <$> (#peek struct ggml_init_params, mem_size) ptr
            <*> (#peek struct ggml_init_params, mem_buffer) ptr
            <*> (#peek struct ggml_init_params, no_alloc) ptr
    poke ptr (GGMLInitParameters memSize memBuffer noAlloc) = do
        (#poke struct ggml_init_params, mem_size) ptr memSize
        (#poke struct ggml_init_params, mem_buffer) ptr memBuffer
        (#poke struct ggml_init_params, no_alloc) ptr noAlloc

{- FOURMOLU_ENABLE -}

data GGMLTaskType = GGMLTaskInit | GGMLTaskCompute | GGMLTaskFinalize
    deriving (Eq, Show, Enum)

instance Storable GGMLTaskType where
    sizeOf _ = sizeOf (undefined :: CInt)
    alignment _ = alignment (undefined :: CInt)
    peek ptr =
        toEnum . fromIntegral
            <$> peek (castPtr ptr :: Ptr CInt)
    poke ptr level = do
        poke (castPtr ptr :: Ptr CInt) (fromIntegral $ fromEnum level)

data GGMLComputeParameters = GGMLComputeParameters
    { ggmlComputeParametersType :: GGMLTaskType
    , ggmlComputeParametersIth :: CInt
    , ggmlComputeParametersNth :: CInt
    , ggmlComputeParametersWSize :: CSize
    , ggmlComputeParametersWData :: Ptr ()
    }
    deriving (Eq, Show)

{- FOURMOLU_DISABLE -}

instance Storable GGMLComputeParameters where
    sizeOf _ = (#size struct ggml_compute_params)
    alignment _ = alignment (undefined :: CSize)
    peek ptr = do
        GGMLComputeParameters
            <$> (#peek struct ggml_compute_params, type) ptr
            <*> (#peek struct ggml_compute_params, ith) ptr
            <*> (#peek struct ggml_compute_params, nth) ptr
            <*> (#peek struct ggml_compute_params, wsize) ptr
            <*> (#peek struct ggml_compute_params, wdata) ptr
    poke ptr (GGMLComputeParameters type_ ith nth wsize wdata) = do
        (#poke struct ggml_compute_params, type) ptr type_
        (#poke struct ggml_compute_params, ith) ptr ith
        (#poke struct ggml_compute_params, nth) ptr nth
        (#poke struct ggml_compute_params, wsize) ptr wsize
        (#poke struct ggml_compute_params, wdata) ptr wdata

{- FOURMOLU_ENABLE -}

foreign import ccall unsafe "ggml.h ggml_time_init"
    ggml_time_init :: IO ()

foreign import ccall unsafe "ggml.h ggml_time_ms"
    ggml_time_ms :: IO CLong

foreign import ccall unsafe "ggml.h ggml_time_us"
    ggml_time_us :: IO CLong

foreign import ccall unsafe "ggml.h ggml_cycles"
    ggml_cycles :: IO CLong

foreign import ccall unsafe "ggml.h ggml_cycles_per_ms"
    ggml_cycles_per_ms :: IO CLong

foreign import ccall unsafe "ggml.h ggml_numa_init"
    ggml_numa_init :: IO ()

foreign import ccall unsafe "ggml.h ggml_is_numa"
    ggml_is_numa :: IO Bool

foreign import ccall unsafe "ggml.h ggml_print_object"
    ggml_print_object :: Ptr GGMLObject -> IO ()

foreign import ccall unsafe "ggml.h ggml_print_objects"
    ggml_print_objects :: GGMLContext -> IO ()

foreign import ccall unsafe "ggml.h ggml_nelements"
    ggml_nelements :: GGMLTensor -> IO Int64

foreign import ccall unsafe "ggml.h ggml_nrows"
    ggml_nrows :: GGMLTensor -> IO Int64

foreign import ccall unsafe "ggml.h ggml_nbytes"
    ggml_nbytes :: GGMLTensor -> IO CSize

foreign import ccall unsafe "ggml.h ggml_nbytes_pad"
    ggml_nbytes_pad :: GGMLTensor -> IO CSize

foreign import ccall unsafe "ggml.h ggml_nbytes_split"
    ggml_nbytes_split :: GGMLTensor -> CInt -> IO CSize

foreign import ccall unsafe "ggml.h ggml_blck_size"
    ggml_blck_size :: CInt -> IO Int

foreign import ccall unsafe "ggml.h ggml_type_size"
    ggml_type_size :: CInt -> IO CSize

foreign import ccall unsafe "ggml.h ggml_type_sizef"
    ggml_type_sizef :: CInt -> IO Float

foreign import ccall unsafe "ggml.h ggml_type_name"
    ggml_type_name :: CInt -> IO CString

foreign import ccall unsafe "ggml.h ggml_op_name"
    ggml_op_name :: CInt -> IO CString

foreign import ccall unsafe "ggml.h ggml_op_symbol"
    ggml_op_symbol :: CInt -> IO CString

foreign import ccall unsafe "ggml.h ggml_element_size"
    ggml_element_size :: GGMLTensor -> IO CSize

foreign import ccall unsafe "ggml.h ggml_is_quantized"
    ggml_is_quantized :: CInt -> IO Bool

foreign import ccall unsafe "ggml.h ggml_ftype_to_ggml_type"
    ggml_ftype_to_ggml_type :: CInt -> IO CInt

foreign import ccall unsafe "ggml.h ggml_is_transposed"
    ggml_is_transposed :: GGMLTensor -> IO Bool

foreign import ccall unsafe "ggml.h ggml_is_contiguous"
    ggml_is_contiguous :: GGMLTensor -> IO Bool

foreign import ccall unsafe "ggml.h ggml_is_permuted"
    ggml_is_permuted :: GGMLTensor -> IO Bool

foreign import ccall unsafe "ggml.h ggml_are_same_shape"
    ggml_are_same_shape :: GGMLTensor -> GGMLTensor -> IO Bool

foreign import ccall unsafe "ggml.h ggml_tensor_overhead"
    ggml_tensor_overhead :: IO CSize

-- GGML_API struct ggml_context * ggml_init(struct ggml_init_params params);

foreign import ccall unsafe "ggml.h ggml_free"
    ggml_free :: GGMLContext -> IO ()

foreign import ccall unsafe "ggml.h ggml_used_mem"
    ggml_used_mem :: GGMLContext -> IO CSize

-- GGML_API size_t  ggml_set_scratch (struct ggml_context * ctx, struct ggml_scratch scratch);

foreign import ccall unsafe "ggml.h ggml_get_no_alloc"
    ggml_get_no_alloc :: GGMLContext -> IO Bool

foreign import ccall unsafe "ggml.h ggml_set_no_alloc"
    ggml_set_no_alloc :: GGMLContext -> Bool -> IO ()

foreign import ccall unsafe "ggml.h ggml_get_mem_buffer"
    ggml_get_mem_buffer :: GGMLContext -> IO (Ptr ())

foreign import ccall unsafe "ggml.h ggml_get_mem_size"
    ggml_get_mem_size :: GGMLContext -> IO CSize

foreign import ccall unsafe "ggml.h ggml_get_max_tensor_size"
    ggml_get_max_tensor_size :: GGMLContext -> IO CSize

foreign import ccall unsafe "ggml.h ggml_new_tensor"
    ggml_new_tensor :: GGMLContext -> CInt -> CInt -> Ptr CLong -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_new_tensor_1d"
    ggml_new_tensor_1d :: GGMLContext -> CInt -> CLong -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_new_tensor_2d"
    ggml_new_tensor_2d :: GGMLContext -> CInt -> CLong -> CLong -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_new_tensor_3d"
    ggml_new_tensor_3d
        :: GGMLContext -> CInt -> CLong -> CLong -> CLong -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_new_tensor_4d"
    ggml_new_tensor_4d
        :: GGMLContext -> CInt -> CLong -> CLong -> CLong -> CLong -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_new_i32"
    ggml_new_i32 :: GGMLContext -> CInt -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_new_f32"
    ggml_new_f32 :: GGMLContext -> Float -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_dup_tensor"
    ggml_dup_tensor :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_view_tensor"
    ggml_view_tensor :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_get_tensor"
    ggml_get_tensor :: GGMLContext -> CString -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_set_zero"
    ggml_set_zero :: GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_set_i32"
    ggml_set_i32 :: GGMLTensor -> CInt -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_set_f32"
    ggml_set_f32 :: GGMLTensor -> Float -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_get_i32_1d"
    ggml_get_i32_1d :: GGMLTensor -> CInt -> IO CInt

foreign import ccall unsafe "ggml.h ggml_set_i32_1d"
    ggml_set_i32_1d :: GGMLTensor -> CInt -> CInt -> IO ()

foreign import ccall unsafe "ggml.h ggml_get_f32_1d"
    ggml_get_f32_1d :: GGMLTensor -> CInt -> IO Float

foreign import ccall unsafe "ggml.h ggml_set_f32_1d"
    ggml_set_f32_1d :: GGMLTensor -> CInt -> Float -> IO ()

foreign import ccall unsafe "ggml.h ggml_get_data"
    ggml_get_data :: GGMLTensor -> IO (Ptr ())

foreign import ccall unsafe "ggml.h ggml_get_data_f32"
    ggml_get_data_f32 :: GGMLTensor -> IO (Ptr Float)

foreign import ccall unsafe "ggml.h ggml_get_unary_op"
    ggml_get_unary_op :: GGMLTensor -> IO CInt

foreign import ccall unsafe "ggml.h ggml_get_name"
    ggml_get_name :: GGMLTensor -> IO CString

foreign import ccall unsafe "ggml.h ggml_set_name"
    ggml_set_name :: GGMLTensor -> CString -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_format_name"
    ggml_format_name :: GGMLTensor -> CString -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_dup"
    ggml_dup :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_dup_inplace"
    ggml_dup_inplace :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_add"
    ggml_add :: GGMLContext -> GGMLTensor -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_add_inplace"
    ggml_add_inplace :: GGMLContext -> GGMLTensor -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_add1"
    ggml_add1 :: GGMLContext -> GGMLTensor -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_add1_inplace"
    ggml_add1_inplace :: GGMLContext -> GGMLTensor -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_acc"
    ggml_acc
        :: GGMLContext
        -> GGMLTensor
        -> GGMLTensor
        -> CSize
        -> CSize
        -> CSize
        -> CSize
        -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_acc_inplace"
    ggml_acc_inplace
        :: GGMLContext
        -> GGMLTensor
        -> GGMLTensor
        -> CSize
        -> CSize
        -> CSize
        -> CSize
        -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_sub"
    ggml_sub :: GGMLContext -> GGMLTensor -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_sub_inplace"
    ggml_sub_inplace :: GGMLContext -> GGMLTensor -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_mul"
    ggml_mul :: GGMLContext -> GGMLTensor -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_mul_inplace"
    ggml_mul_inplace :: GGMLContext -> GGMLTensor -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_div"
    ggml_div :: GGMLContext -> GGMLTensor -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_div_inplace"
    ggml_div_inplace :: GGMLContext -> GGMLTensor -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_sqr"
    ggml_sqr :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_sqr_inplace"
    ggml_sqr_inplace :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_sqrt"
    ggml_sqrt :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_sqrt_inplace"
    ggml_sqrt_inplace :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_log"
    ggml_log :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_log_inplace"
    ggml_log_inplace :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_sum"
    ggml_sum :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_sum_rows"
    ggml_sum_rows :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_mean"
    ggml_mean :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_argmax"
    ggml_argmax :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_repeat"
    ggml_repeat :: GGMLContext -> GGMLTensor -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_repeat_back"
    ggml_repeat_back :: GGMLContext -> GGMLTensor -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_concat"
    ggml_concat :: GGMLContext -> GGMLTensor -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_abs"
    ggml_abs :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_abs_inplace"
    ggml_abs_inplace :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_sgn"
    ggml_sgn :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_sgn_inplace"
    ggml_sgn_inplace :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_neg"
    ggml_neg :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_neg_inplace"
    ggml_neg_inplace :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_step"
    ggml_step :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_step_inplace"
    ggml_step_inplace :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_tanh"
    ggml_tanh :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_tanh_inplace"
    ggml_tanh_inplace :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_elu"
    ggml_elu :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_elu_inplace"
    ggml_elu_inplace :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_relu"
    ggml_relu :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_relu_inplace"
    ggml_relu_inplace :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_gelu"
    ggml_gelu :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_gelu_inplace"
    ggml_gelu_inplace :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_gelu_quick"
    ggml_gelu_quick :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml.h ggml_gelu_quick_inplace"
    ggml_gelu_quick_inplace :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml_silu"
    ggml_silu :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml_silu_inplace"
    ggml_silu_inplace :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml_silu_back"
    ggml_silu_back :: GGMLContext -> GGMLTensor -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml_norm"
    ggml_norm :: GGMLContext -> GGMLTensor -> Float -> IO GGMLTensor

foreign import ccall unsafe "ggml_norm_inplace"
    ggml_norm_inplace :: GGMLContext -> GGMLTensor -> Float -> IO GGMLTensor

foreign import ccall unsafe "ggml_rms_norm"
    ggml_rms_norm :: GGMLContext -> GGMLTensor -> Float -> IO GGMLTensor

foreign import ccall unsafe "ggml_rms_norm_inplace"
    ggml_rms_norm_inplace :: GGMLContext -> GGMLTensor -> Float -> IO GGMLTensor

foreign import ccall unsafe "ggml_group_norm"
    ggml_group_norm :: GGMLContext -> GGMLTensor -> CInt -> IO GGMLTensor

foreign import ccall unsafe "ggml_group_norm_inplace"
    ggml_group_norm_inplace :: GGMLContext -> GGMLTensor -> CInt -> IO GGMLTensor

foreign import ccall unsafe "ggml_rms_norm_back"
    ggml_rms_norm_back
        :: GGMLContext -> GGMLTensor -> GGMLTensor -> Float -> IO GGMLTensor

foreign import ccall unsafe "ggml_mul_mat"
    ggml_mul_mat :: GGMLContext -> GGMLTensor -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml_out_prod"
    ggml_out_prod :: GGMLContext -> GGMLTensor -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml_scale"
    ggml_scale :: GGMLContext -> GGMLTensor -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml_scale_inplace"
    ggml_scale_inplace :: GGMLContext -> GGMLTensor -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml_set"
    ggml_set
        :: GGMLContext
        -> GGMLTensor
        -> GGMLTensor
        -> CSize
        -> CSize
        -> CSize
        -> CSize
        -> IO GGMLTensor

foreign import ccall unsafe "ggml_set_inplace"
    ggml_set_inplace
        :: GGMLContext
        -> GGMLTensor
        -> GGMLTensor
        -> CSize
        -> CSize
        -> CSize
        -> CSize
        -> IO GGMLTensor

foreign import ccall unsafe "ggml_set_1d"
    ggml_set_1d :: GGMLContext -> GGMLTensor -> GGMLTensor -> CSize -> IO GGMLTensor

foreign import ccall unsafe "ggml_set_1d_inplace"
    ggml_set_1d_inplace
        :: GGMLContext -> GGMLTensor -> GGMLTensor -> CSize -> IO GGMLTensor

foreign import ccall unsafe "ggml_set_2d"
    ggml_set_2d
        :: GGMLContext -> GGMLTensor -> GGMLTensor -> CSize -> CSize -> IO GGMLTensor

foreign import ccall unsafe "ggml_set_2d_inplace"
    ggml_set_2d_inplace
        :: GGMLContext -> GGMLTensor -> GGMLTensor -> CSize -> CSize -> IO GGMLTensor

foreign import ccall unsafe "ggml_cpy"
    ggml_cpy :: GGMLContext -> GGMLTensor -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml_cpy_inplace"
    ggml_cpy_inplace :: GGMLContext -> GGMLTensor -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml_cont"
    ggml_cont :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml_cont_inplace"
    ggml_cont_inplace :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml_reshape"
    ggml_reshape :: GGMLContext -> GGMLTensor -> GGMLTensor -> IO GGMLTensor

foreign import ccall unsafe "ggml_reshape_1d"
    ggml_reshape_1d :: GGMLContext -> GGMLTensor -> CLLong -> IO GGMLTensor

foreign import ccall unsafe "ggml_reshape_2d"
    ggml_reshape_2d
        :: GGMLContext -> GGMLTensor -> CLLong -> CLLong -> IO GGMLTensor

foreign import ccall unsafe "ggml_reshape_3d"
    ggml_reshape_3d
        :: GGMLContext -> GGMLTensor -> CLLong -> CLLong -> CLLong -> IO GGMLTensor

foreign import ccall unsafe "ggml_reshape_4d"
    ggml_reshape_4d
        :: GGMLContext
        -> GGMLTensor
        -> CLLong
        -> CLLong
        -> CLLong
        -> CLLong
        -> IO GGMLTensor

foreign import ccall unsafe "ggml_view_1d"
    ggml_view_1d :: GGMLContext -> GGMLTensor -> CLLong -> CSize -> IO GGMLTensor

foreign import ccall unsafe "ggml_view_2d"
    ggml_view_2d
        :: GGMLContext -> GGMLTensor -> CLLong -> CLLong -> CSize -> CSize -> IO GGMLTensor

foreign import ccall unsafe "ggml_view_3d"
    ggml_view_3d
        :: GGMLContext
        -> GGMLTensor
        -> CLLong
        -> CLLong
        -> CLLong
        -> CSize
        -> CSize
        -> CSize
        -> IO GGMLTensor

foreign import ccall unsafe "ggml_view_4d"
    ggml_view_4d
        :: GGMLContext
        -> GGMLTensor
        -> CLLong
        -> CLLong
        -> CLLong
        -> CLLong
        -> CSize
        -> CSize
        -> CSize
        -> CSize
        -> IO GGMLTensor

foreign import ccall unsafe "ggml_permute"
    ggml_permute
        :: GGMLContext -> GGMLTensor -> CInt -> CInt -> CInt -> CInt -> IO GGMLTensor

foreign import ccall unsafe "ggml_transpose"
    ggml_transpose :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall safe "ggml_get_rows"
    ggml_get_rows :: GGMLContext -> GGMLTensor -> GGMLTensor -> IO GGMLTensor

foreign import ccall safe "ggml_get_rows_back"
    ggml_get_rows_back
        :: GGMLContext -> GGMLTensor -> GGMLTensor -> GGMLTensor -> IO GGMLTensor

foreign import ccall safe "ggml_diag"
    ggml_diag :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall safe "ggml_diag_mask_inf"
    ggml_diag_mask_inf :: GGMLContext -> GGMLTensor -> CInt -> IO GGMLTensor

foreign import ccall safe "ggml_diag_mask_inf_inplace"
    ggml_diag_mask_inf_inplace :: GGMLContext -> GGMLTensor -> CInt -> IO GGMLTensor

foreign import ccall safe "ggml_diag_mask_zero"
    ggml_diag_mask_zero :: GGMLContext -> GGMLTensor -> CInt -> IO GGMLTensor

foreign import ccall safe "ggml_diag_mask_zero_inplace"
    ggml_diag_mask_zero_inplace
        :: GGMLContext -> GGMLTensor -> CInt -> IO GGMLTensor

foreign import ccall safe "ggml_soft_max"
    ggml_soft_max :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall safe "ggml_soft_max_inplace"
    ggml_soft_max_inplace :: GGMLContext -> GGMLTensor -> IO GGMLTensor

foreign import ccall safe "ggml_soft_max_back"
    ggml_soft_max_back :: GGMLContext -> GGMLTensor -> GGMLTensor -> IO GGMLTensor

foreign import ccall safe "ggml_soft_max_back_inplace"
    ggml_soft_max_back_inplace
        :: GGMLContext -> GGMLTensor -> GGMLTensor -> IO GGMLTensor

foreign import ccall safe "ggml_rope"
    ggml_rope
        :: GGMLContext -> GGMLTensor -> CInt -> CInt -> CInt -> CInt -> IO GGMLTensor

foreign import ccall safe "ggml_rope_inplace"
    ggml_rope_inplace
        :: GGMLContext -> GGMLTensor -> CInt -> CInt -> CInt -> CInt -> IO GGMLTensor

foreign import ccall safe "ggml_rope_custom"
    ggml_rope_custom
        :: GGMLContext
        -> GGMLTensor
        -> CInt
        -> CInt
        -> CInt
        -> CInt
        -> Float
        -> Float
        -> IO GGMLTensor

foreign import ccall safe "ggml_rope_custom_inplace"
    ggml_rope_custom_inplace
        :: GGMLContext
        -> GGMLTensor
        -> CInt
        -> CInt
        -> CInt
        -> CInt
        -> Float
        -> Float
        -> IO GGMLTensor

foreign import ccall safe "ggml_rope_xpos_inplace"
    ggml_rope_xpos_inplace
        :: GGMLContext -> GGMLTensor -> CInt -> CInt -> Float -> Bool -> IO GGMLTensor

foreign import ccall safe "ggml_rope_back"
    ggml_rope_back
        :: GGMLContext
        -> GGMLTensor
        -> CInt
        -> CInt
        -> CInt
        -> CInt
        -> Float
        -> Float
        -> Float
        -> Bool
        -> IO GGMLTensor

foreign import ccall safe "ggml_alibi"
    ggml_alibi
        :: GGMLContext -> GGMLTensor -> CInt -> CInt -> Float -> IO GGMLTensor

foreign import ccall safe "ggml_clamp"
    ggml_clamp :: GGMLContext -> GGMLTensor -> Float -> Float -> IO GGMLTensor

foreign import ccall safe "ggml_conv_1d"
    ggml_conv_1d
        :: GGMLContext -> GGMLTensor -> GGMLTensor -> CInt -> CInt -> CInt -> IO GGMLTensor

foreign import ccall safe "ggml_conv_1d_ph"
    ggml_conv_1d_ph
        :: GGMLContext -> GGMLTensor -> GGMLTensor -> CInt -> CInt -> IO GGMLTensor

foreign import ccall safe "ggml_conv_2d"
    ggml_conv_2d
        :: GGMLContext
        -> GGMLTensor
        -> GGMLTensor
        -> CInt
        -> CInt
        -> CInt
        -> CInt
        -> CInt
        -> CInt
        -> IO GGMLTensor

foreign import ccall safe "ggml_conv_2d_sk_p0"
    ggml_conv_2d_sk_p0 :: GGMLContext -> GGMLTensor -> GGMLTensor -> IO GGMLTensor

foreign import ccall safe "ggml_conv_2d_s1_ph"
    ggml_conv_2d_s1_ph :: GGMLContext -> GGMLTensor -> GGMLTensor -> IO GGMLTensor

foreign import ccall safe "ggml_conv_transpose_2d_p0"
    ggml_conv_transpose_2d_p0
        :: GGMLContext -> GGMLTensor -> GGMLTensor -> CInt -> IO GGMLTensor

data GGMLPoolOp = GGML_OP_POOL_MAX | GGML_OP_POOL_AVG | GGML_OP_POOL_COUNT
    deriving (Eq, Show, Enum)

instance Storable GGMLPoolOp where
    sizeOf _ = sizeOf (undefined :: CInt)
    alignment _ = alignment (undefined :: CInt)
    peek ptr = toEnum . fromIntegral <$> peek (castPtr ptr :: Ptr CInt)
    poke ptr = poke (castPtr ptr :: Ptr CInt) . fromIntegral . fromEnum

foreign import ccall safe "ggml_pool_1d"
    ggml_pool_1d
        :: GGMLContext -> GGMLTensor -> CInt -> CInt -> CInt -> CInt -> IO GGMLTensor

foreign import ccall safe "ggml_pool_2d"
    ggml_pool_2d
        :: GGMLContext
        -> GGMLTensor
        -> CInt
        -> CInt
        -> CInt
        -> CInt
        -> CInt
        -> CInt
        -> CInt
        -> IO GGMLTensor

foreign import ccall safe "ggml_upscale"
    ggml_upscale :: GGMLContext -> GGMLTensor -> CInt -> IO GGMLTensor

foreign import ccall safe "ggml_flash_attn"
    ggml_flash_attn
        :: GGMLContext -> GGMLTensor -> GGMLTensor -> GGMLTensor -> Bool -> IO GGMLTensor

foreign import ccall safe "ggml_flash_attn_back"
    ggml_flash_attn_back
        :: GGMLContext
        -> GGMLTensor
        -> GGMLTensor
        -> GGMLTensor
        -> GGMLTensor
        -> Bool
        -> IO GGMLTensor

foreign import ccall safe "ggml_flash_ff"
    ggml_flash_ff
        :: GGMLContext
        -> GGMLTensor
        -> GGMLTensor
        -> GGMLTensor
        -> GGMLTensor
        -> GGMLTensor
        -> IO GGMLTensor

foreign import ccall safe "ggml_win_part"
    ggml_win_part :: GGMLContext -> GGMLTensor -> CInt -> IO GGMLTensor

foreign import ccall safe "ggml_win_unpart"
    ggml_win_unpart
        :: GGMLContext -> GGMLTensor -> CInt -> CInt -> CInt -> IO GGMLTensor

foreign import ccall safe "ggml_unary"
    ggml_unary :: GGMLContext -> GGMLTensor -> CInt -> IO GGMLTensor

foreign import ccall safe "ggml_unary_inplace"
    ggml_unary_inplace :: GGMLContext -> GGMLTensor -> CInt -> IO GGMLTensor

foreign import ccall safe "ggml_get_rel_pos"
    ggml_get_rel_pos :: GGMLContext -> GGMLTensor -> CInt -> CInt -> IO GGMLTensor

foreign import ccall safe "ggml_add_rel_pos"
    ggml_add_rel_pos
        :: GGMLContext -> GGMLTensor -> GGMLTensor -> GGMLTensor -> IO GGMLTensor

foreign import ccall safe "ggml_add_rel_pos_inplace"
    ggml_add_rel_pos_inplace
        :: GGMLContext -> GGMLTensor -> GGMLTensor -> GGMLTensor -> IO GGMLTensor

type GGMLCustom1Op = GGMLTensor -> GGMLTensor -> CInt -> CInt -> Ptr () -> IO ()

type GGMLCustom2Op =
    GGMLTensor -> GGMLTensor -> GGMLTensor -> CInt -> CInt -> Ptr () -> IO ()

type GGMLCustom3Op =
    GGMLTensor
    -> GGMLTensor
    -> GGMLTensor
    -> GGMLTensor
    -> CInt
    -> CInt
    -> Ptr ()
    -> IO ()

foreign import ccall safe "ggml_map_custom1"
    ggml_map_custom1
        :: GGMLContext
        -> GGMLTensor
        -> FunPtr GGMLCustom1Op
        -> CInt
        -> Ptr ()
        -> IO GGMLTensor

foreign import ccall safe "ggml_map_custom1_inplace"
    ggml_map_custom1_inplace
        :: GGMLContext
        -> GGMLTensor
        -> FunPtr GGMLCustom1Op
        -> CInt
        -> Ptr ()
        -> IO GGMLTensor

foreign import ccall safe "ggml_map_custom2"
    ggml_map_custom2
        :: GGMLContext
        -> GGMLTensor
        -> GGMLTensor
        -> FunPtr GGMLCustom2Op
        -> CInt
        -> Ptr ()
        -> IO GGMLTensor

foreign import ccall safe "ggml_map_custom2_inplace"
    ggml_map_custom2_inplace
        :: GGMLContext
        -> GGMLTensor
        -> GGMLTensor
        -> FunPtr GGMLCustom2Op
        -> CInt
        -> Ptr ()
        -> IO GGMLTensor

foreign import ccall safe "ggml_map_custom3"
    ggml_map_custom3
        :: GGMLContext
        -> GGMLTensor
        -> GGMLTensor
        -> GGMLTensor
        -> FunPtr GGMLCustom3Op
        -> CInt
        -> Ptr ()
        -> IO GGMLTensor

foreign import ccall safe "ggml_map_custom3_inplace"
    ggml_map_custom3_inplace
        :: GGMLContext
        -> GGMLTensor
        -> GGMLTensor
        -> GGMLTensor
        -> FunPtr GGMLCustom3Op
        -> CInt
        -> Ptr ()
        -> IO GGMLTensor

foreign import ccall safe "ggml_cross_entropy_loss"
    ggml_cross_entropy_loss
        :: GGMLContext -> GGMLTensor -> GGMLTensor -> IO GGMLTensor

foreign import ccall safe "ggml_cross_entropy_loss_back"
    ggml_cross_entropy_loss_back
        :: GGMLContext -> GGMLTensor -> GGMLTensor -> GGMLTensor -> IO GGMLTensor

foreign import ccall safe "ggml_set_param"
    ggml_set_param :: GGMLContext -> GGMLTensor -> IO ()

foreign import ccall safe "ggml_build_forward_expand"
    ggml_build_forward_expand :: Ptr GGMLGraph -> GGMLTensor -> IO ()

foreign import ccall safe "ggml_build_backward_expand"
    ggml_build_backward_expand
        :: GGMLContext -> Ptr GGMLGraph -> Ptr GGMLGraph -> Bool -> IO ()

foreign import ccall safe "ggml_build_forward"
    ggml_build_forward :: GGMLTensor -> IO GGMLGraph

foreign import ccall safe "ggml_build_backward"
    ggml_build_backward :: GGMLContext -> Ptr GGMLGraph -> Bool -> IO GGMLGraph

foreign import ccall safe "ggml_new_graph"
    ggml_new_graph :: GGMLContext -> IO (Ptr GGMLGraph)

foreign import ccall safe "ggml_build_forward_ctx"
    ggml_build_forward_ctx :: GGMLContext -> GGMLTensor -> IO (Ptr GGMLGraph)

foreign import ccall safe "ggml_graph_overhead"
    ggml_graph_overhead :: IO CSize

foreign import ccall safe "ggml_graph_plan"
    ggml_graph_plan :: Ptr GGMLGraph -> CInt -> IO GGMLPlan

foreign import ccall safe "ggml_graph_compute"
    ggml_graph_compute :: Ptr GGMLGraph -> Ptr GGMLPlan -> IO CInt

foreign import ccall safe "ggml_graph_reset"
    ggml_graph_reset :: Ptr GGMLGraph -> IO ()

foreign import ccall safe "ggml_graph_compute_with_ctx"
    ggml_graph_compute_with_ctx :: GGMLContext -> Ptr GGMLGraph -> CInt -> IO ()

foreign import ccall safe "ggml_graph_get_tensor"
    ggml_graph_get_tensor :: Ptr GGMLGraph -> CString -> IO GGMLTensor

foreign import ccall safe "ggml_graph_export"
    ggml_graph_export :: Ptr GGMLGraph -> CString -> IO ()

foreign import ccall safe "ggml_graph_import"
    ggml_graph_import
        :: CString -> Ptr (GGMLContext) -> Ptr (GGMLContext) -> IO GGMLGraph

foreign import ccall safe "ggml_graph_print"
    ggml_graph_print :: Ptr GGMLGraph -> IO ()

foreign import ccall safe "ggml_graph_dump_dot"
    ggml_graph_dump_dot :: Ptr GGMLGraph -> Ptr GGMLGraph -> CString -> IO ()

data GGMLOptType = GGML_OPT_ADAM | GGML_OPT_LBFGS deriving (Eq, Show, Enum)

instance Storable GGMLOptType where
    sizeOf _ = sizeOf (undefined :: CInt)
    alignment _ = alignment (undefined :: CInt)
    peek ptr = toEnum . fromIntegral <$> peek (castPtr ptr :: Ptr CInt)
    poke ptr = poke (castPtr ptr :: Ptr CInt) . fromIntegral . fromEnum

data GGMLLineSearch
    = GGML_LINESEARCH_DEFAULT
    | GGML_LINESEARCH_BACKTRACKING_ARMIJO
    | GGML_LINESEARCH_BACKTRACKING_WOLFE
    | GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE
    deriving (Eq, Show)

instance Enum GGMLLineSearch where
    fromEnum GGML_LINESEARCH_DEFAULT = 1
    fromEnum GGML_LINESEARCH_BACKTRACKING_ARMIJO = 0
    fromEnum GGML_LINESEARCH_BACKTRACKING_WOLFE = 1
    fromEnum GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE = 2

    toEnum 0 = GGML_LINESEARCH_BACKTRACKING_ARMIJO
    toEnum 1 = GGML_LINESEARCH_BACKTRACKING_WOLFE
    toEnum 2 = GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE
    toEnum _ = GGML_LINESEARCH_DEFAULT

instance Storable GGMLLineSearch where
    sizeOf _ = sizeOf (undefined :: CInt)
    alignment _ = alignment (undefined :: CInt)
    peek ptr = toEnum . fromIntegral <$> peek (castPtr ptr :: Ptr CInt)
    poke ptr = poke (castPtr ptr :: Ptr CInt) . fromIntegral . fromEnum

data GGMLOptResult
    = GGML_OPT_OK
    | GGML_OPT_DID_NOT_CONVERGE
    | GGML_OPT_NO_CONTEXT
    | GGML_OPT_INVALID_WOLFE
    | GGML_OPT_FAIL
    | GGML_LINESEARCH_FAIL
    | GGML_LINESEARCH_MINIMUM_STEP
    | GGML_LINESEARCH_MAXIMUM_STEP
    | GGML_LINESEARCH_MAXIMUM_ITERATIONS
    | GGML_LINESEARCH_INVALID_PARAMETERS
    deriving (Eq, Show)

instance Enum GGMLOptResult where
    fromEnum GGML_OPT_OK = 0
    fromEnum GGML_OPT_DID_NOT_CONVERGE = 1
    fromEnum GGML_OPT_NO_CONTEXT = 2
    fromEnum GGML_OPT_INVALID_WOLFE = 3
    fromEnum GGML_OPT_FAIL = 4
    fromEnum GGML_LINESEARCH_FAIL = -128
    fromEnum GGML_LINESEARCH_MINIMUM_STEP = -127
    fromEnum GGML_LINESEARCH_MAXIMUM_STEP = -126
    fromEnum GGML_LINESEARCH_MAXIMUM_ITERATIONS = -125
    fromEnum GGML_LINESEARCH_INVALID_PARAMETERS = -124

    toEnum 0 = GGML_OPT_OK
    toEnum 1 = GGML_OPT_DID_NOT_CONVERGE
    toEnum 2 = GGML_OPT_NO_CONTEXT
    toEnum 3 = GGML_OPT_INVALID_WOLFE
    toEnum 4 = GGML_OPT_FAIL
    toEnum (-128) = GGML_LINESEARCH_FAIL
    toEnum (-127) = GGML_LINESEARCH_MINIMUM_STEP
    toEnum (-126) = GGML_LINESEARCH_MAXIMUM_STEP
    toEnum (-125) = GGML_LINESEARCH_MAXIMUM_ITERATIONS
    toEnum (-124) = GGML_LINESEARCH_INVALID_PARAMETERS
    toEnum _ = GGML_OPT_FAIL

type GGMLOptCallback = FunPtr (Ptr () -> Ptr Float -> IO ())

newtype GGMLOptParameters = GGMLOptParameters (Ptr ()) deriving (Eq, Storable)

-- struct ggml_opt_params {
--     enum ggml_opt_type type;

--     int n_threads;

--     // delta-based convergence test
--     //
--     //   if past == 0 - disabled
--     //   if past > 0:
--     //     stop if |f(x) - f(x_past)| < delta * max(1, |f(x)|)
--     //
--     int past;
--     float delta;

--     // maximum number of iterations without improvement
--     //
--     //   if 0 - disabled
--     //   if > 0:
--     //     assume convergence if no cost improvement in this number of iterations
--     //
--     int max_no_improvement;

--     bool print_forward_graph;
--     bool print_backward_graph;

--     // ADAM parameters
--     struct {
--         int n_iter;

--         float sched; // schedule multiplier (fixed, decay or warmup)
--         float decay; // weight decay for AdamW, use 0.0f to disable
--         int   decay_min_ndim; // minimum number of tensor dimension to apply weight decay
--         float alpha; // learning rate
--         float beta1;
--         float beta2;
--         float eps;   // epsilon for numerical stability
--         float eps_f; // epsilon for convergence test
--         float eps_g; // epsilon for convergence test
--         float gclip; // gradient clipping
--     } adam;

--     // LBFGS parameters
--     struct {
--         int m; // number of corrections to approximate the inv. Hessian
--         int n_iter;
--         int max_linesearch;

--         float eps;      // convergence tolerance
--         float ftol;     // line search tolerance
--         float wolfe;
--         float min_step;
--         float max_step;

--         enum ggml_linesearch linesearch;
--     } lbfgs;
-- };

newtype GGMLOptContext = GGMLOptContext (Ptr ()) deriving (Eq, Storable)

-- struct ggml_opt_context {
--     struct ggml_context * ctx;
--     struct ggml_opt_params params;

--     int iter;
--     int64_t nx; // number of parameter elements

--     bool just_initialized;

--     float loss_before;
--     float loss_after;

--     struct {
--         struct ggml_tensor * m;  // first moment
--         struct ggml_tensor * v;  // second moment
--         struct ggml_tensor * pf; // past function values
--         float fx_best;
--         float fx_prev;
--         int n_no_improvement;
--     } adam;

--     struct {
--         struct ggml_tensor * x;    // current parameters
--         struct ggml_tensor * xp;   // previous parameters
--         struct ggml_tensor * g;    // current gradient
--         struct ggml_tensor * gp;   // previous gradient
--         struct ggml_tensor * d;    // search direction
--         struct ggml_tensor * pf;   // past function values
--         struct ggml_tensor * lmal; // the L-BFGS memory alpha
--         struct ggml_tensor * lmys; // the L-BFGS memory ys
--         struct ggml_tensor * lms;  // the L-BFGS memory s
--         struct ggml_tensor * lmy;  // the L-BFGS memory y
--         float fx_best;
--         float step;
--         int j;
--         int k;
--         int end;
--         int n_no_improvement;
--     } lbfgs;
-- };

-- GGML_API struct ggml_opt_params ggml_opt_default_params(enum ggml_opt_type type);

-- // optimize the function defined by the tensor f
-- GGML_API enum ggml_opt_result ggml_opt(
--         struct ggml_context * ctx,
--         struct ggml_opt_params params,
--         struct ggml_tensor * f);

-- // initialize optimizer context
-- GGML_API void ggml_opt_init(
--         struct ggml_context     * ctx,
--         struct ggml_opt_context * opt,
--         struct ggml_opt_params    params,
--         int64_t                   nx);

foreign import ccall unsafe "ggml.h ggml_opt_resume"
    ggml_opt_resume :: GGMLContext -> Ptr GGMLOptContext -> GGMLTensor -> IO CInt

foreign import ccall unsafe "ggml.h ggml_opt_resume_g"
    ggml_opt_resume_g
        :: GGMLContext
        -> Ptr GGMLOptContext
        -> GGMLTensor
        -> Ptr GGMLGraph
        -> Ptr GGMLGraph
        -> FunPtr GGMLOptCallback
        -> Ptr ()
        -> IO CInt

foreign import ccall unsafe "ggml.h ggml_quantize_q4_0"
    ggml_quantize_q4_0
        :: Ptr Float -> Ptr () -> CInt -> CInt -> Ptr CLong -> IO CSize

foreign import ccall unsafe "ggml.h ggml_quantize_q4_1"
    ggml_quantize_q4_1
        :: Ptr Float -> Ptr () -> CInt -> CInt -> Ptr CLong -> IO CSize

foreign import ccall unsafe "ggml.h ggml_quantize_q5_0"
    ggml_quantize_q5_0
        :: Ptr Float -> Ptr () -> CInt -> CInt -> Ptr CLong -> IO CSize

foreign import ccall unsafe "ggml.h ggml_quantize_q5_1"
    ggml_quantize_q5_1
        :: Ptr Float -> Ptr () -> CInt -> CInt -> Ptr CLong -> IO CSize

foreign import ccall unsafe "ggml.h ggml_quantize_q8_0"
    ggml_quantize_q8_0
        :: Ptr Float -> Ptr () -> CInt -> CInt -> Ptr CLong -> IO CSize

foreign import ccall unsafe "ggml.h ggml_quantize_chunk"
    ggml_quantize_chunk
        :: CInt -> Ptr Float -> Ptr () -> CInt -> CInt -> Ptr CLong -> IO CSize

-- enum gguf_type
data GGUFType
    = GGUFTypeUInt8
    | GGUFTypeInt8
    | GGUFTypeUInt16
    | GGUFTypeInt16
    | GGUFTypeUInt32
    | GGUFTypeInt32
    | GGUFTypeFloat32
    | GGUFTypeBool
    | GGUFTypeString
    | GGUFTypeArray
    | GGUFTypeUInt64
    | GGUFTypeInt64
    | GGUFTypeFloat64
    | GGUFTypeCount
    deriving (Eq, Show, Enum)

instance Storable GGUFType where
    sizeOf _ = sizeOf (undefined :: CInt)
    alignment _ = alignment (undefined :: CInt)
    peek ptr = toEnum . fromIntegral <$> peek (castPtr ptr :: Ptr CInt)
    poke ptr = poke (castPtr ptr :: Ptr CInt) . fromIntegral . fromEnum

newtype GGUFContext = GGUFContext (Ptr ())

data GGUFInitParameters = GGUFInitParameters
    { ggufInitNoAlloc :: Bool
    , ggufInitContext :: GGMLContext
    }

{- FOURMOLU_DISABLE -}

instance Storable GGUFInitParameters where
    sizeOf _ = (#size struct gguf_init_params)
    alignment _ = alignment (undefined :: CInt)
    peek ptr = do
        GGUFInitParameters
            <$> (#peek struct gguf_init_params, no_alloc) ptr
            <*> (#peek struct gguf_init_params, ctx) ptr
    poke ptr (GGUFInitParameters noAlloc ctx) = do
        (#poke struct gguf_init_params, no_alloc) ptr noAlloc
        (#poke struct gguf_init_params, ctx) ptr ctx

{- FOURMOLU_ENABLE -}

foreign import ccall unsafe "gguf_init_empty"
    gguf_init_empty :: IO (Ptr GGUFContext)

-- GGML_API struct gguf_context * gguf_init_from_file(const char * fname, struct gguf_init_params params);

foreign import ccall unsafe "gguf_free"
    gguf_free :: Ptr GGUFContext -> IO ()

foreign import ccall unsafe "gguf_type_name"
    gguf_type_name :: CInt -> IO CString

foreign import ccall unsafe "gguf_get_version"
    gguf_get_version :: Ptr GGUFContext -> IO CInt

foreign import ccall unsafe "gguf_get_alignment"
    gguf_get_alignment :: Ptr GGUFContext -> IO CSize

foreign import ccall unsafe "gguf_get_data_offset"
    gguf_get_data_offset :: Ptr GGUFContext -> IO CSize

foreign import ccall unsafe "gguf_get_data"
    gguf_get_data :: Ptr GGUFContext -> IO (Ptr ())

foreign import ccall unsafe "gguf_get_n_kv"
    gguf_get_n_kv :: Ptr GGUFContext -> IO CInt

foreign import ccall unsafe "gguf_find_key"
    gguf_find_key :: Ptr GGUFContext -> CString -> IO CInt

foreign import ccall unsafe "gguf_get_key"
    gguf_get_key :: Ptr GGUFContext -> CInt -> IO CString

foreign import ccall unsafe "gguf_get_kv_type"
    gguf_get_kv_type :: Ptr GGUFContext -> CInt -> IO CInt

foreign import ccall unsafe "gguf_get_arr_type"
    gguf_get_arr_type :: Ptr GGUFContext -> CInt -> IO CInt

foreign import ccall unsafe "gguf_get_val_u8"
    gguf_get_val_u8 :: Ptr GGUFContext -> CInt -> IO CUChar

foreign import ccall unsafe "gguf_get_val_i8"
    gguf_get_val_i8 :: Ptr GGUFContext -> CInt -> IO CSChar

foreign import ccall unsafe "gguf_get_val_u16"
    gguf_get_val_u16 :: Ptr GGUFContext -> CInt -> IO CUShort

foreign import ccall unsafe "gguf_get_val_i16"
    gguf_get_val_i16 :: Ptr GGUFContext -> CInt -> IO CShort

foreign import ccall unsafe "gguf_get_val_u32"
    gguf_get_val_u32 :: Ptr GGUFContext -> CInt -> IO CUInt

foreign import ccall unsafe "gguf_get_val_i32"
    gguf_get_val_i32 :: Ptr GGUFContext -> CInt -> IO CInt

foreign import ccall unsafe "gguf_get_val_f32"
    gguf_get_val_f32 :: Ptr GGUFContext -> CInt -> IO Float

foreign import ccall unsafe "gguf_get_val_u64"
    gguf_get_val_u64 :: Ptr GGUFContext -> CInt -> IO CULong

foreign import ccall unsafe "gguf_get_val_i64"
    gguf_get_val_i64 :: Ptr GGUFContext -> CInt -> IO CLong

foreign import ccall unsafe "gguf_get_val_f64"
    gguf_get_val_f64 :: Ptr GGUFContext -> CInt -> IO CDouble

foreign import ccall unsafe "gguf_get_val_bool"
    gguf_get_val_bool :: Ptr GGUFContext -> CInt -> IO Bool

foreign import ccall unsafe "gguf_get_val_str"
    gguf_get_val_str :: Ptr GGUFContext -> CInt -> IO CString

foreign import ccall unsafe "gguf_get_arr_n"
    gguf_get_arr_n :: Ptr GGUFContext -> CInt -> IO CInt

foreign import ccall unsafe "gguf_get_arr_data"
    gguf_get_arr_data :: Ptr GGUFContext -> CInt -> IO (Ptr ())

foreign import ccall unsafe "gguf_get_arr_str"
    gguf_get_arr_str :: Ptr GGUFContext -> CInt -> CInt -> IO CString

foreign import ccall unsafe "gguf_get_n_tensors"
    gguf_get_n_tensors :: Ptr GGUFContext -> IO CInt

foreign import ccall unsafe "gguf_find_tensor"
    gguf_find_tensor :: Ptr GGUFContext -> CString -> IO CInt

foreign import ccall unsafe "gguf_get_tensor_offset"
    gguf_get_tensor_offset :: Ptr GGUFContext -> CInt -> IO CSize

foreign import ccall unsafe "gguf_get_tensor_name"
    gguf_get_tensor_name :: Ptr GGUFContext -> CInt -> IO CString

foreign import ccall unsafe "gguf_set_val_u8"
    gguf_set_val_u8 :: Ptr GGUFContext -> CString -> CUChar -> IO ()

foreign import ccall unsafe "gguf_set_val_i8"
    gguf_set_val_i8 :: Ptr GGUFContext -> CString -> CSChar -> IO ()

foreign import ccall unsafe "gguf_set_val_u16"
    gguf_set_val_u16 :: Ptr GGUFContext -> CString -> CUShort -> IO ()

foreign import ccall unsafe "gguf_set_val_i16"
    gguf_set_val_i16 :: Ptr GGUFContext -> CString -> CShort -> IO ()

foreign import ccall unsafe "gguf_set_val_u32"
    gguf_set_val_u32 :: Ptr GGUFContext -> CString -> CUInt -> IO ()

foreign import ccall unsafe "gguf_set_val_i32"
    gguf_set_val_i32 :: Ptr GGUFContext -> CString -> CInt -> IO ()

foreign import ccall unsafe "gguf_set_val_f32"
    gguf_set_val_f32 :: Ptr GGUFContext -> CString -> Float -> IO ()

foreign import ccall unsafe "gguf_set_val_u64"
    gguf_set_val_u64 :: Ptr GGUFContext -> CString -> CULong -> IO ()

foreign import ccall unsafe "gguf_set_val_i64"
    gguf_set_val_i64 :: Ptr GGUFContext -> CString -> CLong -> IO ()

foreign import ccall unsafe "gguf_set_val_f64"
    gguf_set_val_f64 :: Ptr GGUFContext -> CString -> CDouble -> IO ()

foreign import ccall unsafe "gguf_set_val_bool"
    gguf_set_val_bool :: Ptr GGUFContext -> CString -> Bool -> IO ()

foreign import ccall unsafe "gguf_set_val_str"
    gguf_set_val_str :: Ptr GGUFContext -> CString -> CString -> IO ()

foreign import ccall unsafe "gguf_set_arr_data"
    gguf_set_arr_data
        :: Ptr GGUFContext -> CString -> CInt -> Ptr () -> CInt -> IO ()

foreign import ccall unsafe "gguf_set_arr_str"
    gguf_set_arr_str :: Ptr GGUFContext -> CString -> Ptr CString -> CInt -> IO ()

foreign import ccall unsafe "gguf_set_kv"
    gguf_set_kv :: Ptr GGUFContext -> Ptr GGUFContext -> IO ()

foreign import ccall unsafe "gguf_add_tensor"
    gguf_add_tensor :: Ptr GGUFContext -> GGMLTensor -> IO ()

foreign import ccall unsafe "gguf_set_tensor_type"
    gguf_set_tensor_type :: Ptr GGUFContext -> CString -> CInt -> IO ()

foreign import ccall unsafe "gguf_set_tensor_data"
    gguf_set_tensor_data :: Ptr GGUFContext -> CString -> Ptr () -> CSize -> IO ()

foreign import ccall unsafe "gguf_write_to_file"
    gguf_write_to_file :: Ptr GGUFContext -> CString -> Bool -> IO ()

foreign import ccall unsafe "gguf_get_meta_size"
    gguf_get_meta_size :: Ptr GGUFContext -> IO CSize

foreign import ccall unsafe "gguf_get_meta_data"
    gguf_get_meta_data :: Ptr GGUFContext -> Ptr () -> IO ()

foreign import ccall unsafe "ggml_cpu_has_avx"
    ggml_cpu_has_avx :: IO CInt

foreign import ccall unsafe "ggml_cpu_has_avx2"
    ggml_cpu_has_avx2 :: IO CInt

foreign import ccall unsafe "ggml_cpu_has_avx512"
    ggml_cpu_has_avx512 :: IO CInt

foreign import ccall unsafe "ggml_cpu_has_avx512_vbmi"
    ggml_cpu_has_avx512_vbmi :: IO CInt

foreign import ccall unsafe "ggml_cpu_has_avx512_vnni"
    ggml_cpu_has_avx512_vnni :: IO CInt

foreign import ccall unsafe "ggml_cpu_has_fma"
    ggml_cpu_has_fma :: IO CInt

foreign import ccall unsafe "ggml_cpu_has_neon"
    ggml_cpu_has_neon :: IO CInt

foreign import ccall unsafe "ggml_cpu_has_arm_fma"
    ggml_cpu_has_arm_fma :: IO CInt

foreign import ccall unsafe "ggml_cpu_has_f16c"
    ggml_cpu_has_f16c :: IO CInt

foreign import ccall unsafe "ggml_cpu_has_fp16_va"
    ggml_cpu_has_fp16_va :: IO CInt

foreign import ccall unsafe "ggml_cpu_has_wasm_simd"
    ggml_cpu_has_wasm_simd :: IO CInt

foreign import ccall unsafe "ggml_cpu_has_blas"
    ggml_cpu_has_blas :: IO CInt

foreign import ccall unsafe "ggml_cpu_has_cublas"
    ggml_cpu_has_cublas :: IO CInt

foreign import ccall unsafe "ggml_cpu_has_clblast"
    ggml_cpu_has_clblast :: IO CInt

foreign import ccall unsafe "ggml_cpu_has_gpublas"
    ggml_cpu_has_gpublas :: IO CInt

foreign import ccall unsafe "ggml_cpu_has_sse3"
    ggml_cpu_has_sse3 :: IO CInt

foreign import ccall unsafe "ggml_cpu_has_ssse3"
    ggml_cpu_has_ssse3 :: IO CInt

foreign import ccall unsafe "ggml_cpu_has_vsx"
    ggml_cpu_has_vsx :: IO CInt

-- typedef void (*ggml_to_float_t)  (const void  * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
-- typedef void (*ggml_from_float_t)(const float * GGML_RESTRICT x, void  * GGML_RESTRICT y, int k);
-- typedef void (*ggml_vec_dot_t)   (const int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT x, const void * GGML_RESTRICT y);

-- typedef struct {
--     const char      * type_name;
--     int               blck_size;
--     size_t            type_size;
--     bool              is_quantized;
--     ggml_to_float_t   to_float;
--     ggml_from_float_t from_float;
--     ggml_from_float_t from_float_reference;
--     ggml_vec_dot_t    vec_dot;
--     enum ggml_type    vec_dot_type;
-- } ggml_type_traits_t;

-- ggml_type_traits_t ggml_internal_get_type_traits(enum ggml_type type);
