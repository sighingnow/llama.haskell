module AI.LanguageModel.LLaMA.FFI where

import Data.Int
import Foreign.C.String
import Foreign.C.Types
import Foreign.Marshal.Array
import Foreign.Ptr
import Foreign.Storable

{- FOURMOLU_DISABLE -}

#include "llama.h"
#include "llama-capi.h"

{- FOURMOLU_ENABLE -}

-- | The language model in llama.cpp.
newtype LLM = LLM {runLLM :: Ptr ()} deriving (Eq, Storable)

-- | The context in llama.cpp.
newtype Context = Context {runContext :: Ptr ()} deriving (Eq, Storable)

type Token = CInt

-- | The log level definition.
data LogLevel = Error | Warn | Info deriving (Eq, Show)

instance Enum LogLevel where
    fromEnum Error = 2
    fromEnum Warn = 3
    fromEnum Info = 4
    toEnum 2 = Error
    toEnum 3 = Warn
    toEnum 4 = Info
    toEnum _ = error "Invalid log level"

instance Storable LogLevel where
    sizeOf _ = sizeOf (undefined :: CInt)
    alignment _ = alignment (undefined :: CInt)
    peek ptr =
        toEnum . fromIntegral
            <$> peek (castPtr ptr :: Ptr CInt)
    poke ptr level = do
        poke (castPtr ptr :: Ptr CInt) (fromIntegral $ fromEnum level)

-- | The vocabulary type definition.
data VocabType
    = -- | SentencePiece
      SPM
    | -- | Byte Pair Encoding
      BPE
    deriving (Eq, Show, Enum)

instance Storable VocabType where
    sizeOf _ = sizeOf (undefined :: CInt)
    alignment _ = alignment (undefined :: CInt)
    peek ptr =
        toEnum . fromIntegral
            <$> peek (castPtr ptr :: Ptr CInt)
    poke ptr level = do
        poke (castPtr ptr :: Ptr CInt) (fromIntegral $ fromEnum level)

-- | The token type definition.
data TokenType = Undefined | Normal | Unknown | Control | UserDefined | Unused | Byte
    deriving (Eq, Show, Enum)

instance Storable TokenType where
    sizeOf _ = sizeOf (undefined :: CInt)
    alignment _ = alignment (undefined :: CInt)
    peek ptr =
        toEnum . fromIntegral
            <$> peek (castPtr ptr :: Ptr CInt)
    poke ptr level = do
        poke (castPtr ptr :: Ptr CInt) (fromIntegral $ fromEnum level)

-- | The model file type definition.
data FloatType
    = AllF32
    | -- | except 1d tensors
      MostlyF16
    | -- | except 1d tensors
      MostlyQ4_0
    | -- | except 1d tensors
      MostlyQ4_1
    | -- | tok_embeddings.weight and output.weight are F16
      MostlyQ4_1SomeF16
    | -- | except 1d tensors
      MostlyQ8_0
    | -- | except 1d tensors
      MostlyQ5_0
    | -- | except 1d tensors
      MostlyQ5_1
    | -- | except 1d tensors
      MostlyQ2K
    | -- | except 1d tensors
      MostlyQ3KS
    | -- | except 1d tensors
      MostlyQ3KM
    | -- | except 1d tensors
      MostlyQ3KL
    | -- | except 1d tensors
      MostlyQ4KS
    | -- | except 1d tensors
      MostlyQ4KM
    | -- | except 1d tensors
      MostlyQ5KS
    | -- | except 1d tensors
      MostlyQ5KM
    | -- | except 1d tensors
      MostlyQ6K
    | -- | not specified in the model file
      Guessed
    deriving (Eq, Show)

instance Enum FloatType where
    fromEnum AllF32 = 0
    fromEnum MostlyF16 = 1
    fromEnum MostlyQ4_0 = 2
    fromEnum MostlyQ4_1 = 3
    fromEnum MostlyQ4_1SomeF16 = 4
    fromEnum MostlyQ8_0 = 7
    fromEnum MostlyQ5_0 = 8
    fromEnum MostlyQ5_1 = 9
    fromEnum MostlyQ2K = 10
    fromEnum MostlyQ3KS = 11
    fromEnum MostlyQ3KM = 12
    fromEnum MostlyQ3KL = 13
    fromEnum MostlyQ4KS = 14
    fromEnum MostlyQ4KM = 15
    fromEnum MostlyQ5KS = 16
    fromEnum MostlyQ5KM = 17
    fromEnum MostlyQ6K = 18
    fromEnum Guessed = 1024

    toEnum 0 = AllF32
    toEnum 1 = MostlyF16
    toEnum 2 = MostlyQ4_0
    toEnum 3 = MostlyQ4_1
    toEnum 4 = MostlyQ4_1SomeF16
    toEnum 7 = MostlyQ8_0
    toEnum 8 = MostlyQ5_0
    toEnum 9 = MostlyQ5_1
    toEnum 10 = MostlyQ2K
    toEnum 11 = MostlyQ3KS
    toEnum 12 = MostlyQ3KM
    toEnum 13 = MostlyQ3KL
    toEnum 14 = MostlyQ4KS
    toEnum 15 = MostlyQ4KM
    toEnum 16 = MostlyQ5KS
    toEnum 17 = MostlyQ5KM
    toEnum 18 = MostlyQ6K
    toEnum 1024 = Guessed
    toEnum _ = error "Invalid model type"

instance Storable FloatType where
    sizeOf _ = sizeOf (undefined :: CInt)
    alignment _ = alignment (undefined :: CInt)
    peek ptr =
        toEnum . fromIntegral
            <$> peek (castPtr ptr :: Ptr CInt)
    poke ptr level = do
        poke (castPtr ptr :: Ptr CInt) (fromIntegral $ fromEnum level)

-- | The token data definition.
data TokenData = TokenData
    { token :: Token
    -- ^ The token ID.
    , logit :: Float
    -- ^ The log-odds of the token.
    , probability :: Float
    -- ^ The probability of the token.
    }
    deriving (Eq, Show)

instance Storable TokenData where
    sizeOf _ = (# size llama_token_data)
    alignment _ = alignment (undefined :: CInt)
    peek ptr =
        TokenData
            <$> (# peek llama_token_data, id) ptr
            <*> (# peek llama_token_data, logit) ptr
            <*> (# peek llama_token_data, p) ptr
    poke ptr (TokenData token score typ) = do
        (# poke llama_token_data, id) ptr token
        (# poke llama_token_data, logit) ptr score
        (# poke llama_token_data, p) ptr typ

data TokenDataArray = TokenDataArray
    { dataPtr :: Ptr TokenData
    , size :: CSize
    , sorted :: Bool
    }

{- FOURMOLU_DISABLE -}

instance Storable TokenDataArray where
    sizeOf _ = (#size llama_token_data_array)
    alignment _ = alignment (undefined :: Ptr TokenData)
    peek ptr =
        TokenDataArray
            <$> (#peek llama_token_data_array, data) ptr
            <*> (#peek llama_token_data_array, size) ptr
            <*> (#peek llama_token_data_array, sorted) ptr
    poke ptr (TokenDataArray dataPtr size sorted) = do
        (#poke llama_token_data_array, data) ptr dataPtr
        (#poke llama_token_data_array, size) ptr size
        (#poke llama_token_data_array, sorted) ptr sorted

{- FOURMOLU_ENABLE -}

-- | The progress callback function.
type ProgressCallback = FunPtr (Float -> Ptr () -> IO ())

-- | The context parameters.
data ContextParameters = ContextParameters
    { seed :: Int32
    -- ^ The random seed.
    , nCtx :: Int32
    -- ^ The text context.
    , nBatch :: Int32
    -- ^ The prompt processing batch size.
    , nGpuLayers :: Int32
    -- ^ The number of layers to store in VRAM.
    , mainGpu :: Int32
    -- ^ The GPU that is used for scratch and small tensors.
    , tensorSplit :: [Float]
    -- ^ How to split layers across multiple GPUs.
    , ropeFreqBase :: Float
    -- ^ RoPE base frequency.
    , ropeFreqScale :: Float
    -- ^ RoPE frequency scaling factor.
    , progressCallback :: ProgressCallback
    -- ^ The progress callback function.
    , progressCallbackUserData :: Ptr ()
    -- ^ The context pointer passed to the progress callback.
    , lowVram :: Bool
    -- ^ If true, reduce VRAM usage at the cost of performance.
    , mulMatQ :: Bool
    -- ^ If true, use experimental mul_mat_q kernels.
    , f16Kv :: Bool
    -- ^ Use fp16 for KV cache.
    , logitsAll :: Bool
    -- ^ The llama_eval() call computes all logits, not just the last one.
    , vocabOnly :: Bool
    -- ^ Only load the vocabulary, no weights.
    , useMmap :: Bool
    -- ^ Use mmap if possible.
    , useMlock :: Bool
    -- ^ Force system to keep model in RAM.
    , embedding :: Bool
    -- ^ Embedding mode only.
    }

{- FOURMOLU_DISABLE -}

instance Storable ContextParameters where
    sizeOf _ = (#size struct llama_context_params)
    alignment _ = alignment (undefined :: CInt)
    peek ptr =
        ContextParameters
            <$> (#peek struct llama_context_params, seed) ptr
            <*> (#peek struct llama_context_params, n_ctx) ptr
            <*> (#peek struct llama_context_params, n_batch) ptr
            <*> (#peek struct llama_context_params, n_gpu_layers) ptr
            <*> (#peek struct llama_context_params, main_gpu) ptr
            <*> peekArray (#const LLAMA_MAX_DEVICES) (castPtr ptr)
            <*> (#peek struct llama_context_params, rope_freq_base) ptr
            <*> (#peek struct llama_context_params, rope_freq_scale) ptr
            <*> (#peek struct llama_context_params, progress_callback) ptr
            <*> (#peek struct llama_context_params, progress_callback_user_data) ptr
            <*> (#peek struct llama_context_params, low_vram) ptr
            <*> (#peek struct llama_context_params, mul_mat_q) ptr
            <*> (#peek struct llama_context_params, f16_kv) ptr
            <*> (#peek struct llama_context_params, logits_all) ptr
            <*> (#peek struct llama_context_params, vocab_only) ptr
            <*> (#peek struct llama_context_params, use_mmap) ptr
            <*> (#peek struct llama_context_params, use_mlock) ptr
            <*> (#peek struct llama_context_params, embedding) ptr
    poke ptr (ContextParameters seed nCtx nBatch nGpuLayers mainGpu tensorSplit repoFreqBase ropeFreqScale progressCallback progressCallbackUserData lowVram mulMatQ f16Kv logitsAll vocabOnly useMmap useMlock embedding) = do
        (#poke struct llama_context_params, seed) ptr seed
        (#poke struct llama_context_params, n_ctx) ptr nCtx
        (#poke struct llama_context_params, n_batch) ptr nBatch
        (#poke struct llama_context_params, n_gpu_layers) ptr nGpuLayers
        (#poke struct llama_context_params, main_gpu) ptr mainGpu
        pokeArray (castPtr ptr) tensorSplit
        (#poke struct llama_context_params, rope_freq_base) ptr repoFreqBase
        (#poke struct llama_context_params, rope_freq_scale) ptr ropeFreqScale
        (#poke struct llama_context_params, progress_callback) ptr progressCallback
        (#poke struct llama_context_params, progress_callback_user_data) ptr progressCallbackUserData
        (#poke struct llama_context_params, low_vram) ptr lowVram
        (#poke struct llama_context_params, mul_mat_q) ptr mulMatQ
        (#poke struct llama_context_params, f16_kv) ptr f16Kv
        (#poke struct llama_context_params, logits_all) ptr logitsAll
        (#poke struct llama_context_params, vocab_only) ptr vocabOnly
        (#poke struct llama_context_params, use_mmap) ptr useMmap
        (#poke struct llama_context_params, use_mlock) ptr useMlock
        (#poke struct llama_context_params, embedding) ptr embedding

{- FOURMOLU_ENABLE -}

-- | The callback for logging events.
type LogCallback = FunPtr (LogLevel -> CString -> Ptr () -> IO ())

-- | The model quantization parameters.
data QuantizeParameters = QuantizeParameters
    { nThread :: Int32
    -- ^ The number of threads to use for quantizing, if <=0 will use std::thread::hardware_concurrency().
    , ftype :: FloatType
    -- ^ Quantize to this llama_ftype.
    , allowRequantize :: Bool
    -- ^ Allow quantizing non-f32/f16 tensors.
    , quantizeOutputTensor :: Bool
    -- ^ Quantize output.weight.
    }

{- FOURMOLU_DISABLE -}

instance Storable QuantizeParameters where
    sizeOf _ = (#size struct llama_model_quantize_params)
    alignment _ = alignment (undefined :: CInt)
    peek ptr =
        QuantizeParameters
            <$> (#peek struct llama_model_quantize_params, nthread) ptr
            <*> (#peek struct llama_model_quantize_params, ftype) ptr
            <*> (#peek struct llama_model_quantize_params, allow_requantize) ptr
            <*> (#peek struct llama_model_quantize_params, quantize_output_tensor) ptr
    poke ptr (QuantizeParameters nThread ftype allowRequantize quantizeOutputTensor) = do
        (#poke struct llama_model_quantize_params, nthread) ptr nThread
        (#poke struct llama_model_quantize_params, ftype) ptr ftype
        (#poke struct llama_model_quantize_params, allow_requantize) ptr allowRequantize
        (#poke struct llama_model_quantize_params, quantize_output_tensor) ptr quantizeOutputTensor

{- FOURMOLU_ENABLE -}

newtype Grammar = Grammar (Ptr ()) deriving (Eq, Storable)

-- | The grammar element type
data GrammarType
    = -- | end of rule definition
      End
    | -- | start of alternate definition for rule
      Alt
    | -- | non-terminal element: reference to rule
      RuleRef
    | -- | terminal element: character (code point)
      Char
    | -- | inverse char(s) ([^a], [^a-b] [^abc])
      CharNot
    | -- | modifies a preceding LLAMA_GRETYPE_CHAR or LLAMA_GRETYPE_CHAR_ALT to
      -- | be an inclusive range ([a-z])
      CharRngUpper
    | -- | modifies a preceding LLAMA_GRETYPE_CHAR or
      -- | LLAMA_GRETYPE_CHAR_RNG_UPPER to add an alternate char to match ([ab], [a-zA])
      CharAlt
    deriving (Eq, Show, Enum)

instance Storable GrammarType where
    sizeOf _ = sizeOf (undefined :: CInt)
    alignment _ = alignment (undefined :: CInt)
    peek ptr =
        toEnum . fromIntegral
            <$> peek (castPtr ptr :: Ptr CInt)
    poke ptr level = do
        poke (castPtr ptr :: Ptr CInt) (fromIntegral $ fromEnum level)

data GrammarElement = GrammarElement
    { typ :: GrammarType
    , value :: CUInt
    -- ^ Unicode code point or rule ID
    }
    deriving (Eq, Show)

{- FOURMOLU_DISABLE -}

instance Storable GrammarElement where
  sizeOf _ = (#size llama_grammar_element)
  alignment _ = alignment (undefined :: CInt)
  peek ptr =
    GrammarElement
      <$> (#peek llama_grammar_element, type) ptr
      <*> (#peek llama_grammar_element, value) ptr
  poke ptr (GrammarElement typ value) = do
    (#poke llama_grammar_element, type) ptr typ
    (#poke llama_grammar_element, value) ptr value

{- FOURMOLU_ENABLE -}

-- | Performance timing information.
data Timings = Timings
    { tStartMs :: Double
    , tEndMs :: Double
    , tLoadMs :: Double
    , tSampleMs :: Double
    , tPEvalMs :: Double
    , tEvalMs :: Double
    , nSample :: Int32
    , nPEval :: Int32
    , nEval :: Int32
    }
    deriving (Eq, Show)

{- FOURMOLU_DISABLE -}

instance Storable Timings where
  sizeOf _ = (#size struct llama_timings)
  alignment _ = alignment (undefined :: CInt)
  peek ptr =
    Timings
      <$> (#peek struct llama_timings, t_start_ms) ptr
      <*> (#peek struct llama_timings, t_end_ms) ptr
      <*> (#peek struct llama_timings, t_load_ms) ptr
      <*> (#peek struct llama_timings, t_sample_ms) ptr
      <*> (#peek struct llama_timings, t_p_eval_ms) ptr
      <*> (#peek struct llama_timings, t_eval_ms) ptr
      <*> (#peek struct llama_timings, n_sample) ptr
      <*> (#peek struct llama_timings, n_p_eval) ptr
      <*> (#peek struct llama_timings, n_eval) ptr
  poke ptr (Timings tStartMs tEndMs tLoadMs tSampleMs tPEvalMs tEvalMs nSample nPEval nEval) = do
    (#poke struct llama_timings, t_start_ms) ptr tStartMs
    (#poke struct llama_timings, t_end_ms) ptr tEndMs
    (#poke struct llama_timings, t_load_ms) ptr tLoadMs
    (#poke struct llama_timings, t_sample_ms) ptr tSampleMs
    (#poke struct llama_timings, t_p_eval_ms) ptr tPEvalMs
    (#poke struct llama_timings, t_eval_ms) ptr tEvalMs
    (#poke struct llama_timings, n_sample) ptr nSample
    (#poke struct llama_timings, n_p_eval) ptr nPEval
    (#poke struct llama_timings, n_eval) ptr nEval

{- FOURMOLU_ENABLE -}

foreign import ccall unsafe "llama-capi.h llama_context_default_params_capi"
    llama_context_default_params :: Ptr ContextParameters -> IO ()

foreign import ccall unsafe "llama-capi.h llama_model_quantize_default_params_capi"
    llama_model_quantize_default_params :: Ptr QuantizeParameters -> IO ()

-- | Initialize the llama + ggml backend. Call once at the start of the program.
foreign import ccall unsafe "llama.h llama_backend_init"
    llama_backend_init
        :: Bool
        -- ^ If true, use NUMA optimizations.
        -> IO ()

-- | Call once at the end of the program - currently only used for MPI.
foreign import ccall unsafe "llama.h llama_backend_free"
    llama_backend_free :: IO ()

-- | Load a model from a file.
foreign import ccall unsafe "llama-capi.h llama_load_model_from_file_capi"
    llama_load_model_from_file
        :: CString
        -- ^ The path to the model file.
        -> Ptr ContextParameters
        -- ^ The context parameters.
        -> IO LLM
        -- ^ The loaded model.

-- | Free the model.
foreign import ccall unsafe "llama.h llama_free_model"
    llama_free_model :: LLM -> IO ()

foreign import ccall unsafe "llama-capi.h llama_new_context_with_model_capi"
    llama_new_context_with_model
        :: LLM
        -> Ptr ContextParameters
        -> IO Context

-- | Free all allocated memory.
foreign import ccall unsafe "llama.h llama_free"
    llama_free :: Context -> IO ()

-- | Timing information in milliseconds.
foreign import ccall unsafe "llama.h llama_time_us"
    llama_time_us :: Context -> IO CLong

foreign import ccall unsafe "llama.h llama_max_devices"
    llama_max_devices :: IO CInt

foreign import ccall unsafe "llama.h llama_mmap_supported"
    llama_mmap_supported :: IO Bool

foreign import ccall unsafe "llama.h llama_mlock_supported"
    llama_mlock_supported :: IO Bool

foreign import ccall unsafe "llama.h llama_n_vocab"
    llama_n_vocab :: Context -> IO CInt

foreign import ccall unsafe "llama.h llama_n_ctx"
    llama_n_ctx :: Context -> IO CInt

foreign import ccall unsafe "llama.h llama_n_embd"
    llama_n_embd :: Context -> IO CInt

foreign import ccall unsafe "llama.h llama_vocab_type"
    llama_vocab_type :: Context -> IO CInt

foreign import ccall unsafe "llama.h llama_model_n_vocab"
    llama_model_n_vocab :: LLM -> IO CInt

foreign import ccall unsafe "llama.h llama_model_n_ctx"
    llama_model_n_ctx :: LLM -> IO CInt

foreign import ccall unsafe "llama.h llama_model_n_embd"
    llama_model_n_embd :: LLM -> IO CInt

-- | Get a string describing the model type.
foreign import ccall unsafe "llama.h llama_model_desc"
    llama_model_desc :: LLM -> CString -> CSize -> IO CInt

-- | Returns the total size of all the tensors in the model in bytes.
foreign import ccall unsafe "llama.h llama_model_size"
    llama_model_size :: LLM -> IO CSize

-- | Returns the total number of parameters in the model.
foreign import ccall unsafe "llama.h llama_model_n_params"
    llama_model_n_params :: LLM -> IO CSize

-- | Returns 0 on quantize success.
foreign import ccall unsafe "llama.h llama_model_quantize"
    llama_model_quantize :: CString -> CString -> Ptr QuantizeParameters -> IO CInt

-- | Apply a LoRA adapter to a loaded model.
--
-- path_base_model is the path to a higher quality model to use as a base for
-- the layers modified by the adapter. Can be NULL to use the current loaded model.
--
-- The model needs to be reloaded before applying a new adapter, otherwise the adapter
-- will be applied on top of the previous one.
--
-- Returns 0 on success
foreign import ccall unsafe "llama.h llama_model_apply_lora_from_file"
    llama_model_apply_lora_from_file :: LLM -> CString -> CString -> CInt -> IO CInt

-- | Returns the number of tokens in the KV cache
foreign import ccall unsafe "llama.h llama_get_kv_cache_token_count"
    llama_get_kv_cache_token_count :: Context -> IO CInt

-- | Sets the current rng seed.
foreign import ccall unsafe "llama.h llama_set_rng_seed"
    llama_set_rng_seed :: Context -> CUInt -> IO ()

-- | Returns the maximum size in bytes of the state (rng, logits, embedding
-- and kv_cache) - will often be smaller after compacting tokens.
foreign import ccall unsafe "llama.h llama_get_state_size"
    llama_get_state_size :: Context -> IO CSize

-- | Copies the state to the specified destination address.
--
-- Destination needs to have allocated enough memory.
--
-- Returns the number of bytes copied
foreign import ccall unsafe "llama.h llama_copy_state_data"
    llama_copy_state_data :: Context -> Ptr CUInt -> IO CSize

-- | Set the state reading from the specified address.
--
-- Returns the number of bytes read
foreign import ccall unsafe "llama.h llama_set_state_data"
    llama_set_state_data :: Context -> Ptr CUInt -> IO CSize

foreign import ccall unsafe "llama.h llama_load_session_file"
    llama_load_session_file
        :: Context -> CString -> Ptr Token -> CSize -> Ptr CSize -> IO Bool

foreign import ccall unsafe "llama.h llama_save_session_file"
    llama_save_session_file :: Context -> CString -> Ptr Token -> CSize -> IO Bool

-- | Run the llama inference to obtain the logits and probabilities for the next token.
--
--  - tokens + n_tokens is the provided batch of new tokens to process
--  - n_past is the number of tokens to use from previous eval calls
--
-- Returns 0 on success
foreign import ccall unsafe "llama.h llama_eval"
    llama_eval :: Context -> Ptr Token -> CInt -> CInt -> CInt -> IO CInt

-- | Same as llama_eval, but use float matrix input directly.
foreign import ccall unsafe "llama.h llama_eval_embd"
    llama_eval_embd :: Context -> Ptr Float -> CInt -> CInt -> CInt -> IO CInt

-- | Export a static computation graph for context of 511 and batch size of 1.
foreign import ccall unsafe "llama.h llama_eval_export"
    llama_eval_export :: Context -> CString -> IO CInt

-- | Token logits obtained from the last call to llama_eval().
-- The logits for the last token are stored in the last row,
-- can be mutated in order to change the probabilities of the next token.
--
--  - Rows: n_tokens
--  - Cols: n_vocab
foreign import ccall unsafe "llama.h llama_get_logits"
    llama_get_logits :: Context -> IO (Ptr Float)

-- | Get the embeddings for the input.
--
--  - shape: [n_embd] (1-dimensional)
foreign import ccall unsafe "llama.h llama_get_embeddings"
    llama_get_embeddings :: Context -> IO (Ptr Float)

foreign import ccall unsafe "llama.h llama_token_get_text"
    llama_token_get_text :: Context -> Token -> IO CString

foreign import ccall unsafe "llama.h llama_token_get_score"
    llama_token_get_score :: Context -> Token -> IO Float

foreign import ccall unsafe "llama.h llama_token_get_type"
    llama_token_get_type :: Context -> Token -> IO CInt

-- | Beginning of sentence token.
foreign import ccall unsafe "llama.h llama_token_bos"
    llama_token_bos :: Context -> IO Token

-- | End of sentence token.
foreign import ccall unsafe "llama.h llama_token_eos"
    llama_token_eos :: Context -> IO Token

-- | Next line token.
foreign import ccall unsafe "llama.h llama_token_nl"
    llama_token_nl :: Context -> IO Token

-- | Convert the provided text into tokens.
--
-- The tokens pointer must be large enough to hold the resulting tokens.
--
--  - Returns the number of tokens on success, no more than n_max_tokens
--  - Returns a negative number on failure - the number of tokens that would have been returned
foreign import ccall unsafe "llama.h llama_tokenize"
    llama_tokenize :: Context -> CString -> Ptr Token -> CInt -> Bool -> IO CInt

foreign import ccall unsafe "llama.h llama_tokenize_with_model"
    llama_tokenize_with_model
        :: LLM -> CString -> Ptr Token -> CInt -> Bool -> IO CInt

-- | Token Id -> Piece.
foreign import ccall unsafe "llama.h llama_token_to_piece"
    llama_token_to_piece :: Context -> Token -> CString -> CInt -> IO CInt

foreign import ccall unsafe "llama.h llama_token_to_piece_with_model"
    llama_token_to_piece_with_model :: LLM -> Token -> CString -> CInt -> IO CInt

foreign import ccall unsafe "llama.h llama_grammar_init"
    llama_grammar_init :: Ptr (Ptr GrammarElement) -> CSize -> CSize -> IO Grammar

foreign import ccall unsafe "llama.h llama_grammar_free"
    llama_grammar_free :: Grammar -> IO ()

-- | @details Repetition penalty described in CTRL academic paper https://arxiv.org/abs/1909.05858, with negative logit fix.
foreign import ccall unsafe "llama.h llama_sample_repetition_penalty"
    llama_sample_repetition_penalty
        :: Context -> Ptr TokenDataArray -> Ptr Token -> CSize -> Float -> IO ()

-- | @details Frequency and presence penalties described in OpenAI API https://platform.openai.com/docs/api-reference/parameter-details.
foreign import ccall unsafe "llama.h llama_sample_frequency_and_presence_penalties"
    llama_sample_frequency_and_presence_penalties
        :: Context -> Ptr TokenDataArray -> Ptr Token -> CSize -> Float -> Float -> IO ()

-- | @details Apply classifier-free guidance to the logits as described in academic paper "Stay on topic with Classifier-Free Guidance" https://arxiv.org/abs/2306.17806
--
--  - @param candidates A vector of `llama_token_data` containing the candidate tokens, the logits must be directly extracted from the original generation context without being sorted.
--  - @params guidance_ctx A separate context from the same model. Other than a negative prompt at the beginning, it should have all generated and user input tokens copied from the main context.
--  - @params scale Guidance strength. 1.0f means no guidance. Higher values mean stronger guidance.
foreign import ccall unsafe "llama.h llama_sample_classifier_free_guidance"
    llama_sample_classifier_free_guidance
        :: Context -> Ptr TokenDataArray -> Context -> Float -> IO ()

-- | @details Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits.
foreign import ccall unsafe "llama.h llama_sample_softmax"
    llama_sample_softmax :: Context -> Ptr TokenDataArray -> IO ()

-- | @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
foreign import ccall unsafe "llama.h llama_sample_top_k"
    llama_sample_top_k :: Context -> Ptr TokenDataArray -> CInt -> CSize -> IO ()

-- | @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
foreign import ccall unsafe "llama.h llama_sample_top_p"
    llama_sample_top_p :: Context -> Ptr TokenDataArray -> Float -> CSize -> IO ()

-- | @details Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/.
foreign import ccall unsafe "llama.h llama_sample_tail_free"
    llama_sample_tail_free
        :: Context -> Ptr TokenDataArray -> Float -> CSize -> IO ()

-- | @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
foreign import ccall unsafe "llama.h llama_sample_typical"
    llama_sample_typical
        :: Context -> Ptr TokenDataArray -> Float -> CSize -> IO ()

foreign import ccall unsafe "llama.h llama_sample_temperature"
    llama_sample_temperature :: Context -> Ptr TokenDataArray -> Float -> IO ()

-- | @details Apply constraints from grammar
foreign import ccall unsafe "llama.h llama_sample_grammar"
    llama_sample_grammar :: Context -> Ptr TokenDataArray -> Ptr Grammar -> IO ()

-- | @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
--
--  - @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
--  - @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
--  - @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
--  - @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
--  - @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
foreign import ccall unsafe "llama.h llama_sample_token_mirostat"
    llama_sample_token_mirostat
        :: Context -> Ptr TokenDataArray -> Float -> Float -> CInt -> Ptr Float -> IO Token

-- | @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
--
--  - @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
--  - @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
--  - @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
--  - @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
foreign import ccall unsafe "llama.h llama_sample_token_mirostat_v2"
    llama_sample_token_mirostat_v2
        :: Context -> Ptr TokenDataArray -> Float -> Float -> Ptr Float -> IO Token

-- | @details Selects the token with the highest probability.
foreign import ccall unsafe "llama.h llama_sample_token_greedy"
    llama_sample_token_greedy :: Context -> Ptr TokenDataArray -> IO Token

-- | @details Randomly selects a token from the candidates based on their probabilities.
foreign import ccall unsafe "llama.h llama_sample_token"
    llama_sample_token :: Context -> Ptr TokenDataArray -> IO Token

-- | @details Accepts the sampled token into the grammar
foreign import ccall unsafe "llama.h llama_grammar_accept_token"
    llama_grammar_accept_token :: Context -> Grammar -> Token -> IO ()

data BeamView = BeamView
    { tokens :: Ptr Token
    , nTokens :: CSize
    , p :: Float
    -- ^ Cumulative beam probability (renormalized relative to all beams)
    , eob :: Bool
    -- ^ Callback should set this to true when a beam is at end-of-beam.
    }
    deriving (Eq, Show)

{- FOURMOLU_DISABLE -}

instance Storable BeamView where
  sizeOf _ = (#size struct llama_beam_view)
  alignment _ = alignment (undefined :: Ptr Token)
  peek ptr =
    BeamView
      <$> (#peek struct llama_beam_view, tokens) ptr
      <*> (#peek struct llama_beam_view, n_tokens) ptr
      <*> (#peek struct llama_beam_view, p) ptr
      <*> (#peek struct llama_beam_view, eob) ptr
  poke ptr (BeamView tokens nTokens p eob) = do
    (#poke struct llama_beam_view, tokens) ptr tokens
    (#poke struct llama_beam_view, n_tokens) ptr nTokens
    (#poke struct llama_beam_view, p) ptr p
    (#poke struct llama_beam_view, eob) ptr eob

{- FOURMOLU_ENABLE -}

data BeamsState = BeamsState
    { beamViews :: Ptr BeamView
    , nBeams :: CSize
    -- ^ Number of elements in beam_views[].
    , commonPrefixLength :: CSize
    -- ^ Current max length of prefix tokens shared by all beams.
    , lastCall :: Bool
    -- ^ True iff this is the last callback invocation.
    }
    deriving (Eq, Show)

{- FOURMOLU_DISABLE -}

instance Storable BeamsState where
  sizeOf _ = (#size struct llama_beams_state)
  alignment _ = alignment (undefined :: Ptr BeamView)
  peek ptr =
    BeamsState
      <$> (#peek struct llama_beams_state, beam_views) ptr
      <*> (#peek struct llama_beams_state, n_beams) ptr
      <*> (#peek struct llama_beams_state, common_prefix_length) ptr
      <*> (#peek struct llama_beams_state, last_call) ptr
  poke ptr (BeamsState beamViews nBeams commonPrefixLength lastCall) = do
    (#poke struct llama_beams_state, beam_views) ptr beamViews
    (#poke struct llama_beams_state, n_beams) ptr nBeams
    (#poke struct llama_beams_state, common_prefix_length) ptr commonPrefixLength
    (#poke struct llama_beams_state, last_call) ptr lastCall

{- FOURMOLU_ENABLE -}

--     // Type of pointer to the beam_search_callback function.
--     // void* callback_data is any custom data passed to llama_beam_search, that is subsequently
--     // passed back to beam_search_callback. This avoids having to use global variables in the callback.
--     typedef void (*llama_beam_search_callback_fn_t)(void * callback_data, struct llama_beams_state);

-- | @details Deterministically returns entire sentence constructed by a beam search.
--
--  - @param ctx Pointer to the llama_context.
--  - @param callback Invoked for each iteration of the beam_search loop, passing in beams_state.
--  - @param callback_data A pointer that is simply passed back to callback.
--  - @param n_beams Number of beams to use.
--  - @param n_past Number of tokens already evaluated.
--  - @param n_predict Maximum number of tokens to predict. EOS may occur earlier.
--  - @param n_threads Number of threads as passed to llama_eval().
foreign import ccall unsafe "llama.h llama_beam_search"
    llama_beam_search
        :: Context
        -> FunPtr (Ptr BeamsState -> IO ())
        -> Ptr ()
        -> CSize
        -> CInt
        -> CInt
        -> CInt
        -> IO ()

foreign import ccall unsafe "llama-capi.h llama_get_timings_capi"
    llama_get_timings :: Context -> Ptr Timings -> IO ()

foreign import ccall unsafe "llama.h llama_print_timings"
    llama_print_timings :: Context -> IO ()

foreign import ccall unsafe "llama.h llama_reset_timings"
    llama_reset_timings :: Context -> IO ()

-- | Print system information.
foreign import ccall unsafe "llama.h llama_print_system_info"
    llama_print_system_info :: IO CString

-- | Set callback for all future logging events.
--
-- If this is not called, or NULL is supplied, everything is output on stderr.
foreign import ccall unsafe "llama.h llama_log_set"
    llama_log_set :: FunPtr LogCallback -> Ptr () -> IO ()

foreign import ccall unsafe "llama.h llama_dump_timing_info_yaml"
    llama_dump_timing_info_yaml :: Ptr CFile -> Context -> IO ()
