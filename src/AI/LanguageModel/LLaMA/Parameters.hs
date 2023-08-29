module AI.LanguageModel.LLaMA.Parameters where

data Parameters = Parameters
    { seed :: Int
    -- ^ RNG seed
    , nThreads :: Int
    -- ^ number of threads
    , nPredict :: Int
    -- ^ new tokens to predict
    , nCtx :: Int
    -- ^ context size
    , nBatch :: Int
    -- ^ batch size for prompt processing (must be >=32 to use BLAS)
    , nKeep :: Int
    -- ^ number of tokens to keep from initial prompt
    , nChunks :: Int
    -- ^ max number of chunks to process (-1 = unlimited)
    , nGpuLayers :: Int
    -- ^ number of layers to store in VRAM
    , mainGpu :: Int
    -- ^ the GPU that is used for scratch and small tensors
    , tensorSplit :: [Float]
    -- ^ how split tensors should be distributed across GPUs
    , nProbs :: Int
    -- ^ if greater than 0, output the probabilities of top n_probs tokens.
    , nBeams :: Int
    -- ^ if non-zero then use beam search of given width.
    , ropeFreqBase :: Float
    -- ^ RoPE base frequency
    , ropeFreqScale :: Float
    -- ^ RoPE frequency scaling factor
    , topK :: Int
    -- ^ <= 0 to use vocab size
    , topP :: Float
    -- ^ 1.0 = disabled
    , tfsZ :: Float
    -- ^ 1.0 = disabled
    , typicalP :: Float
    -- ^ 1.0 = disabled
    , temperature :: Float
    -- ^ 1.0 = disabled
    , repeatPenalty :: Float
    -- ^ 1.0 = disabled
    , repeatLastN :: Int
    -- ^ last n tokens to penalize (0 = disable penalty, -1 = context size)
    , frequencyPenalty :: Float
    -- ^ 0.0 = disabled
    , presencePenalty :: Float
    -- ^ 0.0 = disabled
    , mirostat :: Int
    -- ^ 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    , mirostatTau :: Float
    -- ^ target entropy
    , mirostatEta :: Float
    -- ^ learning rate
    , logitBias :: [(Int, Float)]
    -- ^ logit bias for specific tokens
    , cfgNegativePrompt :: String
    -- ^ string to help guidance
    , cfgScale :: Float
    -- ^ How strong is guidance
    , model :: String
    -- ^ model path
    , modelAlias :: String
    -- ^ model alias
    , prompt :: String
    -- ^ prompt
    , pathPromptCache :: String
    -- ^ path to file for saving/loading prompt eval state
    , inputPrefix :: String
    -- ^ string to prefix user inputs with
    , inputSuffix :: String
    -- ^ string to suffix user inputs with
    , grammar :: String
    -- ^ optional BNF-like grammar to constrain sampling
    , antiprompt :: [String]
    -- ^ string upon seeing which more user input is prompted
    , logdir :: String
    -- ^ directory in which to save YAML log files
    , loraAdaptor :: String
    -- ^ lora adaptor path
    , loraBase :: String
    -- ^ base model path for the lora adaptor
    , pplStride :: Int
    -- ^ stride for perplexity calculations. If left at 0, the pre-existing approach will be used.
    , pplOutputType :: Int
    -- ^ = 0 -> ppl output is as usual, = 1 -> ppl output is num_tokens, ppl, one per line
    -- (which is more convenient to use for plotting)
    , hellaswag :: Bool
    -- ^ compute HellaSwag score over random tasks from datafile supplied in prompt
    , hellaswagTasks :: Int
    -- ^ number of tasks to use when computing the HellaSwag score
    , lowVram :: Bool
    -- ^ if true, reduce VRAM usage at the cost of performance
    , mulMatQ :: Bool
    -- ^ if true, use mul_mat_q kernels instead of cuBLAS
    , memoryF16 :: Bool
    -- ^ use f16 instead of f32 for memory kv
    , randomPrompt :: Bool
    -- ^ do not randomize prompt if none provided
    , useColor :: Bool
    -- ^ use color to distinguish generations and inputs
    , interactive :: Bool
    -- ^ interactive mode
    , promptCacheAll :: Bool
    -- ^ save user input and generations to prompt cache
    , promptCacheRo :: Bool
    -- ^ open the prompt cache read-only and do not update it
    , embedding :: Bool
    -- ^ get only sentence embedding
    , escape :: Bool
    -- ^ escape "\n", "\r", "\t", "\'", "\"", and "\\"
    , interactiveFirst :: Bool
    -- ^ wait for user input immediately
    , multilineInput :: Bool
    -- ^ reverse the usage of `\`
    , simpleIo :: Bool
    -- ^ improves compatibility with subprocesses and limited consoles
    , inputPrefixBos :: Bool
    -- ^ prefix BOS to user inputs, preceding input_prefix
    , ignoreEos :: Bool
    -- ^ ignore generated EOS tokens
    , instruct :: Bool
    -- ^ instruction mode (used for Alpaca models)
    , penalizeNl :: Bool
    -- ^ consider newlines as a repeatable token
    , perplexity :: Bool
    -- ^ compute perplexity over the prompt
    , useMmap :: Bool
    -- ^ use mmap for faster loads
    , useMlock :: Bool
    -- ^ use mlock to keep model in memory
    , memTest :: Bool
    -- ^ compute maximum memory usage
    , numa :: Bool
    -- ^ attempt optimizations that help on some NUMA systems
    , exportCgraph :: Bool
    -- ^ export the computation graph
    , verbosePrompt :: Bool
    -- ^ print prompt tokens before generation
    }
    deriving (Show, Eq)
