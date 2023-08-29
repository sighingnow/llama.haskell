{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TemplateHaskell #-}

module AI.LanguageModel.LLaMA
    ( LLaMAContext (..)

      -- * initialize LLaMA context
    , initLLaMAContext
    , deinitLLaMAContext
    , tokenize
    , printTimings

      -- * eval monad and lens
    , Eval (..)
    , EvalContext (..)
    , _interacting'
    , _inputEcho'
    , _nTokens'
    , _past'
    , _remain'
    , _consumed'
    , _lastTokens'
    , _embed'
    , prepare
    , runEval

      -- * LLaMA evaluation functions
    , evalOnce

      -- * re-export
    , Parameters (..)
    , ContextParameters
    , LLM (..)
    )
where

import AI.LanguageModel.LLaMA.FFI as FFI
import AI.LanguageModel.LLaMA.Parameters as Parameters
import Control.Monad
import Control.Monad.State
import Foreign.C
import Foreign.Marshal.Alloc
import Foreign.Marshal.Array
import Foreign.Ptr
import Foreign.Storable
import Lens.Micro
import Lens.Micro.Mtl
import Lens.Micro.TH

newtype Tokens = Tokens {runTokens :: (Ptr Token, CInt)}

allocateTokens :: Context -> IO Tokens
allocateTokens context = do
    tapeLength <- llama_n_ctx context
    tape <- mallocArray (fromIntegral tapeLength)
    return $ Tokens (tape, tapeLength)

deallocateTokens :: Tokens -> IO ()
deallocateTokens (Tokens (tape, _)) = free tape

data LLaMAContext = LLaMAContext
    { args :: Parameters
    , llm :: LLM
    , context :: Context
    , contextGuidence :: Maybe Context
    -- ^ context guidence
    , tokens :: Tokens
    , buffer :: Ptr CChar
    }

initLLaMAContext :: Parameters -> IO LLaMAContext
initLLaMAContext args = do
    initBackend False
    contextParameters <- initContextParameters args
    (llm, context) <- initFromFromFile (model args) contextParameters
    (llm', context') <-
        applyLora
            llm
            context
            (loraAdaptor args)
            (loraBase args)
            (nThreads args)
    contextGuidence <-
        if cfgScale args >= 1.0
            then Just <$> initContextFromModel llm' contextParameters
            else return Nothing
    tokens <- allocateTokens context
    buffer <- castPtr <$> mallocBytes 1024

    -- update the nctx parameter
    ctxLength <- llama_n_ctx context
    let args' = args {Parameters.nCtx = fromIntegral ctxLength}
    return $ LLaMAContext {args = args', llm = llm', context = context', ..}

deinitLLaMAContext :: LLaMAContext -> IO ()
deinitLLaMAContext LLaMAContext {..} = do
    free buffer
    deallocateTokens tokens
    llama_free context
    case contextGuidence of
        Nothing -> return ()
        Just ctx -> llama_free ctx
    llama_free_model llm

initBackend :: Bool -> IO ()
initBackend = llama_backend_init

initContextParameters :: Parameters -> IO ContextParameters
initContextParameters Parameters {..} = alloca $ \p -> do
    llama_context_default_params p
    defaultParameters <- peek p
    return $
        defaultParameters
            { FFI.nCtx = fromIntegral nCtx
            , FFI.nBatch = fromIntegral nBatch
            , FFI.nGpuLayers = fromIntegral nGpuLayers
            , FFI.mainGpu = fromIntegral mainGpu
            , FFI.tensorSplit = tensorSplit
            , FFI.lowVram = lowVram
            , FFI.mulMatQ = mulMatQ
            , FFI.seed = fromIntegral seed
            , FFI.f16Kv = memoryF16
            , FFI.useMmap = useMmap
            , FFI.useMlock = useMlock
            , FFI.logitsAll = perplexity
            , FFI.embedding = embedding
            , FFI.ropeFreqBase = ropeFreqBase
            , FFI.ropeFreqScale = ropeFreqScale
            }

initFromFromFile :: FilePath -> ContextParameters -> IO (LLM, Context)
initFromFromFile path args = withCString path $ \path' ->
    alloca $ \p -> do
        poke p args
        llm <- llama_load_model_from_file path' p
        when (runLLM llm == nullPtr) $
            fail $
                "Failed to load model "
                    ++ path
        context <- llama_new_context_with_model llm p
        when (runContext context == nullPtr) $ do
            llama_free_model llm
            fail $ "Failed to create context with model " ++ path
        return (llm, context)

initContextFromModel :: LLM -> ContextParameters -> IO Context
initContextFromModel llm args = alloca $ \p -> do
    poke p args
    context <- llama_new_context_with_model llm p
    when (runContext context == nullPtr) $ do
        llama_free_model llm
        fail "Failed to create context with model"
    return context

applyLora
    :: LLM
    -> Context
    -> String
    -- ^ lora adaptor
    -> String
    -- ^ lora base
    -> Int
    -- ^ n threads
    -> IO (LLM, Context)
applyLora llm context [] _ _ = return (llm, context)
applyLora llm context adaptor base nThreads = do
    withCString adaptor $ \adaptor' -> do
        err <-
            if null base
                then llama_model_apply_lora_from_file llm adaptor' nullPtr (fromIntegral nThreads)
                else withCString base $ \base' ->
                    llama_model_apply_lora_from_file llm adaptor' base' (fromIntegral nThreads)
        when (err /= 0) $ do
            llama_free context
            llama_free_model llm
            fail $ "Failed to apply lora adaptor " ++ adaptor ++ " from base " ++ base
        return (llm, context)

warmUp :: LLaMAContext -> IO ()
warmUp LLaMAContext {..} = alloca $ \p -> do
    bos <- llama_token_bos context
    poke p bos
    void $ llama_eval context p 1 0 (fromIntegral (nThreads args))
    llama_reset_timings context

tokenize
    :: LLaMAContext
    -> String
    -- ^ prompt
    -> IO Int
tokenize LLaMAContext {..} prompt = do
    -- Add BOS if SPM tokenizer
    vocabType <- llama_vocab_type context
    print $ "tokenize: vocabType = " ++ show vocabType ++ ", prompt = " ++ prompt
    let addBOS = fromIntegral vocabType == fromEnum SPM
        promptLength = length prompt
        (tape, tapeLength) = runTokens tokens
    print $
        "tokenize: tapeLength = " ++ show tapeLength ++ ", prompt = " ++ show prompt
    withCString prompt $ \prompt' -> do
        print $ "before tokenize: " ++ show prompt'
        nTokens <- llama_tokenize context prompt' tape (tapeLength - 4) addBOS
        print $ "tokenize: " ++ show nTokens ++ " tokens"
        when (nTokens < 0) $
            fail $
                "error: prompt is too long, maximum number of tokens: " ++ show (tapeLength - 4)
        print $ "tokenize: " ++ show nTokens ++ " tokens"
        when (nTokens == 0) $ do
            bos <- llama_token_bos context
            pokeElemOff tape 0 bos
        return (fromIntegral (max 1 nTokens))

data EvalContext = EvalContext
    { _interacting :: Bool
    , _inputEcho :: Bool
    , _nTokens :: Int
    , _past :: Int
    , _remain :: Int
    , _consumed :: Int
    , _lastTokens :: [Token]
    -- ^ last n tokens
    , _embed :: [Token]
    }
    deriving (Show)

makeLensesWith
    ( lensRulesFor
        [ ("_interacting", "_interacting'")
        , ("_inputEcho", "_inputEcho'")
        , ("_nTokens", "_nTokens'")
        , ("_past", "_past'")
        , ("_remain", "_remain'")
        , ("_consumed", "_consumed'")
        , ("_lastTokens", "_lastTokens'")
        , ("_embed", "_embed'")
        ]
        & simpleLenses .~ True
    )
    ''EvalContext

newtype Eval a = Eval (StateT EvalContext IO a)
    deriving (Functor, Applicative, Monad, MonadIO, MonadState EvalContext)

runEval :: EvalContext -> Eval a -> IO (a, EvalContext)
runEval ctx (Eval m) = runStateT m ctx

prepare :: LLaMAContext -> IO EvalContext
prepare LLaMAContext {..} = do
    return $
        EvalContext
            { _interacting = False
            , _inputEcho = False
            , _nTokens = 0
            , _past = 0
            , _remain = nPredict args
            , _consumed = 0
            , _lastTokens = replicate (Parameters.nCtx args) 0
            , _embed = []
            }

evalOnce
    :: LLaMAContext
    -> Eval ()
evalOnce ctx@LLaMAContext {..} = do
    gets _embed >>= \case
        [] -> return ()
        _ -> predict ctx

    nTokens <- gets _nTokens
    consumed <- gets _consumed
    interacting <- gets _interacting
    if nTokens <= consumed && (not interacting)
        then sample ctx
        else forward ctx

    -- display text
    liftIO $ putStrLn "starts the first round of generation: "
    gets _inputEcho >>= \case
        True -> do
            embed <- gets _embed
            liftIO $ forM_ embed $ \token -> do
                w <- tokenToPiece ctx token
                putStr w
        False -> return ()

    -- finish generation
    nTokens <- gets _nTokens
    consumed <- gets _consumed
    when (nTokens <= consumed) $ do
        processEnd ctx

    -- end of text token
    embed <- gets _embed
    eos <- liftIO $ llama_token_eos context
    when ((length embed > 0) && (last embed == eos)) $
        liftIO $
            putStrLn " [end of text]"

    remain <- gets _remain
    when (interactive args && (remain <= 0) && (nPredict args >= 0)) $ do
        assign _remain' (nPredict args)
        assign _interacting' True

sample
    :: LLaMAContext
    -> Eval ()
sample LLaMAContext {..} = do
    logits <- liftIO $ llama_get_logits context
    nVocab <- liftIO $ fromIntegral <$> llama_n_vocab context

    -- apply params.logit_bias map
    forM_ (logitBias args) $ \(index, bias) -> liftIO $ pokeElemOff logits index bias

    candidates <- liftIO $ mallocArray nVocab

    -- prepare candidates
    forM_ [0 .. nVocab - 1] $ \index -> liftIO $ do
        logit <- peekElemOff logits index
        pokeElemOff candidates index (TokenData (fromIntegral index) logit 0.0)

    candidates' <- liftIO $ malloc
    liftIO $
        poke candidates' (TokenDataArray candidates (fromIntegral nVocab) False)

    case contextGuidence of
        Just guidence ->
            liftIO $
                llama_sample_classifier_free_guidance
                    context
                    candidates'
                    guidence
                    (cfgScale args)
        Nothing -> return ()

    -- apply penalties
    nlToken <- liftIO $ llama_token_nl context
    nlLogits <- liftIO $ peekElemOff logits (fromIntegral nlToken)

    lastTokens <- gets _lastTokens
    let lastTokensLength = length lastTokens
        lastRepeatN =
            if repeatLastN args < 0
                then min lastTokensLength (Parameters.nCtx args)
                else min (min lastTokensLength (repeatLastN args)) (Parameters.nCtx args)

    liftIO $ withArray lastTokens $ \lastTokens' -> do
        let lastTokensData = plusPtr lastTokens' (lastTokensLength - lastRepeatN)
        llama_sample_repetition_penalty
            context
            candidates'
            lastTokensData
            (fromIntegral lastRepeatN)
            (repeatPenalty args)
        llama_sample_frequency_and_presence_penalties
            context
            candidates'
            lastTokensData
            (fromIntegral lastRepeatN)
            (frequencyPenalty args)
            (presencePenalty args)
        unless (penalizeNl args) $ do
            nlToken <- llama_token_nl context
            forM_ [0 .. nVocab - 1] $ \index -> do
                TokenData token logit prob <- peekElemOff candidates index
                when (token == nlToken) $
                    pokeElemOff candidates index (TokenData token nlLogits prob)

    let temp = temperature args
        tau = mirostatTau args
        eta = mirostatEta args
    let sampleGreedy = llama_sample_token_greedy context candidates'
        sampleMirostat = case mirostat args of
            1 -> do
                let mu = 2.0 * tau
                    m = 100
                llama_sample_temperature context candidates' temp
                alloca $ \mu' -> do
                    poke mu' mu
                    llama_sample_token_mirostat context candidates' tau eta m mu'
            2 -> do
                let mu = 2.0 * tau
                llama_sample_temperature context candidates' temp
                alloca $ \mu' -> do
                    poke mu' mu
                    llama_sample_token_mirostat_v2 context candidates' tau eta mu'
            _ -> do
                llama_sample_top_k context candidates' (fromIntegral (topK args)) 1
                llama_sample_tail_free context candidates' (tfsZ args) 1
                llama_sample_typical context candidates' (typicalP args) 1
                llama_sample_top_p context candidates' (topP args) 1
                llama_sample_temperature context candidates' temp
                llama_sample_token context candidates'
    token <-
        liftIO $
            if temp <= 0
                then sampleGreedy
                else sampleMirostat

    lastTokens <- gets _lastTokens
    embed <- gets _embed
    remain <- gets _remain

    assign _lastTokens' $ tail lastTokens ++ [token]
    assign _embed' $ embed ++ [token]
    assign _inputEcho' True
    assign _remain' $ remain - 1

    -- cleanup
    liftIO $ do
        free candidates'
        free candidates

forward :: LLaMAContext -> Eval ()
forward LLaMAContext {..} = do
    nTokens <- gets _nTokens
    consumed <- gets _consumed
    embed <- gets _embed
    lastTokens <- gets _lastTokens

    when ((nTokens > consumed) && (length embed < (Parameters.nBatch args))) $ do
        let Tokens (tape, _) = tokens
        token <- liftIO $ peekElemOff tape consumed
        assign _lastTokens' $ tail lastTokens ++ [token]
        assign _consumed' $ consumed + 1

predict
    :: LLaMAContext
    -> Eval ()
predict LLaMAContext {..} = do
    let ctxLength = Parameters.nCtx args
    embed <- gets _embed
    when (length embed > ctxLength - 4) $
        assign _embed' $
            take (ctxLength - 4) embed

    past <- gets _past
    when (past + length embed > ctxLength) $ do
        lastTokens <- gets _lastTokens
        embed <- gets _embed
        let nLeft = past - (nKeep args)
            past' = max past (nKeep args)
            -- last_n_tokens.begin() + n_ctx - n_left/2 - embd.size()
            -- last_n_tokens.end() - embd.size()
            offset = ctxLength - nLeft `div` 2 - length embed
            size = length lastTokens - length embed - offset
            embed' = take size (drop offset lastTokens) ++ embed
        assign _past' past'
        assign _embed' embed'

    -- evaluate tokens in batches
    embed <- gets _embed
    past <- gets _past
    past <- liftIO $ eval embed past

    assign _past' past
    assign _embed' []
  where
    eval :: [Token] -> Int -> IO Int
    eval embed past = do
        let (batch, others) = splitAt (Parameters.nBatch args) embed
        withArrayLen batch $ \nEval batch' -> do
            err <-
                llama_eval
                    context
                    batch'
                    (fromIntegral nEval)
                    (fromIntegral past)
                    (fromIntegral (nThreads args))
            when (err /= 0) $
                fail "Failed to evaluate tokens"
            if null others
                then return (past + nEval)
                else eval others (past + nEval)

processEnd :: LLaMAContext -> Eval ()
processEnd LLaMAContext {..} = return ()

--     eos <- llama_token_eos context
--   where
--     do
--     interacting <- when (last lastTokens == eos) $ do

tokenToPiece :: LLaMAContext -> Token -> IO String
tokenToPiece LLaMAContext {..} token = do
    piece <- llama_token_to_piece context token buffer 1024
    peekCStringLen (buffer, fromIntegral piece)

printTimings :: LLaMAContext -> IO ()
printTimings LLaMAContext {..} = llama_print_timings context
