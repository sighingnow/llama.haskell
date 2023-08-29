{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE RecordWildCards #-}

module Main (main) where

import AI.LanguageModel.LLaMA
import AI.LanguageModel.LLaMA qualified as LLaMA
import AI.LanguageModel.LLaMA.FFI qualified as LLaMA
import AI.LanguageModel.LLaMA.Parameters
import Control.Exception
import Control.Monad
import Control.Monad.State
import GHC.Conc (numCapabilities)
import Lens.Micro.Mtl
import Options.Applicative
import System.Random (randomIO)
import Text.Printf

-- import AI.LanguageModel.LLaMA.FFI

main :: IO ()
main = do
    args <- parseOpts
    bracket (initLLaMAContext args) deinitLLaMAContext $ \ctx -> do
        print $ "runLLaMA: " ++ (show (LLaMA.args ctx))

        -- eval <- prepare ctx
        -- (r, eval') <- runEval eval $ do
        --     forward ctx
        -- print $ "after prepare runLLaMA: " ++ (show eval')

        eval <- prepare ctx
        print $ "after prepare runLLaMA: " ++ (show eval)

        runEval eval $ runLLaMA ctx
        print $ "after eval runLLaMA: " ++ (show eval)

        printTimings ctx

runLLaMA :: LLaMAContext -> Eval ()
runLLaMA ctx@LLaMAContext {..} = do
    nTokens <- liftIO $ tokenize ctx (prompt args)
    assign _nTokens' nTokens
    liftIO $ print $ "runLLaMA: nTokens = " ++ show nTokens
    go
  where
    go :: Eval ()
    go =
        gets _remain >>= \case
            0 -> return ()
            _ -> evalOnce ctx >> go

log :: PrintfType r => String -> r
log fmt = printf ("[info]" ++ fmt)

warn :: PrintfType r => String -> r
warn fmt = printf ("[warn]" ++ fmt)

error :: PrintfType r => String -> r
error fmt = printf ("[error]" ++ fmt)

parseOpts :: IO Parameters
parseOpts = execParser opts >>= tweak
  where
    opts :: ParserInfo Parameters
    opts =
        info
            (parser <**> helper)
            ( fullDesc
                <> progDesc "llama-haskell: Haskell bindings for llama.cpp"
                <> header "llama-haskell: Evaluate your LLM models (in gguf format)"
            )

    tweak :: Parameters -> IO Parameters
    tweak args = do
        let nCtx' =
                if nCtx args < 8
                    then 8
                    else nCtx args
        prompt' <-
            if randomPrompt args
                then genRandomPrompt
                else return (prompt args)
        let interactiveFirst' =
                if instruct args
                    then True
                    else interactiveFirst args
        let interactive' =
                if interactiveFirst'
                    then True
                    else interactive args
        return $
            args
                { nCtx = nCtx'
                , prompt = prompt'
                , interactiveFirst = interactiveFirst'
                , interactive = interactive'
                }

    genRandomPrompt :: IO String
    genRandomPrompt = do
        index <- (`mod` 10) <$> randomIO :: IO Int
        return $ case index of
            0 -> "So"
            1 -> "Once upon a time"
            2 -> "When"
            3 -> "The"
            4 -> "After"
            5 -> "If"
            6 -> "import"
            7 -> "He"
            8 -> "She"
            9 -> "They"
            _ -> "To"

parser :: Parser Parameters
parser =
    Parameters
        <$> option
            auto
            ( short 's'
                <> long "seed"
                <> metavar "SEED"
                <> showDefault
                <> value (-1)
                <> help "RNG seed"
            )
        <*> option
            auto
            ( short 't'
                <> long "threads"
                <> metavar "N"
                <> showDefault
                <> value numCapabilities
                <> help "number of threads"
            )
        <*> option
            auto
            ( short 'n'
                <> long "n-predict"
                <> metavar "N"
                <> showDefault
                <> value (-1)
                <> help "new tokens to predict"
            )
        <*> option
            auto
            ( short 'c'
                <> long "ctx-size"
                <> metavar "N"
                <> showDefault
                <> value 512
                <> help "context size"
            )
        <*> option
            auto
            ( short 'b'
                <> long "batch-size"
                <> metavar "N"
                <> showDefault
                <> value 512
                <> help "batch size for prompt processing (must be >=32 to use BLAS)"
            )
        <*> option
            auto
            ( long "keep"
                <> metavar "N"
                <> showDefault
                <> value 0
                <> help "number of tokens to keep from initial prompt"
            )
        <*> option
            auto
            ( long "chunks"
                <> metavar "N"
                <> showDefault
                <> value (-1)
                <> help "max number of chunks to process (-1 = unlimited)"
            )
        <*> option
            auto
            ( long "n-gpu-layers"
                <> metavar "N"
                <> showDefault
                <> value 0
                <> help "number of layers to store in VRAM"
            )
        <*> option
            auto
            ( long "main-gpu"
                <> metavar "i"
                <> showDefault
                <> value 0
                <> help "the GPU that is used for scratch and small tensors"
            )
        <*> option
            auto
            ( long "tensor-split"
                <> metavar "SPLIT"
                <> showDefault
                <> value [0]
                <> help
                    "how split tensors should be distributed across GPUs, comma-separated list of proportions, e.g. 3,1"
            )
        <*> option
            auto
            ( long "n-probs"
                <> metavar "N"
                <> showDefault
                <> value 0
                <> help "if greater than 0, output the probabilities of top n_probs tokens."
            )
        <*> option
            auto
            ( long "n-beams"
                <> metavar "N"
                <> showDefault
                <> value 0
                <> help "if non-zero then use beam search of given width."
            )
        <*> option
            auto
            ( long "rope-freq-base"
                <> metavar "N"
                <> showDefault
                <> value 10000.0
                <> help "RoPE base frequency"
            )
        <*> option
            auto
            ( long "rope-freq-scale"
                <> metavar "N"
                <> showDefault
                <> value 1.0
                <> help "RoPE frequency scaling factor"
            )
        <*> option
            auto
            ( long "top-k"
                <> metavar "N"
                <> showDefault
                <> value 40
                <> help "top-k sampling, <= 0 to use vocab size"
            )
        <*> option
            auto
            ( long "top-p"
                <> metavar "N"
                <> showDefault
                <> value 0.95
                <> help "top-p sampling, 1.0 = disabled"
            )
        <*> option
            auto
            ( long "tfs"
                <> metavar "N"
                <> showDefault
                <> value 1.0
                <> help "tail free sampling, parameter z, 1.0 = disabled"
            )
        <*> option
            auto
            ( long "typical"
                <> metavar "N"
                <> showDefault
                <> value 1.0
                <> help "locally typical sampling, parameter p, 1.0 = disabled"
            )
        <*> option
            auto
            ( long "temp"
                <> metavar "N"
                <> showDefault
                <> value 0.80
                <> help "temperature, 1.0 = disabled"
            )
        <*> option
            auto
            ( long "repeat-penalty"
                <> metavar "N"
                <> showDefault
                <> value 1.10
                <> help "penalize repeat sequence of tokens, 1.0 = disabled"
            )
        <*> option
            auto
            ( long "repeat-last-n"
                <> metavar "N"
                <> showDefault
                <> value 64
                <> help "last n tokens to penalize, 0 = disable penalty, -1 = ctx_size"
            )
        <*> option
            auto
            ( long "frequency-penalty"
                <> metavar "N"
                <> showDefault
                <> value 0.00
                <> help "repeat alpha frequency penalty, 0.0 = disabled"
            )
        <*> option
            auto
            ( long "presence-penalty"
                <> metavar "N"
                <> showDefault
                <> value 0.00
                <> help "repeat alpha presence penalty, 0.0 = disabled"
            )
        <*> option
            auto
            ( long "mirostat"
                <> metavar "N"
                <> showDefault
                <> value 0
                <> help "use Mirostat sampling, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0"
            )
        <*> option
            auto
            ( long "mirostat-ent"
                <> metavar "N"
                <> showDefault
                <> value 5.00
                <> help "Mirostat target entropy, parameter tau"
            )
        <*> option
            auto
            ( long "mirostat-lr"
                <> metavar "N"
                <> showDefault
                <> value 0.10
                <> help "Mirostat learning rate, parameter eta"
            )
        <*> option
            auto
            ( long "logit-bias"
                <> metavar "TOKEN_ID(+/-)BIAS"
                <> showDefault
                <> value []
                <> help
                    "modifies the likelihood of token appearing in the completion, i.e. `--logit-bias 15043+1` to increase likelihood of token ' Hello', or `--logit-bias 15043-1` to decrease likelihood of token ' Hello'"
            )
        <*> strOption
            ( long "cfg-negative-prompt"
                <> metavar "PROMPT"
                <> showDefault
                <> value ""
                <> help "negative prompt to use for guidance"
            )
        <*> option
            auto
            ( long "cfg-scale"
                <> metavar "N"
                <> showDefault
                <> value 1.0
                <> help "strength of guidance, 1.0 = disable"
            )
        <*> strOption
            ( short 'm'
                <> long "model"
                <> metavar "FNAME"
                <> showDefault
                <> value "models/7B/ggml-model-f16.gguf"
                <> help "model path"
            )
        <*> strOption
            ( long "model-alias"
                <> metavar "ALIAS"
                <> showDefault
                <> value "unknown"
                <> help "model alias"
            )
        <*> strOption
            ( short 'p'
                <> long "prompt"
                <> metavar "PROMPT"
                <> showDefault
                <> value ""
                <> help "prompt to start generation with"
            )
        <*> strOption
            ( long "prompt-cache"
                <> metavar "FNAME"
                <> showDefault
                <> value ""
                <> help "file to cache prompt state for faster startup"
            )
        <*> strOption
            ( long "in-prefix"
                <> metavar "STRING"
                <> showDefault
                <> value ""
                <> help "string to prefix user inputs with"
            )
        <*> strOption
            ( long "in-suffix"
                <> metavar "STRING"
                <> showDefault
                <> value ""
                <> help "string to suffix after user inputs with"
            )
        <*> strOption
            ( long "grammar"
                <> metavar "GRAMMAR"
                <> showDefault
                <> value ""
                <> help "BNF-like grammar to constrain generations (see samples in grammars/ dir)"
            )
        <*> option
            auto
            ( long "antiprompt"
                <> metavar "PROMPT"
                <> showDefault
                <> value []
                <> help "negative prompt to use for guidance"
            )
        <*> strOption
            ( long "logdir"
                <> metavar "LOGDIR"
                <> showDefault
                <> value ""
                <> help "path under which to save YAML logs (no logging if unset)"
            )
        <*> strOption
            ( long "lora"
                <> metavar "FNAME"
                <> showDefault
                <> value ""
                <> help "apply LoRA adaptor (implies --no-mmap)"
            )
        <*> strOption
            ( long "lora-base"
                <> metavar "FNAME"
                <> showDefault
                <> value ""
                <> help
                    "optional model to use as a base for the layers modified by the LoRA adaptor"
            )
        <*> option
            auto
            ( long "ppl-stride"
                <> metavar "N"
                <> showDefault
                <> value 0
                <> help
                    "stride for perplexity calculations. If left at 0, the pre-existing approach will be used."
            )
        <*> option
            auto
            ( long "ppl-output-type"
                <> metavar "N"
                <> showDefault
                <> value 0
                <> help
                    "= 0 -> ppl output is as usual, = 1 -> ppl output is num_tokens, ppl, one per line (which is more convenient to use for plotting)"
            )
        <*> switch
            ( long "hellaswag"
                <> help
                    "compute HellaSwag score over random tasks from datafile supplied in prompt"
            )
        <*> option
            auto
            ( long "hellaswag-tasks"
                <> metavar "N"
                <> showDefault
                <> value 400
                <> help "number of tasks to use when computing the HellaSwag score"
            )
        <*> switch
            ( long "low-vram"
                <> help "don't allocate VRAM scratch buffer"
            )
        <*> flag
            True
            False
            ( long "no-mul-mat-q"
                <> help
                    ( "use GGML_CUBLAS_NAME instead of custom mul_mat_q GGML_CUDA_NAME kernels. "
                        ++ "Not recommended since this is both slower and uses more VRAM."
                    )
            )
        <*> switch
            ( long "memory-f32"
                <> help
                    "use f32 instead of f16 for memory key+value (default: disabled). Not recommended: doubles context memory required and no measurable increase in quality"
            )
        <*> switch
            ( long "random-prompt"
                <> help "start with a randomized prompt."
            )
        <*> switch
            ( long "color"
                <> help "colorise output to distinguish prompt and user input from generations"
            )
        <*> switch
            ( short 'i'
                <> long "interactive"
                <> help "run in interactive mode"
            )
        <*> switch
            ( long "prompt-cache-all"
                <> help
                    "if specified, saves user input and generations to cache as well. not supported with --interactive or other interactive options"
            )
        <*> switch
            ( long "prompt-cache-ro"
                <> help "if specified, uses the prompt cache but does not update it."
            )
        <*> switch
            ( long "embedding"
                <> help "get only sentence embedding"
            )
        <*> switch
            ( long "escape"
                <> help "process prompt escapes sequences (\\n, \\r, \\t, \\', \\\", \\\\)"
            )
        <*> switch
            ( long "interactive-first"
                <> help "run in interactive mode and wait for input right away"
            )
        <*> switch
            ( long "multiline-input"
                <> help "allows you to write or paste multiple lines without ending each in '\\'"
            )
        <*> switch
            ( long "simple-io"
                <> help
                    "use basic IO for better compatibility in subprocesses and limited consoles"
            )
        <*> switch
            ( long "in-prefix-bos"
                <> help "prefix BOS to user inputs, preceding the `--in-prefix` string"
            )
        <*> switch
            ( long "ignore-eos"
                <> help
                    "ignore end of stream token and continue generating (implies --logit-bias 2-inf)"
            )
        <*> switch
            ( long "instruct"
                <> help
                    "instruction mode (used for Alpaca models)"
            )
        <*> flag
            True
            False
            ( long "no-penalize-nl"
                <> help "do not penalize newline token"
            )
        <*> switch
            ( long "perplexity"
                <> help "compute perplexity over each ctx window of the prompt"
            )
        <*> flag
            True
            False
            ( long "no-mmap"
                <> help
                    "do not memory-map model (slower load but may reduce pageouts if not using mlock)"
            )
        <*> switch
            ( long "mlock"
                <> help "force system to keep model in RAM rather than swapping or compressing"
            )
        <*> switch
            ( long "men-test"
                <> help "compute maximum memory usage"
            )
        <*> switch
            ( long "numa"
                <> help
                    "attempt optimizations that help on some NUMA systems. if run without this previously, it is recommended to drop the system page cache before using this"
            )
        <*> switch
            ( long "export"
                <> help "export the computation graph to 'llama.ggml'"
            )
        <*> switch
            ( long "verbose-prompt"
                <> help "print prompt before generation"
            )
