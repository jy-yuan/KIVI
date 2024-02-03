import transformers
import torch
from lm_eval.api.model import LM
import fnmatch


class LMEvalAdaptor(LM):

    def __init__(self, model_name, model, tokenizer, batch_size=1, max_length=-1):
        super().__init__()

        assert isinstance(batch_size, int)

        self.model_name = model_name
        self.model = model
        self.model.eval()

        self.tokenizer = tokenizer

        # assert isinstance(self.tokenizer, (
        #     transformers.GPT2Tokenizer, transformers.GPT2TokenizerFast,
        #     transformers.T5Tokenizer, transformers.T5TokenizerFast,
        # )), "this tokenizer has not been checked for compatibility yet!"

        self.vocab_size = self.tokenizer.vocab_size

        self._batch_size = batch_size

        self._max_length = max_length

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length != -1:
            return self._max_length
        if hasattr(self.model.config, 'n_ctx'):
            return self.model.config.n_ctx
        elif hasattr(self.model.config, 'max_position_embeddings'):
            return self.model.config.max_position_embeddings
        elif hasattr(self.model.config, 'n_positions'):
            return self.model.config.n_positions
        elif 'bloom' in self.model_name:
            return 2048
        elif 'llama' in self.model_name:
            return 2048  # TODO: did not check this
        elif 'mpt' in self.model_name:
            return 2048
        elif 'falcon' in self.model_name:
            return 2048
        else:
            print(self.model.config)
            raise NotImplementedError

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return "cuda"

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            if isinstance(self.model, transformers.models.t5.modeling_t5.T5ForConditionalGeneration):
                dec_inps = torch.cat(
                    [
                        torch.tensor(
                            self.model.generation_config.decoder_start_token_id,
                        )
                        .tile(len(inps), 1)
                        .to(inps),
                        inps,
                    ],
                    dim=1,
                )
             
                kwargs = {"decoder_input_ids": dec_inps,}
            else:
                kwargs = {}
            out = self.model(inps, **kwargs)[0]
            if "opt" in self.model_name:  # there are a few extra tokens in opt, which we should omit
                return out[:, :, :50257]
            else:
                return out  # [:, :, :self.tokenizer.vocab_size]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context,
            max_length=max_length,
            eos_token_id=eos_token_id,
            do_sample=False
        )

    def loglikelihood(self, requests):
        new_reqs = []
        for context, continuation in requests:
            if context == "":
                # end of text as context
                context_enc, continuation_enc = [self.eot_token_id], self.tok_encode(
                    continuation
                )
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(self, requests):
        # TODO: Implement caching once we've confirmed the perplexity implementation

        # automatic batch size detection for vectorization
        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size

        loglikelihoods = []
        for (string,) in tqdm(requests):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            # TODO: extract out this call so it only gets called once and also somehow figure out partial caching for
            # that
            string_nll = self._loglikelihood_tokens(
                rolling_token_windows,
                disable_tqdm=True,
                override_bs=adaptive_batch_size,
            )

            # discard is_greedy
            string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)

        return loglikelihoods

    def _loglikelihood_tokens(self, requests, disable_tqdm=False, override_bs=None):
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = utils.Reorderer(requests, _collate)

        reordered_requests = re_ord.get_reordered()
        n_reordered_requests = len(reordered_requests)

        # automatic (variable) batch size detection for vectorization
        # pull longest context sample from request
        def _batch_scheduler(pos):
            sched = pos // int(n_reordered_requests / self.batch_schedule)
            if sched in self.batch_sizes:
                return self.batch_sizes[sched]
            print(
                f"Passed argument batch_size = auto:{self.batch_schedule}. Detecting largest batch size"
            )
            self.batch_sizes[sched] = self._detect_batch_size(reordered_requests, pos)
            print(f"Determined largest batch size: {self.batch_sizes[sched]}")
            return self.batch_sizes[sched]

        for chunk in utils.chunks(
            tqdm(reordered_requests, disable=disable_tqdm),
            n=self.batch_size
            if self.batch_size != "auto"
            else override_bs
            if override_bs is not None
            else 0,
            fn=_batch_scheduler
            if self.batch_size == "auto"
            and n_reordered_requests > 0
            and not override_bs
            else None,
        ):
            inps = []
            cont_toks_list = []
            inplens = []

            padding_length = None

            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works:
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # gpt2    \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                    dtype=torch.long,
                ).to(self.device)
                (inplen,) = inp.shape

                cont = continuation_enc

                # since in _collate we make sure length is descending, the longest is always the first one.
                padding_length = (
                    padding_length if padding_length is not None else inplen
                )

                # pad length from seq to padding_length
                inp = torch.cat(
                    [
                        inp,  # [seq]
                        torch.zeros(padding_length - inplen, dtype=torch.long).to(
                            inp.device
                        ),  # [padding_length - seq]
                    ],
                    dim=0,
                )

                inps.append(inp.unsqueeze(0))  # [1, padding_length]
                cont_toks_list.append(cont)
                inplens.append(inplen)

            batched_inps = torch.cat(inps, dim=0)  # [batch, padding_length]
            multi_logits = F.log_softmax(
                self._model_call(batched_inps), dim=-1
            ).cpu()  # [batch, padding_length, vocab]

            for (cache_key, _, _), logits, inp, inplen, cont_toks in zip(
                chunk, multi_logits, inps, inplens, cont_toks_list
            ):

                # Slice to original seq length
                contlen = len(cont_toks)
                inplen = inplen + (
                    logits.shape[0] - padding_length
                )  # if "virtual tokens" (from prompt tuning) are added, inplen is larger
                logits = logits[inplen - contlen : inplen].unsqueeze(
                    0
                )  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)
                cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(
                    0
                )  # [1, seq]
                max_equal = (greedy_tokens == cont_toks).all()

                # Obtain log-probs at the corresponding continuation token indices
                # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                    -1
                )  # [1, seq]

                # Answer: (log prob, is-exact-match)
                answer = (float(logits.sum()), bool(max_equal))

                # partial caching
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)

                res.append(answer)

        return re_ord.get_original(res)

    def generate_until(self, requests):
        # TODO: implement fully general `until` that handles until that are
        #       multiple tokens or that span multiple tokens correctly

        # TODO: extract to TokenizedLM?
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        re_ord = utils.Reorderer(requests, _collate)

        warn_stop_seq = False
        for context, request_args in tqdm(re_ord.get_reordered()):
            until = request_args["until"]
            if isinstance(until, str):
                until = [until]

            if until:
                try:
                    (primary_until,) = self.tok_encode(until[0])
                except ValueError:
                    if not warn_stop_seq:
                        print(
                            "Warning: a primary stop sequence is multi-token! Will default to EOS token for this tokenizer. Consider using `hf-causal-experimental` for multi-token stop sequence support for the time being."
                        )
                        warn_stop_seq = True
                    primary_until = self.eot_token_id
            else:
                primary_until = None

            context_enc = torch.tensor(
                [self.tok_encode(context)[self.max_gen_toks - self.max_length :]]
            ).to(self.device)

            max_gen_tokens = min(
                self.max_gen_toks, request_args.get("max_length", self.max_gen_toks)
            )
            cont = self._model_generate(
                context_enc, context_enc.shape[1] + max_gen_tokens, primary_until
            )

            s = self.tok_decode(cont[0].tolist()[context_enc.shape[1] :])

            for term in until:
                s = s.split(term)[0]

            # partial caching
            self.cache_hook.add_partial("greedy_until", (context, until), s)

            res.append(s)

        return re_ord.get_original(res)

