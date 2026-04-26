import asyncio
from typing import Literal, List, Set, Tuple

from llama_index.core.llms import ChatResponseAsyncGen
from llama_index.core.memory import FactExtractionMemoryBlock, Memory
from llama_index.core.prompts import ChatMessage, MessageRole, PromptTemplate
from llama_index.core.retrievers import VectorIndexAutoRetriever, VectorIndexRetriever
from llama_index.core.schema import Document, NodeWithScore, QueryBundle
from llama_index.core.vector_stores.types import (
    VectorStoreQueryMode,
    VectorStoreQuerySpec,
)
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.utils.workflow import (
    draw_all_possible_flows,
    draw_all_possible_flows_mermaid,
)
from pydantic import BaseModel, Field

from ariadne.agent.prompts import (
    FACT_CONDENSE_PROMPT,
    FACT_EXTRACT_PROMPT,
    system_prompt,
)
from ariadne.core.config import settings
from ariadne.core.dependencies import (
    embed_model,
    fact_llm,
    gemini_tokenizer,
    get_qdrant_index,
    get_semantic_cache,
    llm,
    lite_llm,
    vector_info,
)
from ariadne.core.logger import session_context, setup_logger
from ariadne.core.tracing import init_phoenix_tracing

init_phoenix_tracing()
logger, log_listener = setup_logger()


class SmartAutoRetriever(VectorIndexAutoRetriever):
    """
    A custom wrapper around `VectorIndexAutoRetriever` that returns BOTH
    the nodes and the generated `VectorStoreQuerySpec`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def aretrieve_with_spec(
        self, str_or_query_bundle: str | QueryBundle
    ) -> Tuple[List[NodeWithScore], VectorStoreQuerySpec]:

        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(query_str=str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle

        spec = await self.agenerate_retrieval_spec(query_bundle)
        retriever, _ = self._build_retriever_from_spec(spec=spec)

        nodes = await retriever.aretrieve(query_bundle)

        return nodes, spec

# ==========================================
#      Models for Structured output
# ==========================================
class RouteDecision(BaseModel):
    reasoning: str = Field(
        description="Μια σύντομη εξήγηση (concise rationale) για την απόφαση δρομολόγησης."
    )
    route: Literal["rag", "general"] = Field(
        description="Επίλεξε 'rag' αν το ερώτημα αφορά το πανεπιστήμιο, τα μαθήματα, τον οδηγό σπουδών, ανακοινώσεις, επικοινωνία με διδάσκοντες. Επίλεξε 'general' για χαιρετισμούς, άσχετες ερωτήσεις ή small talk."
    )


class RouteAndCondenseDecision(BaseModel):
    reasoning: str = Field(
        description="Μια σύντομη εξήγηση (concise rationale) για την απόφαση δρομολόγησης και σύνοψης (contextualization) του ερωτήματος"
    )
    route: Literal["rag", "general"] = Field(
        description="Επίλεξε 'rag' αν το ερώτημα αφορά το πανεπιστήμιο, τα μαθήματα, τον οδηγό σπουδών, ανακοινώσεις, επικοινωνία με διδάσκοντες. Επίλεξε 'general' για χαιρετισμούς, small talk."
    )
    condensed_query: str = Field(
        description="If 'rag', rewrite the query using conversation history to make it standalone. If 'general', leave as is."
    )

class RelevanceEvaluation(BaseModel):
    reasoning: str = Field(
        description="Σύντομη εξήγηση για το αν τα έγγραφα περιέχουν την απάντηση."
    )
    is_relevant: bool = Field(
        description="True αν τα έγγραφα μπορούν να απαντήσουν την ερώτηση, αλλιώς False."
    )

class RewriteQuery(BaseModel):
    reasoning: str = Field(
        description="Σύντομη εξήγηση (concise rationale) για το γιατί η νέα ερώτηση είναι καλύτερη και τι άλλαξε σε σχέση με την παλιά."
    )
    new_query: str = Field(
        description="Η νέα, αναδιατυπωμένη ερώτηση. Πρέπει να είναι γενική, καθαρή και να χρησιμοποιεί συνώνυμα."
    )

# ==========================================
#                 Events
# ==========================================
class RouteEvent(Event):
    query: str

class RetrieveEvent(Event):
    query: str


class CheckCacheEvent(Event):
    condensed_query: str

class SynthesizeEvent(Event):
    query: str
    nodes: List[NodeWithScore]

class SynthesizeGeneralEvent(Event):
    query: str

class RewriteQueryEvent(Event):
    query: str

class SynthesizeFallbackEvent(Event):
    query: str

class UIProgressEvent(Event):
    """Event that emits the processing event of the workflow to the UI"""

    step_name: str = Field(description="The current step of the workflow.")
    msg: str = Field(description="Message to show to the UI.")

# ==========================================
#               Workflow
# ==========================================
class RAGWorkflow(Workflow):

    def __init__(self, session_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session_id = session_id

        self.llm = llm
        self.lite_llm = lite_llm
        self.embed_model = embed_model

        self.index = get_qdrant_index()
        self.cache_index, self.cache_retriever = get_semantic_cache()

        self.auto_index_retriever = SmartAutoRetriever(
            index=self.index,
            vector_store_info=vector_info,
            llm=self.lite_llm,
            max_top_k=20,
            similarity_top_k=settings.similarity_top_k,
            empty_query_top_k=20,
            vector_store_query_mode=VectorStoreQueryMode.HYBRID,
            alpha=settings.hybrid_search_alpha,  # 0.7
            sparse_top_k=settings.similarity_top_k,
            hybrid_top_k=settings.similarity_top_k,
            embed_model=embed_model,
            verbose=False,
        )

        # fall back retriever
        self.index_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=settings.similarity_top_k,
            vector_store_query_mode=VectorStoreQueryMode.HYBRID,
            alpha=settings.hybrid_search_alpha,
            sparse_top_k=settings.similarity_top_k,
            hybrid_top_k=settings.similarity_top_k,
            embed_model=embed_model,
            verbose=False,
        )

        self._fact_memory_block = FactExtractionMemoryBlock(
            priority=1,
            llm=fact_llm,
            max_facts=20,
            fact_extraction_prompt_template=FACT_EXTRACT_PROMPT,
            fact_condense_prompt_template=FACT_CONDENSE_PROMPT,
        )

        self.memory = Memory(
            token_limit=10000,
            memory_blocks=[self._fact_memory_block],
            tokenizer_fn=gemini_tokenizer,
        )

        self.max_retries = 2

    async def _stream_response_and_update_state(
        self,
        raw_stream: ChatResponseAsyncGen,
        role: MessageRole,
        on_complete_callback=None,
    ):
        """DRY helper method to manage streaming response generation and state updates."""
        full_response = ""
        async for chunk in raw_stream:
            full_response += chunk.delta or ""
            yield chunk

        await self.memory.aput(ChatMessage(role=role, content=full_response))
        if on_complete_callback:
            await on_complete_callback(full_response)

    # ==========================================
    #                 STEPS
    # ==========================================

    @step
    async def initialize_session(self, ctx: Context, ev: StartEvent) -> RouteEvent:
        """Setup context and history. Pass the raw query to the router."""
        # set the context var for logging
        session_context.set(self.session_id)

        user_msg = ev.user_msg
        await ctx.store.set("query", user_msg)
        await ctx.store.set("retries", 0)
        await self.memory.aput(ChatMessage(role=MessageRole.USER, content=user_msg))

        logger.info("Initializing new session", extra={"user_query": user_msg})
        return RouteEvent(query=user_msg)

    @step
    async def route_and_condense_query(
        self, ctx: Context, ev: RouteEvent
    ) -> CheckCacheEvent | SynthesizeGeneralEvent:
        """Combines routing, history condensation and semantic caching."""

        chat_history = await self.memory.aget()
        # conv history without latest message
        conversation_history = [
            m for m in chat_history if m.role != MessageRole.SYSTEM
        ][:-1]
        # last 4 messages
        recent_history = conversation_history[-4:]
        history_str = "\n".join(
            [f"{msg.role.value}: {msg.content}" for msg in recent_history]
        )

        ctx.write_event_to_stream(
            UIProgressEvent(
                step_name="Δρομολόγηση Ερωτήματος",
                msg="Γίνεται αναζήτηση στα αρχεία του τμήματος",
            )
        )

        prompt = PromptTemplate(
            "Αξιολόγησε το ερώτημα. Αν αφορά τη σχολή ('rag'), χρησιμοποίησε το ιστορικό για να φτιάξεις ένα ανεξάρτητο ερώτημα.\n"
            "Ιστορικό:\n{history}\nΕρώτημα: {query}"
        )

        decision: RouteAndCondenseDecision = await self.lite_llm.astructured_predict(
            output_cls=RouteAndCondenseDecision,
            prompt=prompt,
            history=history_str,
            query=ev.query,
        )
        logger.info(
            "Routing decision made",
            extra={
                "route": decision.route,
                "condensed_query": decision.condensed_query,
                "reasoning": decision.reasoning,
            },
        )
        await ctx.store.set("condensed_query", decision.condensed_query)

        # Handle General Route
        if decision.route.strip().lower() == "general":
            return SynthesizeGeneralEvent(query=ev.query)

        return CheckCacheEvent(condensed_query=decision.condensed_query)

    @step
    async def check_semantic_cache(
        self, ctx: Context, ev: CheckCacheEvent
    ) -> RetrieveEvent | StopEvent:
        if self.cache_retriever:
            cache_results = await self.cache_retriever.aretrieve(ev.condensed_query)

            if (
                cache_results
                and cache_results[0].score > settings.semantic_cache_threshold
            ):
                cached_answer = cache_results[0].metadata.get("answer")

                ctx.write_event_to_stream(
                    UIProgressEvent(
                        step_name="Semantic Cache",
                        msg="Η απάντηση ανασύρθηκε άμεσα από την προσωρινή μνήμη (Cache)!",
                    )
                )

                logger.info(
                    "Cache Hit",
                    extra={
                        "score": cache_results[0].score,
                        "cached_text": cache_results[0].text,
                    },
                )
                await self.memory.aput(
                    ChatMessage(role=MessageRole.SYSTEM, content=cached_answer)
                )

                class FakeChunk:
                    """Workaround in order to mimic streaming to the UI"""

                    def __init__(self, text: str):
                        self.delta = text

                async def cached_response_generator():
                    for word in cached_answer.split(" "):
                        yield FakeChunk(word + " ")
                        await asyncio.sleep(
                            0.005
                        )  # make streaming more natural and not so bursty

                return StopEvent(result=cached_response_generator())
        logger.info("Cache Miss")
        # If Cache misses, proceed to Retrieval
        return RetrieveEvent(query=ev.condensed_query)

    @step
    async def retrieve_and_evaluate(
        self, ctx: Context, ev: RetrieveEvent
    ) -> SynthesizeEvent | RewriteQueryEvent | SynthesizeFallbackEvent:
        """Fetch documents, handle fallback retrieval, and filter logic."""

        ctx.write_event_to_stream(
            UIProgressEvent(
                step_name="Αναζήτηση", msg=f"Ανάκτηση από τα έγγραφα του τμήματος"
            )
        )
        # GEMINI EMBEDDING 2 PREFIX 
        # https://ai.google.dev/gemini-api/docs/embeddings#task-types-embeddings-2
        gemini_search_str = f"task: question answering | query: {ev.query}"

        query_bundle = QueryBundle(
            query_str=ev.query,
            custom_embedding_strs=[gemini_search_str] # use the prefixed query for embedding the query
        )
        # Try Auto Retriever
        nodes, spec = await self.auto_index_retriever.aretrieve_with_spec(query_bundle)

        # Try Fallback Retriever if Auto failed
        if not nodes:
            logger.warning(
                "AutoRetriever failed. Falling back to VectorIndexRetriever."
            )
            nodes = await self.index_retriever.aretrieve(ev.query)
            spec = None

        # Check if both failed -> Retry Logic
        if not nodes:
            return await self._handle_retry(ctx, ev.query)

        await ctx.store.set("top_node_score", nodes[0].score)
        used_filters = spec and hasattr(spec, "filters") and len(spec.filters) > 0

        if used_filters:
            filter_details = ", ".join(
                [f"{f.key} {f.operator.value} {f.value}" for f in spec.filters]
            )
            logger.info(
                "AutoRetriever used metadata filters",
                extra={"filters": spec.filters, "filters_details": filter_details},
            )
            valid_nodes = nodes  # No score threshold if filters are heavily applied
        else:

            valid_nodes = [n for n in nodes if n.score >= settings.semantic_threshold][
                : settings.nodes_sent_to_llm
            ]
            logger.debug(
                "Pure Semantic Search executed. No filters applied.",
                extra={
                    "semantic_threshold": settings.semantic_threshold,
                    "max_nodes_sent_to_llm": settings.nodes_sent_to_llm,
                    "valid_nodes": len(valid_nodes),
                },
            )
        if not valid_nodes:
            return await self._handle_retry(ctx, ev.query)

        return SynthesizeEvent(query=ev.query, nodes=valid_nodes)

    async def _handle_retry(
        self, ctx: Context, query: str
    ) -> RewriteQueryEvent | SynthesizeFallbackEvent:
        """Helper to manage retry logic state cleanly."""
        current_retries = await ctx.store.get("retries", default=0)
        if current_retries < self.max_retries:
            await ctx.store.set("retries", current_retries + 1)
            return RewriteQueryEvent(query=query)
        return SynthesizeFallbackEvent(query=query)

    @step
    async def rewrite_query(self, ctx: Context, ev: RewriteQueryEvent) -> RetrieveEvent:
        """Rewrite query if retrieval failed."""
        rewrite_prompt = PromptTemplate(
            "Το παρακάτω ερώτημα δεν έφερε σωστά αποτελέσματα από τη βάση δεδομένων της Σχολής.\n"
            "Αναδιατύπωσε το ερώτημα ώστε να είναι πιο γενικό, ή χρησιμοποίησε συνώνυμα για καλύτερη αναζήτηση.\n"
            "Αφαίρεσε πολύπλοκους όρους και κράτα την ουσία.\n\n"
            "Αρχικό Ερώτημα: {query}"
        )

        rewritten_data: RewriteQuery = await self.lite_llm.astructured_predict(
            output_cls=RewriteQuery, prompt=rewrite_prompt, query=ev.query
        )

        new_query = rewritten_data.new_query.strip()
        logger.debug(
            "Query rewritten due to retrieval failure",
            extra={
                "old_query": ev.query,
                "new_query": new_query,
                "reasoning": rewritten_data.reasoning,
            },
        )
        return RetrieveEvent(query=new_query)

    @step
    async def synthesize(self, ctx: Context, ev: SynthesizeEvent) -> StopEvent:
        """Final generation using RAG context."""
        ctx.write_event_to_stream(
            UIProgressEvent(
                step_name="Σύνθεση απάντησης", msg="Γίνεται σύνθεση της απάντησης."
            )
        )
        context_str = "\n\n".join([n.node.get_content() for n in ev.nodes])
        await ctx.store.set("retrieved_texts", [n.node.get_content() for n in ev.nodes])

        sys_message = ChatMessage(
            role=MessageRole.SYSTEM,
            content=f"{system_prompt}\n\nΣχετικές Πληροφορίες:\n{context_str}",
        )

        chat_history = await self.memory.aget()

        messages = [sys_message] + chat_history
        raw_response_stream = await self.llm.astream_chat(messages)

        async def cache_callback(full_response: str):
            """Callback to save to Semantic Cache if criteria are met."""
            is_standalone = len(chat_history) <= 2
            top_node_score = await ctx.store.get("top_node_score", default=0)
            condensed_query = await ctx.store.get("condensed_query")

            if self.cache_retriever and top_node_score >= 0.7 and is_standalone:
                logger.info(
                    "Saving query to Cache", extra={"top_node_score": top_node_score}
                )
                cache_doc = Document(
                    text=condensed_query,
                    metadata={"answer": full_response},
                    excluded_embed_metadata_keys=["answer"],
                    excluded_llm_metadata_keys=["answer"],
                )
                # non blocking insertion
                task = asyncio.create_task(self.cache_index.ainsert(cache_doc))

                # create a reference, so it's not getting cleaned by garbage collector before inseted to Redis
                bg_tasks: Set = await ctx.store.get("bg_tasks", default=set())
                bg_tasks.add(task)
                task.add_done_callback(bg_tasks.discard)  # Clean up when done
                await ctx.store.set("bg_tasks", bg_tasks)

            else:
                logger.info(
                    "Query not saved to Cache", extra={"top_node_score": top_node_score}
                )

        stream_gen = self._stream_response_and_update_state(
            raw_response_stream,
            MessageRole.ASSISTANT,
            on_complete_callback=cache_callback,
        )

        return StopEvent(result=stream_gen)

    @step
    async def synthesize_general(
        self, ctx: Context, ev: SynthesizeGeneralEvent
    ) -> StopEvent:
        """General conversational generation."""

        await ctx.store.set("retrieved_texts", [])
        sys_message = ChatMessage(
            role=MessageRole.SYSTEM,
            content=f"{system_prompt}\n\nΟΔΗΓΙΑ: Απάντησε φυσικά, δίχως να ψάξεις στα έγγραφα. Αν σε ρωτάνε κάτι άσχετο με τη σχολή, υπενθύμισε ευγενικά τον ρόλο σου.",
        )

        chat_history = await self.memory.aget()
        messages = [sys_message] + chat_history
        raw_response_stream = await self.llm.astream_chat(messages)

        stream_gen = self._stream_response_and_update_state(
            raw_response_stream, MessageRole.ASSISTANT
        )

        return StopEvent(result=stream_gen)

    @step
    async def synthesize_fallback(
        self, ctx: Context, ev: SynthesizeFallbackEvent
    ) -> StopEvent:
        """Apologetic response when RAG completely fails."""

        await ctx.store.set("retrieved_texts", [])
        sys_message = ChatMessage(
            role=MessageRole.SYSTEM,
            content=f"{system_prompt}\n\nΟΔΗΓΙΑ: Ενημέρωσε τον χρήστη ευγενικά ότι παρά τις προσπάθειες αναζήτησης, δεν μπόρεσες να βρεις τη συγκεκριμένη πληροφορία.",
        )
        chat_history = await self.memory.aget()
        messages = [sys_message] + chat_history
        raw_response_stream = await self.llm.astream_chat(messages)

        stream_gen = self._stream_response_and_update_state(
            raw_response_stream, MessageRole.ASSISTANT
        )
        return StopEvent(result=stream_gen)


async def draw_workflow():

    workflow = RAGWorkflow(session_id="test", timeout=120)

    draw_all_possible_flows(workflow)
    draw_all_possible_flows_mermaid(workflow)
    del workflow


if __name__ == "__main__":
    asyncio.run(draw_workflow())
