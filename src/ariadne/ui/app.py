"""
Chainlit UI application for Ariadne.

This file defines the UI components and message handling logic for the
Ariadne AI assistant, using Chainlit to provide a chat interface.

Author: Georgios Giannopoulos
"""

import chainlit as cl
from time import perf_counter
from typing import Optional

from ariadne.agent.workflow import RAGWorkflow, UIProgressEvent
from ariadne.core.config import settings
from ariadne.core.logger import setup_logger, session_context

logger, log_listener = setup_logger()

@cl.on_chat_start
async def on_chat_start() -> None:
    """
    Initializes the session when a new chat starts.
    
    Sets up the RAGWorkflow and stores it in the user session.
    """

    session_id: str = cl.user_session.get("id") or "anonymous_session"

    workflow: RAGWorkflow = RAGWorkflow(session_id=session_id, timeout=120)

    cl.user_session.set("workflow", workflow)
    cl.user_session.set("session_id", session_id)
    
    logger.info("New Chainlit session started.", extra={"chainlit_session": session_id})

@cl.on_message
async def main(message: cl.Message) -> None:
    """
    Main message handler for incoming user messages.

    Runs the RAGWorkflow, streams progress events to the UI,
    and streams the final response.

    Args:
        message (cl.Message): The message object containing user input.
    """
    session_id: str = cl.user_session.get("session_id")
    workflow: Optional[RAGWorkflow] = cl.user_session.get("workflow")
    if not workflow:
        await cl.Message(content="Session error. Please refresh the page.", author="Ariadne").send()
        return
    
    session_context.set(session_id)

    try:
        start_time: float = perf_counter()
        logger.info("UI: Received user message", extra={"user_msg": message.content})

        if len(message.content) > settings.max_query_length:
            logger.warning(f"User query too long: {len(message.content)} chars")
            await cl.Message(
                content=f"Το μήνυμά σας είναι πολύ μεγάλο ({len(message.content)} χαρακτήρες). Παρακαλώ περιορίστε το ερώτημά σας στους {settings.max_query_length} χαρακτήρες.",
                author='Ariadne'
            ).send()
            return

        handler = workflow.run(user_msg=message.content)

        async with cl.Step(name='H Αριάδνη σκέφτεται...', type='run') as parent_step:
            async for event in handler.stream_events():
                if isinstance(event, UIProgressEvent):
                    async with cl.Step(name=event.step_name, type='tool', parent_id=parent_step.id) as child_step:
                        child_step.output = event.msg
                        await child_step.send()

            parent_step.name='Σκέψεις Αριάδνης'      
            await parent_step.update()
               
        response_stream = await handler
        
        msg: cl.Message = cl.Message(content="", author='Ariadne')
        await msg.send()

        first_token_flag: bool = False
        async for token in response_stream:
            if not first_token_flag:
                ttft: float = perf_counter() - start_time
                logger.info("First token generated", extra={"ttft_seconds": round(ttft, 3)})
                first_token_flag = True

            await msg.stream_token(token.delta or "")
        
        total_time: float = perf_counter() - start_time
        logger.info("Response completed", extra={"total_time_seconds": round(total_time, 3)})
        # msg.content += f"\n\n*(Χρόνος απόκρισης: {total_time:.2f} δευτερόλεπτα)*"
        await msg.update()

    except Exception as e:
        logger.error(f"Workflow failed during message processing: {str(e)}", exc_info=True)

        await cl.Message(
            content="Προέκυψε ένα σφάλμα κατά την επεξεργασία. Παρακαλώ δοκιμάστε ξανά.", 
            author='Ariadne'
        ).send()

@cl.on_chat_end
def on_chat_end() -> None:
    """
    Cleans up resources when a chat session ends.
    """
    session_id: str = cl.user_session.get("session_id")
    session_context.set(session_id) # ensure log knows who is disconnecting
    workflow: Optional[RAGWorkflow] = cl.user_session.get("workflow")

    if workflow:
        del workflow
    cl.user_session.set("workflow", None)

    logger.info("Session disconnected. Resources cleared.", extra={"chainlit_session": session_id})
